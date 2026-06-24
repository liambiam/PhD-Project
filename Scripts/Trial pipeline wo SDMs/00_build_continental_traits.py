"""
Stage 0: Build continental RAINBIO + TRY species-level trait table.

Reads
-----
- Raw RAINBIO continental CSV
- Raw TRY trait export (tab-separated)

Writes (to OUTPUT_DIR)
----------------------
- rainbio_continental.csv : occurrence-level (species, lon, lat) with
                            subspecies collapsed via tax_sp_level
- traits_species.csv      : one row per species with mean trait values
                            (SLA, height, seed_mass) drawn from TRY

Design notes
------------
- Subspecies/varieties are collapsed via RAINBIO's tax_sp_level column.
- TRY trait values are averaged per species across all matching records,
  including all SLA variants (petiole included/excluded/undefined), as a
  practical aggregation consistent with how the LHS framework is used
  in Falster et al. / Andrew et al.
- Species name matching uses normalised genus + specific epithet only
  (first two words, lowercased) to absorb authority strings and any
  remaining infraspecific noise.
- Only TRY rows with non-null StdValue are used (standardised units).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, VERBOSE

# ---------------------------------------------------------------------------
# Raw input paths (eventually move to config.py if you prefer)
# ---------------------------------------------------------------------------

RAINBIO_RAW_CSV = Path(
    "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/"
    "rainbio_published/published_database/RAINBIO.csv"
)
TRY_RAW_TXT = Path(
    "C:/Users/liams/Documents/PhD-Project Data/TRY/SLA, PH, SM/50190.txt"
)

# Canonical trait → list of regex patterns matched (case-insensitive) against
# TRY's TraitName column.
TRY_TRAIT_PATTERNS: dict[str, list[str]] = {
    "SLA":       [r"specific leaf area"],
    "height":    [r"plant height"],   # vegetative + generative
    "seed_mass": [r"seed dry mass"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    if VERBOSE:
        print(f"[00_build_continental] {msg}")


def normalise_name(s) -> str:
    """Lowercased genus + specific epithet, stripped and whitespace-collapsed.

    Used only as a join key; the original species name is preserved alongside.
    """
    if pd.isna(s):
        return ""
    words = re.sub(r"\s+", " ", str(s).strip().lower()).split()
    return " ".join(words[:2])


# ---------------------------------------------------------------------------
# RAINBIO
# ---------------------------------------------------------------------------

def load_rainbio() -> pd.DataFrame:
    log(f"Reading RAINBIO: {RAINBIO_RAW_CSV.name}")
    df = pd.read_csv(RAINBIO_RAW_CSV, low_memory=False)
    log(f"  {len(df):,} rows, {df.shape[1]} columns")

    required = ["tax_sp_level", "decimalLongitude", "decimalLatitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"RAINBIO missing expected columns: {missing}.\n"
            f"Available: {list(df.columns)}"
        )

    out = (
        df[["tax_sp_level", "decimalLongitude", "decimalLatitude"]]
        .rename(columns={
            "tax_sp_level": "species",
            "decimalLongitude": "lon",
            "decimalLatitude": "lat",
        })
        .copy()
    )
    out["species"] = out["species"].astype(str).str.strip()
    out = out[out["species"].ne("") & out["species"].ne("nan")]
    out = out.dropna(subset=["lon", "lat"])

    log(f"  {len(out):,} rows after dropping missing species/coords")
    log(f"  {out['species'].nunique():,} unique species (tax_sp_level)")
    return out


# ---------------------------------------------------------------------------
# TRY
# ---------------------------------------------------------------------------

def load_try() -> pd.DataFrame:
    log(f"Reading TRY: {TRY_RAW_TXT.name}")
    # TRY exports are tab-separated and frequently Latin-1 encoded.
    try:
        df = pd.read_csv(
            TRY_RAW_TXT,
            sep="\t",
            encoding="latin-1",
            low_memory=False,
            on_bad_lines="warn",
        )
    except Exception as e:
        sys.exit(f"Failed to read TRY file: {e}")

    log(f"  {len(df):,} rows, {df.shape[1]} columns")

    required = ["AccSpeciesName", "TraitName", "StdValue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(
            f"TRY file missing expected columns: {missing}.\n"
            f"Available: {list(df.columns)}"
        )

    # Show what's actually in the file so trait pattern coverage is visible.
    trait_counts = df["TraitName"].value_counts(dropna=True)
    log(f"  Unique TraitNames in file: {len(trait_counts)}")
    log("  Top 10 by row count:")
    for name, n in trait_counts.head(10).items():
        log(f"    {n:>8,}  {name}")

    return df


def classify_try_traits(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'canonical' column mapping each row to SLA/height/seed_mass."""
    df = df.copy()
    df["canonical"] = pd.NA

    for canonical, patterns in TRY_TRAIT_PATTERNS.items():
        for pat in patterns:
            mask = df["TraitName"].astype(str).str.contains(
                pat, case=False, na=False, regex=True
            )
            df.loc[mask, "canonical"] = canonical

    classified = df.dropna(subset=["canonical"])
    log(f"  Classified {len(classified):,} / {len(df):,} TRY rows:")
    for canonical in TRY_TRAIT_PATTERNS:
        sub = classified[classified["canonical"] == canonical]
        n_sp = sub["AccSpeciesName"].nunique()
        log(f"    {canonical:>10}: {len(sub):>8,} rows, {n_sp:>6,} species")
        # Show which TraitName(s) got caught for each canonical trait
        for name in sub["TraitName"].value_counts().head(3).index:
            log(f"        ↳ {name}")
    return classified


def aggregate_try_to_species(df: pd.DataFrame) -> pd.DataFrame:
    """Per-species mean StdValue per canonical trait, pivoted wide."""
    df = df.dropna(subset=["StdValue", "AccSpeciesName"]).copy()
    df["AccSpeciesName"] = df["AccSpeciesName"].astype(str).str.strip()
    df = df[df["AccSpeciesName"].ne("")]

    long = (
        df.groupby(["AccSpeciesName", "canonical"], as_index=False)["StdValue"]
          .mean()
    )
    wide = (
        long.pivot(index="AccSpeciesName", columns="canonical", values="StdValue")
            .reset_index()
            .rename(columns={"AccSpeciesName": "species"})
    )

    # Guarantee all three trait columns exist
    for col in TRY_TRAIT_PATTERNS:
        if col not in wide.columns:
            wide[col] = np.nan

    log(f"  Species × trait wide table: {len(wide):,} species")
    log("  Trait completeness:")
    for col in TRY_TRAIT_PATTERNS:
        n = wide[col].notna().sum()
        log(f"    {col:>10}: {n:>5,} / {len(wide):,} species "
            f"({100*n/len(wide):.1f}%)")

    complete = wide.dropna(subset=list(TRY_TRAIT_PATTERNS.keys()))
    log(f"  Species with all three LHS traits: {len(complete):,}")

    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rainbio = load_rainbio()
    try_raw = load_try()
    try_classified = classify_try_traits(try_raw)
    traits = aggregate_try_to_species(try_classified)

    # --- Match TRY species to RAINBIO species (normalised genus+epithet) ---
    log("Matching species names between TRY and RAINBIO")
    rainbio["_match"] = rainbio["species"].apply(normalise_name)
    traits["_match"] = traits["species"].apply(normalise_name)

    rainbio_set = set(rainbio["_match"].unique())
    try_set = set(traits["_match"].unique())
    overlap = rainbio_set & try_set
    log(f"  RAINBIO species (normalised): {len(rainbio_set):,}")
    log(f"  TRY species (any trait):     {len(try_set):,}")
    log(f"  Overlap:                     {len(overlap):,}")

    # Filter both sides to the overlap
    rainbio_matched = rainbio[rainbio["_match"].isin(overlap)].copy()
    traits_matched = traits[traits["_match"].isin(overlap)].copy()

    # Use the canonical RAINBIO name in the traits table (one name per match key).
    name_map = (
        rainbio_matched.drop_duplicates("_match")[["_match", "species"]]
        .rename(columns={"species": "species_rainbio"})
    )
    traits_matched = traits_matched.merge(name_map, on="_match", how="left")
    traits_matched["species"] = traits_matched["species_rainbio"].fillna(
        traits_matched["species"]
    )
    traits_matched = traits_matched.drop(columns=["_match", "species_rainbio"])
    rainbio_matched = rainbio_matched.drop(columns=["_match"])

    log(f"  Occurrences with at least one trait: {len(rainbio_matched):,}")

    # --- Trait-complete subset diagnostic --------------------------------
    complete_species = traits_matched.dropna(
        subset=list(TRY_TRAIT_PATTERNS.keys())
    )["species"].unique()
    n_occ_complete = rainbio_matched["species"].isin(complete_species).sum()
    log(f"  Trait-complete species: {len(complete_species):,}")
    log(f"  Occurrences for trait-complete species: {n_occ_complete:,}")

    # --- Write outputs ----------------------------------------------------
    traits_out = OUTPUT_DIR / "traits_species.csv"
    occ_out = OUTPUT_DIR / "rainbio_continental.csv"
    traits_matched.to_csv(traits_out, index=False)
    rainbio_matched.to_csv(occ_out, index=False)
    log(f"Wrote {traits_out.name}: {len(traits_matched):,} species")
    log(f"Wrote {occ_out.name}: {len(rainbio_matched):,} occurrences")
    log("Stage 0 complete.")


if __name__ == "__main__":
    main()