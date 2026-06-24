"""
Stage 1: Build present-day grid assemblages from RAINBIO occurrences.

Inputs
------
- rainbio_LHS_traits.csv (occurrence rows with trait columns attached)
- WorldClim present-day MAT (bio1) and MAP (bio12) rasters
- Tanzania bounding box (from config)

Outputs (to OUTPUT_DIR)
-----------------------
- species_pool.csv         : species kept after trait + record filters
- assemblage_present.csv   : long format (cell_id, species, weight) for Tanzania
- envelope_occurrences.csv : full-extent occurrences with MAT/MAP attached,
                             used in Stage 4 for tolerance envelopes
- grid_cells.csv           : cell_id, lon_centre, lat_centre, MAT_present, MAP_present

Design notes
------------
Two occurrence subsets are produced:
  1. Tanzania-clipped, used to build the present-day assemblage.
  2. Full-extent (e.g. all of Africa), used to estimate each species' realised
     climatic envelope. Using only Tanzanian records would artificially narrow
     envelopes and inflate apparent climate-change loss in Stage 4.

The `weight` column in assemblage_present.csv is 1.0 for every present species.
When SDM-based assemblages replace this stage, `weight` becomes the calibrated
suitability score and nothing downstream needs to change.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from config import (
    OUTPUT_DIR,
    WORLDCLIM_PRESENT_DIR,
    GRID_RES_DEG,
    TANZANIA_BBOX,
    EAST_AFRICA_BBOX,
    ENVELOPE_EXTENT,
    TRAIT_COLUMNS,
    REQUIRE_COMPLETE_TRAITS,
    MIN_RECORDS_PER_SPECIES,
    VERBOSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    if VERBOSE:
        print(f"[01_assemblages] {msg}")


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Find species / lon / lat columns tolerantly."""
    cols_lower = {c.lower(): c for c in df.columns}

    species_candidates = ["species", "tax_sp_level", "scientificname", "scientific_name", "    species"]
    lon_candidates = ["decimallongitude", "lon", "longitude", "x"]
    lat_candidates = ["decimallatitude", "lat", "latitude", "y"]

    def pick(candidates: list[str], label: str) -> str:
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        raise KeyError(
            f"Could not find a {label} column. Looked for {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return {
        "species": pick(species_candidates, "species"),
        "lon": pick(lon_candidates, "longitude"),
        "lat": pick(lat_candidates, "latitude"),
    }


def assign_cell_id(lon: np.ndarray, lat: np.ndarray, bbox: dict, res: float) -> np.ndarray:
    """Map lon/lat to integer cell ids over the bbox grid."""
    n_cols = int(round((bbox["max_lon"] - bbox["min_lon"]) / res))
    col = ((lon - bbox["min_lon"]) / res).astype(int)
    row = ((lat - bbox["min_lat"]) / res).astype(int)
    return row * n_cols + col


def extract_raster_values(
    raster_path,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Sample a single-band raster at lon/lat points."""
    with rasterio.open(raster_path) as src:
        coords = list(zip(lon, lat))
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        nodata = src.nodata
        if nodata is not None:
            vals[vals == nodata] = np.nan
    return vals


def build_grid_table(bbox: dict, res: float) -> pd.DataFrame:
    """Enumerate every cell in the bbox grid with its centre coordinates."""
    n_cols = int(round((bbox["max_lon"] - bbox["min_lon"]) / res))
    n_rows = int(round((bbox["max_lat"] - bbox["min_lat"]) / res))

    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows = rows.ravel()
    cols = cols.ravel()

    lon_centre = bbox["min_lon"] + (cols + 0.5) * res
    lat_centre = bbox["min_lat"] + (rows + 0.5) * res
    cell_id = rows * n_cols + cols

    return pd.DataFrame({
        "cell_id": cell_id,
        "lon_centre": lon_centre,
        "lat_centre": lat_centre,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load Stage 0 outputs ---------------------------------------------
    rainbio_path = OUTPUT_DIR / "rainbio_continental.csv"
    traits_path = OUTPUT_DIR / "traits_species.csv"
    for p in (rainbio_path, traits_path):
        if not p.exists():
            sys.exit(f"Missing Stage 0 output: {p}. Run 00_build_continental_traits.py first.")

    log(f"Reading {rainbio_path.name}")
    occ = pd.read_csv(rainbio_path)
    log(f"  {len(occ):,} occurrence rows")

    log(f"Reading {traits_path.name}")
    traits = pd.read_csv(traits_path)
    log(f"  {len(traits):,} species in trait table")



    # --- Build species pool ------------------------------------------------
    # Filter to species with complete LHS traits
    if REQUIRE_COMPLETE_TRAITS:
        traits = traits.dropna(subset=TRAIT_COLUMNS)
        log(f"  {len(traits):,} species with complete LHS traits")

    # Restrict occurrences to species that have (complete) traits
    df = occ[occ["species"].isin(traits["species"])].copy()
    log(f"  {len(df):,} occurrence rows after trait filter")

    # Filter 2: minimum records per species
    record_counts = df.groupby("species").size()
    reliable_species = record_counts[record_counts >= MIN_RECORDS_PER_SPECIES].index
    df = df[df["species"].isin(reliable_species)]
    log(f"  {len(reliable_species):,} species with >= {MIN_RECORDS_PER_SPECIES} records")
    log(f"  {len(df):,} occurrence rows in final pool")

    # Save the species pool (one row per species, with trait values from
    # Stage 0 and record counts from the filtered occurrence set).
    record_counts_df = (
        df.groupby("species").size().rename("n_records").reset_index()
    )
    species_pool = (
            traits[traits["species"].isin(reliable_species)]
            .merge(record_counts_df, on="species", how="left")
        )
    species_pool.to_csv(OUTPUT_DIR / "species_pool.csv", index=False)
    log(f"  Wrote species_pool.csv ({len(species_pool):,} species)")

    # --- Envelope occurrences (full extent) --------------------------------
    if ENVELOPE_EXTENT == "africa":
        env_df = df.copy()
    elif ENVELOPE_EXTENT == "east_africa":
        b = EAST_AFRICA_BBOX
        env_df = df[(df["lon"].between(b["min_lon"], b["max_lon"])) &
                    (df["lat"].between(b["min_lat"], b["max_lat"]))].copy()
    elif ENVELOPE_EXTENT == "tanzania":
        b = TANZANIA_BBOX
        env_df = df[(df["lon"].between(b["min_lon"], b["max_lon"])) &
                    (df["lat"].between(b["min_lat"], b["max_lat"]))].copy()
    else:
        sys.exit(f"Unknown ENVELOPE_EXTENT: {ENVELOPE_EXTENT}")

    log(f"  Envelope extent '{ENVELOPE_EXTENT}': {len(env_df):,} occurrences")

    # Attach present-day MAT (bio1) and MAP (bio12)
    bio1_path = WORLDCLIM_PRESENT_DIR / "wc2.1_10m_bio_1.tif"
    bio12_path = WORLDCLIM_PRESENT_DIR / "wc2.1_10m_bio_12.tif"
    log("  Sampling WorldClim bio1 (MAT) and bio12 (MAP) at occurrences")
    env_df["MAT"] = extract_raster_values(bio1_path, env_df["lon"].values, env_df["lat"].values)
    env_df["MAP"] = extract_raster_values(bio12_path, env_df["lon"].values, env_df["lat"].values)

    env_out = env_df[["species", "lon", "lat", "MAT", "MAP"]].dropna(subset=["MAT", "MAP"])
    env_out.to_csv(OUTPUT_DIR / "envelope_occurrences.csv", index=False)
    log(f"  Wrote envelope_occurrences.csv ({len(env_out):,} rows)")

    # --- Tanzania assemblage ----------------------------------------------
    b = TANZANIA_BBOX
    tz_df = df[(df["lon"].between(b["min_lon"], b["max_lon"])) &
               (df["lat"].between(b["min_lat"], b["max_lat"]))].copy()
    log(f"  Tanzania-clipped occurrences: {len(tz_df):,}")

    tz_df["cell_id"] = assign_cell_id(
        tz_df["lon"].values, tz_df["lat"].values, TANZANIA_BBOX, GRID_RES_DEG
    )

    # Long-format assemblage: one row per (cell, species), weight = 1.0
    assemblage = (
        tz_df[["cell_id", "species"]]
        .drop_duplicates()
        .assign(weight=1.0)
        .sort_values(["cell_id", "species"])
        .reset_index(drop=True)
    )
    assemblage.to_csv(OUTPUT_DIR / "assemblage_present.csv", index=False)

    n_cells_occupied = assemblage["cell_id"].nunique()
    n_species_in_tz = assemblage["species"].nunique()
    log(f"  Wrote assemblage_present.csv: {len(assemblage):,} rows, "
        f"{n_cells_occupied:,} occupied cells, {n_species_in_tz:,} species in TZ")

    # --- Grid cell table with present-day climate -------------------------
    grid = build_grid_table(TANZANIA_BBOX, GRID_RES_DEG)
    log(f"  Sampling present-day climate at {len(grid):,} cell centres")
    grid["MAT_present"] = extract_raster_values(
        bio1_path, grid["lon_centre"].values, grid["lat_centre"].values
    )
    grid["MAP_present"] = extract_raster_values(
        bio12_path, grid["lon_centre"].values, grid["lat_centre"].values
    )
    grid.to_csv(OUTPUT_DIR / "grid_cells.csv", index=False)
    log(f"  Wrote grid_cells.csv ({len(grid):,} cells)")

    log("Stage 1 complete.")


if __name__ == "__main__":
    main()