#!/usr/bin/env python3
###############################################################################
# tessera_sample_points.py  (geotessera 0.8.0)
# -----------------------------------------------------------------------------
# Sample TESSERA embeddings at the EXACT presence + background points produced
# by the R SDM script, so the TESSERA arm is evaluated on identical locations
# to WorldClim and AlphaEarth (fair-comparison spine).
#
# Why point-sampling, not a mosaic: a wall-to-wall 10 m TESSERA mosaic over
# Tanzania is ~1.6 TB (10,185 tiles x ~157 MB). Point-sampling only touches the
# tiles your points fall in. This script reports the unique-tile count and the
# download size, then REQUIRES confirmation before pulling anything.
#
# Flow:
#   1. R writes pa_xy (occ + lon + lat) to PA_CSV, then stops.
#   2. This script reads PA_CSV, estimates tiles+size, asks to proceed,
#      samples TESSERA(year), writes TESS_CSV (occ, lon, lat, TS1..TS128).
#   3. R reads TESS_CSV, joins on, continues the 2x3 comparison.
###############################################################################

import os
import sys
import csv

try:
    from geotessera import GeoTessera
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Run: pip install geotessera numpy pandas")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_VERSION = "v1"
YEAR            = 2020          # dense + within AEF 2018-2022 window

PA_CSV   = r"C:/Users/liams/Documents/PhD-Project Data/TESSERA/pa_points.csv"      # from R
TESS_CSV = r"C:/Users/liams/Documents/PhD-Project Data/TESSERA/tessera_at_points.csv"  # to R

LON_COL  = "lon"
LAT_COL  = "lat"
OCC_COL  = "occ"

MAX_GB_AUTO = 5.0   # if estimated download <= this, proceed without prompting
SAMPLE_BATCH = 2000 # sample in batches so progress is visible / memory is flat

# ---------------------------------------------------------------------------
def load_points(path):
    if not os.path.exists(path):
        sys.exit(f"[points] PA_CSV not found: {path}\n"
                 "Run the R script part 1 first to write the presence/background "
                 "points.")
    df = pd.read_csv(path)
    for c in (OCC_COL, LON_COL, LAT_COL):
        if c not in df.columns:
            sys.exit(f"[points] column '{c}' missing from {path}; "
                     f"found {list(df.columns)}")
    df = df.dropna(subset=[LON_COL, LAT_COL]).reset_index(drop=True)
    print(f"[points] {len(df)} points loaded "
          f"(pres={int((df[OCC_COL]==1).sum())}, "
          f"bg={int((df[OCC_COL]==0).sum())})")
    return df

# ---------------------------------------------------------------------------
def estimate_and_confirm(gt, df):
    # Map each point to its 0.1-degree tile centre, count uniques.
    # Tile centres are at .x5 (e.g. 35.05); floor to 0.1 grid then + 0.05.
    def tile_centre(v):
        base = np.floor(v * 10) / 10.0
        return round(base + 0.05, 2)
    tc = {(tile_centre(lon), tile_centre(lat))
          for lon, lat in zip(df[LON_COL], df[LAT_COL])}
    n_tiles = len(tc)

    # Build tile list (year, lon, lat) for the size calculator.
    tiles = [(YEAR, lon, lat) for (lon, lat) in tc]
    try:
        from pathlib import Path
        result = gt.registry.calculate_download_requirements(
            tiles=tiles, output_dir=Path("./tessera_work"), format_type="npy")
        # returns (n_to_download, total_bytes, breakdown) per 0.8.0 signature
        total_bytes = None
        if isinstance(result, tuple):
            ints = [x for x in result if isinstance(x, int)]
            if len(ints) >= 2:
                total_bytes = max(ints)   # the byte count is the large int
        if total_bytes is None:
            # fall back: estimate from documented per-tile embedding size
            total_bytes = n_tiles * 157_565_312
    except Exception as ex:
        print(f"[size] calculator failed ({ex}); using per-tile estimate.")
        total_bytes = n_tiles * 157_565_312

    gb = total_bytes / 1e9
    print(f"\n[size] points touch {n_tiles} unique TESSERA tiles for {YEAR}.")
    print(f"[size] estimated download: ~{gb:.1f} GB "
          f"(only tiles not already cached will actually transfer).")

    if gb <= MAX_GB_AUTO:
        print(f"[size] under {MAX_GB_AUTO} GB threshold; proceeding.")
        return True
    print(f"[size] OVER {MAX_GB_AUTO} GB.")
    if n_tiles > 2000:
        print("[size] TIP: 10k background points scatter across many tiles. "
              "Reducing N_BACKGROUND in the R script (e.g. to 2000) would cut "
              "the tile count and download substantially.")
    ans = input("Proceed with download/sampling? [y/N] ").strip().lower()
    return ans == "y"

# ---------------------------------------------------------------------------
def sample(gt, df):
    pts = list(zip(df[LON_COL].astype(float), df[LAT_COL].astype(float)))
    n = len(pts)
    out = np.full((n, 128), np.nan, dtype="float32")
    for i in range(0, n, SAMPLE_BATCH):
        j = min(i + SAMPLE_BATCH, n)
        print(f"[sample] points {i}-{j} of {n} ...")
        emb = gt.sample_embeddings_at_points(pts[i:j], year=YEAR,
                                             auto_download=True)
        out[i:j, :] = np.asarray(emb, dtype="float32")
    return out

# ---------------------------------------------------------------------------
def write_csv(df, emb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [f"TS{k+1}" for k in range(emb.shape[1])]
    res = df[[OCC_COL, LON_COL, LAT_COL]].copy()
    emb_df = pd.DataFrame(emb, columns=cols)
    res = pd.concat([res.reset_index(drop=True), emb_df], axis=1)
    res.to_csv(path, index=False)
    n_na = int(res[cols].isna().any(axis=1).sum())
    print(f"[write] {path}  ({res.shape[0]} rows, {len(cols)} TESSERA cols; "
          f"{n_na} rows with NA -- outside coverage)")

# ---------------------------------------------------------------------------
def main():
    df = load_points(PA_CSV)
    gt = GeoTessera(dataset_version=DATASET_VERSION)
    print(f"[init] GeoTessera {getattr(gt, 'version', 'n/a')}, year={YEAR}")

    if not estimate_and_confirm(gt, df):
        sys.exit("[abort] not confirmed; nothing downloaded.")

    emb = sample(gt, df)
    write_csv(df, emb, TESS_CSV)
    print(f"\n[done] TESSERA values at points ready:\n  {TESS_CSV}")
    print("       Set TESS_CSV in the R script CONFIG to this path and run part 2.")

if __name__ == "__main__":
    main()
