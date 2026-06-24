#!/usr/bin/env python3
###############################################################################
# tessera_fetch.py  (geotessera 0.8.0)
# -----------------------------------------------------------------------------
# Acquire TESSERA embeddings for Tanzania (+ small buffer), check coverage
# FIRST, then fetch a mosaic, reproject to EPSG:4326, and resample to 1 km,
# producing a single analysis-ready GeoTIFF that drops into the R SDM pipeline
# as a third predictor arm (alongside WorldClim and AlphaEarth).
#
# API notes for 0.8.0 (confirmed against installed signatures):
#   * GeoTessera(dataset_version='v1')  -- no variant arg in this release
#     (v1 == legacy 1.0 / vultr line; upgrade later for v1.1/cambridge).
#   * embeddings_count(bbox, year) -> int           (coverage check)
#   * fetch_mosaic_for_region(bbox, year, target_crs)
#         -> (ndarray, transform, crs)              (fetch+mosaic+reproject)
#   * mosaic comes back at native 10 m; we resample to 1 km here to match
#     WorldClim + AlphaEarth on the shared modelling grid.
#
#   * year = 2021 (single year, mid-point of the AEF 2018-2022 window; TESSERA
#     annual embeddings are learned representations and should NOT be averaged
#     across years the way AEF composites are).
###############################################################################

import os
import sys
import json

# ---------------------------------------------------------------------------
# 0. DEPENDENCIES
#    pip install geotessera geopandas rasterio shapely numpy
# ---------------------------------------------------------------------------
try:
    from geotessera import GeoTessera
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Run: pip install geotessera geopandas rasterio shapely numpy")

# ---------------------------------------------------------------------------
# 1. CONFIG  -- the only block you should normally edit
# ---------------------------------------------------------------------------
GADM_TZA        = r"C:/Users/liams/Documents/PhD-Project Data/GADM TZ Shape/gadm41_TZA_0.shp"
BBOX_FALLBACK   = (29.0, -12.0, 41.0, -0.5)   # (min_lon,min_lat,max_lon,max_lat)

BUFFER_DEG      = 0.5         # small buffer around Tanzania (compromise footprint)

DATASET_VERSION = "v1"        # 0.8.0: legacy 1.0 line; no variant concept
YEAR            = 2021        # single year; see header note on not averaging

TARGET_CRS      = "EPSG:4326"
TARGET_RES_DEG  = 1.0 / 30.0  # ~1 km, MUST match GRID_RES_DEG in the R script

WORK_DIR        = r"C:/Users/liams/Documents/PhD-Project Data/TESSERA/work"
OUT_TIF         = r"C:/Users/liams/Documents/PhD-Project Data/TESSERA/tessera_TZ_2021_1km.tif"

# If True, only run the coverage check and exit (no fetch).
COVERAGE_ONLY   = True

# Years to probe in the coverage table (TESSERA spans 2017-2025).
PROBE_YEARS     = list(range(2017, 2026))

# ---------------------------------------------------------------------------
# 2. FOOTPRINT
# ---------------------------------------------------------------------------
def get_bbox():
    if GADM_TZA and os.path.exists(GADM_TZA):
        gdf = gpd.read_file(GADM_TZA).to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf.total_bounds
        bbox = (minx - BUFFER_DEG, miny - BUFFER_DEG,
                maxx + BUFFER_DEG, maxy + BUFFER_DEG)
        print(f"[footprint] from GADM + {BUFFER_DEG} deg buffer: "
              f"{tuple(round(b, 3) for b in bbox)}")
    else:
        bbox = BBOX_FALLBACK
        print(f"[footprint] GADM not found; using fallback bbox: {bbox}")
    return tuple(float(b) for b in bbox)

# ---------------------------------------------------------------------------
# 3. COVERAGE CHECK  -- embeddings_count over the bbox, per year
# ---------------------------------------------------------------------------
def check_coverage(gt, bbox, year):
    print(f"\n[coverage] dataset_version={DATASET_VERSION}")
    counts = {}
    for y in PROBE_YEARS:
        try:
            counts[y] = gt.embeddings_count(bbox=bbox, year=y)
        except Exception as ex:
            counts[y] = 0
            print(f"[coverage]   {y}: query failed ({ex})")
    best_year = max(counts, key=counts.get)
    print("[coverage] tile counts by year over this footprint:")
    for y in PROBE_YEARS:
        flag = "  <-- requested" if y == year else ""
        star = "  *most complete*" if y == best_year and counts[y] > 0 else ""
        print(f"            {y}: {counts[y]:>4d} tiles{flag}{star}")

    n = counts.get(year, 0)
    if n == 0:
        print(f"[coverage] No tiles for {year}. Most complete year is "
              f"{best_year} ({counts[best_year]} tiles). Edit YEAR and rerun.")
        return False
    if counts[best_year] > n * 1.2:
        print(f"[coverage] NOTE: year {best_year} has notably more tiles "
              f"({counts[best_year]}) than {year} ({n}); consider switching.")
    print(f"[coverage] year {year}: {n} tiles -- OK to proceed.")
    return True

# ---------------------------------------------------------------------------
# 4. FETCH MOSAIC  -- one call: fetch + mosaic + reproject to EPSG:4326
#    Returns native ~10 m mosaic. We normalise to (bands, rows, cols).
# ---------------------------------------------------------------------------
def fetch_mosaic(gt, bbox):
    print(f"\n[fetch] fetch_mosaic_for_region(bbox, year={YEAR}, "
          f"target_crs={TARGET_CRS}) ... (native ~10 m; large)")
    arr, transform, crs = gt.fetch_mosaic_for_region(
        bbox=bbox, year=YEAR, target_crs=TARGET_CRS, auto_download=True)

    arr = np.asarray(arr)
    # Normalise to bands-first. TESSERA has 128 channels; whichever axis is 128
    # is the band axis. If returned (rows, cols, bands) -> move to front.
    if arr.ndim != 3:
        sys.exit(f"[fetch] unexpected mosaic ndim={arr.ndim}; expected 3.")
    if arr.shape[0] != 128 and arr.shape[-1] == 128:
        arr = np.moveaxis(arr, -1, 0)
    print(f"[fetch] mosaic shape (bands,rows,cols): {arr.shape}; crs={crs}")
    return arr, transform, str(crs)

# ---------------------------------------------------------------------------
# 5. WRITE NATIVE MOSAIC, THEN RESAMPLE TO 1 km
#    Write native-res first (don't hold two big arrays at once), then warp the
#    on-disk file down to the 1 km target grid.
# ---------------------------------------------------------------------------
def write_native(arr, transform, crs, path):
    bands, rows, cols = arr.shape
    meta = dict(driver="GTiff", height=rows, width=cols, count=bands,
                dtype="float32", crs=crs, transform=transform,
                compress="deflate", predictor=2, tiled=True, bigtiff="YES")
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype("float32"))
    print(f"[write] native-res mosaic -> {path}  ({bands} bands, {cols}x{rows})")

def resample_to_1km(src_path, out_path, target_crs, target_res):
    with rasterio.open(src_path) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=target_res)
        meta = src.meta.copy()
        meta.update(crs=target_crs, transform=dst_transform,
                    width=dst_w, height=dst_h, compress="deflate",
                    predictor=2, tiled=True, bigtiff="YES")
        with rasterio.open(out_path, "w", **meta) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs=target_crs,
                    resampling=Resampling.average)  # average -> sane 10m->1km
            dst.update_tags(TESSERA_DATASET_VERSION=DATASET_VERSION,
                            TESSERA_YEAR=str(YEAR))
    print(f"[resample] 1 km mosaic -> {out_path}  ({dst_w}x{dst_h} px)")

# ---------------------------------------------------------------------------
# 6. PROVENANCE
# ---------------------------------------------------------------------------
def write_sidecar(out_tif, bbox):
    sidecar = os.path.splitext(out_tif)[0] + "_provenance.json"
    with open(sidecar, "w") as fh:
        json.dump({
            "dataset_version": DATASET_VERSION,
            "year": YEAR,
            "bbox_4326": bbox,
            "target_crs": TARGET_CRS,
            "target_res_deg": TARGET_RES_DEG,
            "geotessera_api": "0.8.0",
            "note": "Single-year TESSERA embeddings; do not mix versions.",
        }, fh, indent=2)
    print(f"[provenance] wrote {sidecar}")

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)

    bbox = get_bbox()
    gt = GeoTessera(dataset_version=DATASET_VERSION)
    print(f"[init] GeoTessera version attr: {getattr(gt, 'version', 'n/a')}")

    if not check_coverage(gt, bbox, YEAR):
        sys.exit("[abort] no coverage for requested year; edit CONFIG.")
    if COVERAGE_ONLY:
        print("\n[done] coverage-only run; exiting before fetch.")
        return

    arr, transform, crs = fetch_mosaic(gt, bbox)
    native_path = os.path.join(WORK_DIR, f"tessera_TZ_{YEAR}_native.tif")
    write_native(arr, transform, crs, native_path)
    del arr  # free the big native array before warping
    resample_to_1km(native_path, OUT_TIF, TARGET_CRS, TARGET_RES_DEG)
    write_sidecar(OUT_TIF, bbox)

    print(f"\n[done] TESSERA predictor ready for the R pipeline:\n  {OUT_TIF}")
    print("       Set TESS_TIF in the R script CONFIG to this path.")

if __name__ == "__main__":
    main()
