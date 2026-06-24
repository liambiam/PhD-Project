#!/usr/bin/env python3
###############################################################################
# tessera_zarr_probe.py
# -----------------------------------------------------------------------------
# CHEAP, REVERSIBLE probe before committing to any large download.
# Tests whether the upgraded geotessera Zarr-streaming backend works on this
# machine and which year has usable coverage over Tanzania -- by sampling a
# SINGLE point (a few hundred KB), not 326 GB.
#
# Run this in a FRESH environment so a broken upgrade can't touch the env your
# WorldClim/AlphaEarth pipeline depends on:
#     python -m venv tessera_env
#     tessera_env\Scripts\activate        (Windows)
#     pip install --upgrade geotessera
#     python tessera_zarr_probe.py
###############################################################################

import sys

# Central Tanzania test point (lon, lat) -- should exist if coverage is good.
TEST_LON, TEST_LAT = 35.0, -6.0
YEARS = [2024, 2020]   # 2024 first: paper says it's the gap-free year.

def main():
    try:
        import geotessera
    except ImportError:
        sys.exit("geotessera not installed in this environment.")
    print("geotessera version:", getattr(geotessera, "__version__", "unknown"))

    # 1. Is the Zarr streaming class present at all?
    zarr_cls = None
    for name in ("GeoTesseraZarr", "GeoTesseraZ"):
        zarr_cls = getattr(geotessera, name, None)
        if zarr_cls is not None:
            print(f"[zarr] found streaming class: {name}")
            break
    if zarr_cls is None:
        print("[zarr] NO Zarr streaming class in this version.")
        print("       Upgrade did not bring streaming, OR import name differs.")
        print("       Available top-level names:",
              [n for n in dir(geotessera) if not n.startswith('_')])
        sys.exit("Cannot stream; do not proceed to bulk download without rethink.")

    # 2. Inspect its constructor + methods so we know how to call it.
    import inspect
    print("[zarr] __init__:", inspect.signature(zarr_cls.__init__))
    methods = [m for m in dir(zarr_cls) if not m.startswith("_")]
    print("[zarr] methods:", methods)

    # 3. Try to instantiate and sample ONE point per year.
    for year in YEARS:
        print(f"\n[probe] year {year}: instantiating + sampling one point ...")
        try:
            z = zarr_cls()  # may need version=... ; adjust after seeing __init__
        except Exception as ex:
            print(f"[probe]   init failed: {ex}")
            continue

        # Try the most likely point-sample method names in order.
        sampled = None
        for m in ("sample_embeddings_at_points", "sample_points",
                  "sample_at_points", "sample"):
            fn = getattr(z, m, None)
            if fn is None:
                continue
            try:
                try:
                    sampled = fn([(TEST_LON, TEST_LAT)], year=year)
                except TypeError:
                    sampled = fn([(TEST_LON, TEST_LAT)])
                print(f"[probe]   {m}() worked.")
                break
            except Exception as ex:
                print(f"[probe]   {m}() failed: {ex}")

        if sampled is not None:
            import numpy as np
            arr = np.asarray(sampled)
            print(f"[probe]   year {year}: shape={arr.shape}, "
                  f"head={arr.ravel()[:4]}")
            print(f"[probe]   year {year}: COVERAGE OK via streaming.")
        else:
            print(f"[probe]   year {year}: no working sample method / no coverage.")

    print("\n[done] If a year above says 'COVERAGE OK via streaming', use that "
          "year and we'll switch the sampler to the Zarr path (tiny downloads).")

if __name__ == "__main__":
    main()
