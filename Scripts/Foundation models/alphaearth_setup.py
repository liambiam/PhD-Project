"""
============================================================
ALPHAEARTH SETUP — RUN THIS FIRST
============================================================
One-time setup for the Tanzania exploration.

Step 1: Install all required packages
Step 2: Authenticate with Earth Engine
Step 3: Verify everything works
============================================================
"""

import subprocess
import sys

REQUIRED = [
    "earthengine-api",
    "geemap",
    "geopandas",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "rasterio",
    "folium",
]

print("=" * 60)
print("ALPHAEARTH SETUP — TANZANIA PHD")
print("=" * 60)

# ------------------------------------------------------------
# STEP 1 — INSTALL PACKAGES
# ------------------------------------------------------------
print("\n[1] Installing required packages...")
for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_").split("[")[0])
        print(f"    [✓] {pkg} already installed")
    except ImportError:
        print(f"    [→] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"    [✓] {pkg} installed")


# ------------------------------------------------------------
# STEP 2 — EARTH ENGINE AUTHENTICATION
# ------------------------------------------------------------
print("\n[2] Earth Engine authentication...")

import ee

try:
    ee.Initialize(project="phdproject-494613")
    print("    [✓] Already authenticated and initialised")
except Exception:
    print("    [→] Authentication required. Opening browser...")
    print("    Follow the prompts to authenticate with your Google account.")
    ee.Authenticate()
    print("\n    Now you need to set up a Cloud Project.")
    print("    Go to: https://console.cloud.google.com/projectcreate")
    print("    Create a new project, then enable the Earth Engine API.")
    print("    Once done, set PROJECT_ID in alphaearth_tanzania_exploration.py")


# ------------------------------------------------------------
# STEP 3 — VERIFY ACCESS TO ALPHAEARTH
# ------------------------------------------------------------
print("\n[3] Verifying access to AlphaEarth dataset...")

try:
    embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    n_images = embeddings.size().getInfo()
    print(f"    [✓] AlphaEarth Satellite Embedding accessible")
    print(f"        Total images in collection: {n_images:,}")

    # Check date range
    first = embeddings.sort("system:time_start").first()
    last = embeddings.sort("system:time_start", False).first()
    first_date = ee.Date(first.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    last_date = ee.Date(last.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    print(f"        Date range: {first_date} → {last_date}")
except Exception as e:
    print(f"    [✗] Could not access AlphaEarth: {e}")
    print(f"        Check your project has Earth Engine API enabled")


# ------------------------------------------------------------
# STEP 4 — VERIFY TANZANIA BOUNDARY DATA
# ------------------------------------------------------------
print("\n[4] Verifying Tanzania boundary data...")

try:
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    tanzania = countries.filter(ee.Filter.eq("ADM0_NAME", "United Republic of Tanzania"))
    n = tanzania.size().getInfo()
    if n > 0:
        print(f"    [✓] Tanzania boundary loaded ({n} feature)")
    else:
        print(f"    [✗] Tanzania not found in FAO GAUL")
except Exception as e:
    print(f"    [✗] Boundary check failed: {e}")


print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print("Next: edit PROJECT_ID at the top of alphaearth_tanzania_exploration.py")
print("Then run: python alphaearth_tanzania_exploration.py")
print("=" * 60)
