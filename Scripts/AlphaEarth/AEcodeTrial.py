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
    ee.Initialize()
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
 
 
