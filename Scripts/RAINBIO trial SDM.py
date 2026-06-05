"""
============================================================
TRIAL SDM — SINGLE SPECIES, TANZANIA
============================================================

Builds a Maxent species distribution model for one widespread
Tanzanian plant species using RAINBIO occurrences and WorldClim
bioclimatic predictors.

What this script does:
  1. Selects a widespread well-recorded species from RAINBIO
  2. Downloads WorldClim bioclim layers for Tanzania (cached)
  3. Extracts environmental values at species occurrences
  4. Generates pseudo-absence / background points
  5. Fits a Maxent-style model (using elapid, a modern Python implementation)
  6. Predicts habitat suitability across Tanzania
  7. Evaluates model performance (AUC + cross-validation)
  8. Plots predicted distribution vs observed records

Inputs:
  - RAINBIO Tanzania CSV
  - WorldClim 2.1 bioclim layers (downloaded automatically)

Outputs (in 'tanzania_sdm/'):
  - 01_species_occurrences.png    — input data map
  - 02_predictor_correlation.png  — predictor multicollinearity check
  - 03_predictor_importance.png   — Maxent variable importance
  - 04_response_curves.png        — partial response curves
  - 05_predicted_distribution.png — habitat suitability map
  - 06_predicted_vs_observed.png  — predictions with occurrence overlay
  - sdm_summary.txt               — performance + chosen species info

Dependencies:
  pip install elapid rasterio numpy pandas matplotlib geopandas \
              scikit-learn requests
============================================================
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import requests
import zipfile
from pathlib import Path

# Modern Python Maxent implementation
import elapid as ela


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\liams\Documents\PhD-Project Data\tanzania\tanzania_points.csv"
OUTPUT_DIR = r"C:\Users\liams\Documents\PhD-Project Data\RAINBIO\tanzania_sdm"
WORLDCLIM_DIR = r"C:/Users/liams/Documents/PhD-Project Data/worldclim/climate/wc2.1_30s/"
WC_RESOLUTION = "30s"

# Tanzania bounding box (lon_min, lat_min, lon_max, lat_max)
TANZANIA_BBOX = (29.3, -11.8, 40.5, -1.0)

# Bioclim variables to use (subset of 19 to avoid multicollinearity)
# These are the most ecologically interpretable + commonly used
BIOCLIM_VARS = [
    "bio1",   # Annual mean temperature
    "bio4",   # Temperature seasonality
    "bio5",   # Max temperature of warmest month
    "bio6",   # Min temperature of coldest month
    "bio12",  # Annual precipitation
    "bio15",  # Precipitation seasonality
    "bio16",  # Precipitation of wettest quarter
    "bio17",  # Precipitation of driest quarter
]

# Minimum records to consider a species "well-recorded"
MIN_RECORDS = 80

# Number of pseudo-absence (background) points
N_BACKGROUND = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORLDCLIM_DIR, exist_ok=True)

print("=" * 70)
print("TRIAL SDM — TANZANIA")
print("=" * 70)


# ============================================================
# STEP 1 — LOAD RAINBIO + PICK A SPECIES
# ============================================================

print("\n[1] LOADING RAINBIO + PICKING A WIDESPREAD SPECIES")
print("-" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip()

lat_col = "decimalLatitude"
lon_col = "decimalLongitude"
sp_col  = "species"
df = df.dropna(subset=[lat_col, lon_col, sp_col])

# Pick a widespread species — between 100 and 300 records is ideal
# Too few = unreliable; too many = trivially easy
records_per_sp = df.groupby(sp_col).size()
candidates = records_per_sp[(records_per_sp >= MIN_RECORDS) &
                             (records_per_sp <= 400)].sort_values()

print(f"    Total species in RAINBIO Tanzania : {len(records_per_sp):,}")
print(f"    Species with {MIN_RECORDS}–400 records      : {len(candidates):,}")

# Pick the median candidate — a well-recorded but not exceptional species
target = candidates.index[len(candidates) // 2]
n_target = candidates[target]

# Get its records and compute spatial spread
sp_data = df[df[sp_col] == target].copy()
lat_range = sp_data[lat_col].max() - sp_data[lat_col].min()
lon_range = sp_data[lon_col].max() - sp_data[lon_col].min()

print(f"\n    SELECTED SPECIES: {target}")
print(f"    Records      : {n_target}")
print(f"    Lat range    : {sp_data[lat_col].min():.2f} → {sp_data[lat_col].max():.2f} ({lat_range:.1f}°)")
print(f"    Lon range    : {sp_data[lon_col].min():.2f} → {sp_data[lon_col].max():.2f} ({lon_range:.1f}°)")


# ============================================================
# STEP 2 — DOWNLOAD WORLDCLIM (IF NOT CACHED)
# ============================================================

print("\n[2] LOADING WORLDCLIM BIOCLIM LAYERS")
print("-" * 70)

extract_dir = Path(WORLDCLIM_DIR)
print(f"    Using local WorldClim files in: {extract_dir}")
print(f"    Resolution: {WC_RESOLUTION} (~1 km at the equator)")

def load_and_clip_bioclim(var_name):
    """Load a bioclim raster and clip to Tanzania bbox."""
    var_num = var_name.replace("bio", "")
    candidates_paths = [
        extract_dir / f"wc2.1_{WC_RESOLUTION}_bio_{var_num}.tif",
        extract_dir / f"wc2.1_{WC_RESOLUTION}_bio_{int(var_num):02d}.tif",
    ]
    raster_path = next((p for p in candidates_paths if p.exists()), None)
    if raster_path is None:
        raise FileNotFoundError(
            f"Couldn't find raster for {var_name}\n"
            f"Looked in: {extract_dir}\n"
            f"Tried: {[str(p) for p in candidates_paths]}"
        )

    with rasterio.open(raster_path) as src:
        from rasterio.windows import from_bounds
        window = from_bounds(*TANZANIA_BBOX, transform=src.transform)
        data = src.read(1, window=window)
        transform = src.window_transform(window)

        # Convert no-data values to NaN so masking works downstream
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data).astype(np.float32)
        # Also catch unrealistic extreme values (defensive — e.g. -3.4e+38)
        data = np.where(np.abs(data) > 1e10, np.nan, data).astype(np.float32)

        return {
            "data": data,
            "transform": transform,
            "crs": src.crs,
            "name": var_name,
        }

# ADD THESE LINES — actually load all the rasters
print(f"    Loading {len(BIOCLIM_VARS)} bioclim variables...")
predictors = {v: load_and_clip_bioclim(v) for v in BIOCLIM_VARS}
print(f"    Done. Each layer: {predictors['bio1']['data'].shape}")

# ============================================================
# STEP 3 — EXTRACT PREDICTOR VALUES AT OCCURRENCES
# ============================================================

print("\n[3] EXTRACTING PREDICTOR VALUES AT OCCURRENCES")
print("-" * 70)


def extract_at_points(raster, lons, lats):
    """Extract raster values at given lon/lat points."""
    inv_transform = ~raster["transform"]
    cols, rows = inv_transform * (np.asarray(lons), np.asarray(lats))
    cols = cols.astype(int)
    rows = rows.astype(int)
    data = raster["data"]
    h, w = data.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(len(lons), np.nan)
    out[valid] = data[rows[valid], cols[valid]]
    return out


# Build presence dataframe
presence_df = pd.DataFrame({"lon": sp_data[lon_col].values,
                             "lat": sp_data[lat_col].values})
for v in BIOCLIM_VARS:
    presence_df[v] = extract_at_points(predictors[v],
                                        presence_df["lon"], presence_df["lat"])

# Drop rows where any predictor is NaN (points outside raster, e.g. coastal)
n_before = len(presence_df)
presence_df = presence_df.dropna()
print(f"    Presence points: {n_before} → {len(presence_df)} after NaN removal")


# ============================================================
# STEP 4 — GENERATE BACKGROUND (PSEUDO-ABSENCE) POINTS
# ============================================================

print(f"\n[4] GENERATING {N_BACKGROUND:,} BACKGROUND POINTS")
print("-" * 70)

rng = np.random.default_rng(seed=42)

# Random points within Tanzania bbox, then drop those falling on NaN raster
n_attempts = N_BACKGROUND * 3   # over-sample to compensate for NaN drops
bg_lons = rng.uniform(TANZANIA_BBOX[0], TANZANIA_BBOX[2], n_attempts)
bg_lats = rng.uniform(TANZANIA_BBOX[1], TANZANIA_BBOX[3], n_attempts)

background_df = pd.DataFrame({"lon": bg_lons, "lat": bg_lats})
for v in BIOCLIM_VARS:
    background_df[v] = extract_at_points(predictors[v],
                                          background_df["lon"], background_df["lat"])
background_df = background_df.dropna().head(N_BACKGROUND).reset_index(drop=True)
print(f"    Final background points: {len(background_df):,}")


# ============================================================
# STEP 5 — CHECK PREDICTOR CORRELATION
# ============================================================

print("\n[5] PREDICTOR CORRELATION CHECK")
print("-" * 70)

# Compute correlation matrix on all sample points (presence + background)
X_combined = pd.concat([presence_df[BIOCLIM_VARS],
                         background_df[BIOCLIM_VARS]])
corr = X_combined.corr()
print(f"    Pairs with |r| > 0.85 (potential multicollinearity):")
high_corr = []
for i, v1 in enumerate(BIOCLIM_VARS):
    for v2 in BIOCLIM_VARS[i+1:]:
        r = corr.loc[v1, v2]
        if abs(r) > 0.85:
            high_corr.append((v1, v2, r))
            print(f"      {v1} ↔ {v2}: r = {r:.2f}")
if not high_corr:
    print(f"      None — all predictors reasonably independent.")

# Plot correlation heatmap
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(BIOCLIM_VARS)))
ax.set_yticks(range(len(BIOCLIM_VARS)))
ax.set_xticklabels(BIOCLIM_VARS, rotation=45, ha="right")
ax.set_yticklabels(BIOCLIM_VARS)
for i in range(len(BIOCLIM_VARS)):
    for j in range(len(BIOCLIM_VARS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="black")
plt.colorbar(im, label="Pearson r")
ax.set_title("Predictor correlation matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_predictor_correlation.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# STEP 6 — FIT MAXENT MODEL
# ============================================================

print("\n[6] FITTING MAXENT MODEL (via elapid)")
print("-" * 70)

# Build training dataset: 1 = presence, 0 = background
y_train = np.concatenate([
    np.ones(len(presence_df), dtype=int),
    np.zeros(len(background_df), dtype=int),
])
X_train = pd.concat([presence_df[BIOCLIM_VARS],
                     background_df[BIOCLIM_VARS]]).reset_index(drop=True)

# Fit Maxent
model = ela.MaxentModel(
    feature_types=["linear", "quadratic", "product"],
    beta_multiplier=1.0,
    use_lambdas="best",
    n_lambdas=100,
    convergence_tolerance=1e-6,
)
model.fit(X_train, y_train)

print(f"    Model fitted on {len(X_train):,} points "
      f"({len(presence_df):,} presences + {len(background_df):,} background)")


# ============================================================
# STEP 7 — EVALUATE MODEL
# ============================================================

print("\n[7] EVALUATING MODEL")
print("-" * 70)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# In-sample AUC
preds_train = model.predict(X_train)
auc_train = roc_auc_score(y_train, preds_train)

# 5-fold cross-validation AUC
kf = KFold(n_splits=5, shuffle=True, random_state=42)
auc_cv = []
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_train)):
    m = ela.MaxentModel(
        feature_types=["linear", "quadratic", "product"],
        beta_multiplier=1.0,
        use_lambdas="best",
        n_lambdas=100,
    )
    m.fit(X_train.iloc[train_idx], y_train[train_idx])
    p = m.predict(X_train.iloc[test_idx])
    auc_fold = roc_auc_score(y_train[test_idx], p)
    auc_cv.append(auc_fold)
    print(f"    Fold {fold_idx + 1}: AUC = {auc_fold:.3f}")

print(f"\n    In-sample AUC      : {auc_train:.3f}")
print(f"    Cross-validation   : {np.mean(auc_cv):.3f} ± {np.std(auc_cv):.3f}")
print(f"    (AUC > 0.7 = useful, > 0.85 = strong)")


# ============================================================
# STEP 8 — VARIABLE IMPORTANCE
# ============================================================

print("\n[8] VARIABLE IMPORTANCE")
print("-" * 70)

# Permutation importance: how much does AUC drop when each variable is shuffled?
importance = {}
baseline_auc = auc_train
for v in BIOCLIM_VARS:
    X_perm = X_train.copy()
    X_perm[v] = rng.permutation(X_perm[v].values)
    p_perm = model.predict(X_perm)
    auc_perm = roc_auc_score(y_train, p_perm)
    importance[v] = baseline_auc - auc_perm
    print(f"    {v:8s}: importance = {importance[v]:.4f}")

# Plot importance
fig, ax = plt.subplots(figsize=(8, 5))
imp_sorted = sorted(importance.items(), key=lambda x: x[1])
names = [k for k, v in imp_sorted]
values = [v for k, v in imp_sorted]
ax.barh(names, values, color="steelblue", edgecolor="white")
ax.set_xlabel("Importance (drop in AUC when shuffled)")
ax.set_title(f"Variable importance — {target}", fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_predictor_importance.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# STEP 9 — PREDICT ACROSS TANZANIA
# ============================================================

print("\n[9] PREDICTING HABITAT SUITABILITY ACROSS TANZANIA")
print("-" * 70)

# Stack all predictor rasters into a 3D array (vars, rows, cols)
ref = predictors[BIOCLIM_VARS[0]]
h, w = ref["data"].shape
stack = np.stack([predictors[v]["data"] for v in BIOCLIM_VARS], axis=0)

# Reshape to (n_pixels, n_vars) for prediction
flat = stack.reshape(len(BIOCLIM_VARS), -1).T  # (h*w, n_vars)
flat_df = pd.DataFrame(flat, columns=BIOCLIM_VARS)

# Mark valid pixels (no NaN)
valid_mask = flat_df.notna().all(axis=1).values

# Predict
predictions = np.full(h * w, np.nan)
predictions[valid_mask] = model.predict(flat_df.loc[valid_mask])
prediction_grid = predictions.reshape(h, w)

print(f"    Predicted {valid_mask.sum():,} pixels")
print(f"    Suitability range: {np.nanmin(prediction_grid):.3f} to {np.nanmax(prediction_grid):.3f}")


# ============================================================
# STEP 10 — PLOT PREDICTED DISTRIBUTION
# ============================================================

print("\n[10] PLOTTING PREDICTED DISTRIBUTION")
print("-" * 70)

# Get geographic extent
trans = ref["transform"]
extent = [trans.c, trans.c + trans.a * w,
          trans.f + trans.e * h, trans.f]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Map 1: predicted suitability ---
ax = axes[0]
im = ax.imshow(prediction_grid, extent=extent, cmap="YlOrRd",
               vmin=0, vmax=np.nanpercentile(prediction_grid, 99),
               origin="upper")
ax.scatter(presence_df["lon"], presence_df["lat"],
           s=8, c="blue", edgecolor="white", linewidth=0.3,
           label=f"Records (n={len(presence_df)})", zorder=5)
plt.colorbar(im, ax=ax, label="Habitat suitability", shrink=0.7)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Predicted distribution — {target}\nMaxent + WorldClim bioclim",
             fontweight="bold")
ax.legend(loc="upper left")
ax.set_xlim(TANZANIA_BBOX[0], TANZANIA_BBOX[2])
ax.set_ylim(TANZANIA_BBOX[1], TANZANIA_BBOX[3])

# --- Map 2: same but with binarised threshold ---
threshold = np.nanpercentile(prediction_grid, 75)
binary = (prediction_grid >= threshold).astype(float)
binary[np.isnan(prediction_grid)] = np.nan
ax = axes[1]
ax.imshow(binary, extent=extent, cmap="Greens", vmin=0, vmax=1, origin="upper")
ax.scatter(presence_df["lon"], presence_df["lat"],
           s=8, c="red", edgecolor="white", linewidth=0.3,
           label=f"Records (n={len(presence_df)})", zorder=5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Suitable habitat (≥75th percentile threshold)\n"
             f"AUC (in-sample) = {auc_train:.3f}, "
             f"CV = {np.mean(auc_cv):.3f} ± {np.std(auc_cv):.3f}",
             fontweight="bold")
ax.legend(loc="upper left")
ax.set_xlim(TANZANIA_BBOX[0], TANZANIA_BBOX[2])
ax.set_ylim(TANZANIA_BBOX[1], TANZANIA_BBOX[3])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_predicted_distribution.png", dpi=200, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/05_predicted_distribution.png")
plt.close()


# ============================================================
# STEP 11 — SAVE SUMMARY
# ============================================================

with open(f"{OUTPUT_DIR}/sdm_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Trial SDM — Tanzania\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Species             : {target}\n")
    f.write(f"Records used        : {len(presence_df):,}\n")
    f.write(f"Background points   : {len(background_df):,}\n")
    f.write(f"Predictors          : {', '.join(BIOCLIM_VARS)}\n")
    f.write(f"Algorithm           : Maxent (via elapid)\n\n")
    f.write(f"Performance:\n")
    f.write(f"  In-sample AUC     : {auc_train:.3f}\n")
    f.write(f"  Cross-validation  : {np.mean(auc_cv):.3f} ± {np.std(auc_cv):.3f}\n")
    f.write(f"  CV folds          : {auc_cv}\n\n")
    f.write(f"Variable importance (drop in AUC when shuffled):\n")
    for v, imp in sorted(importance.items(), key=lambda x: -x[1]):
        f.write(f"  {v:8s} : {imp:.4f}\n")

print("\n" + "=" * 70)
print(f"DONE — SDM trial complete for {target}")
print("=" * 70)
print(f"Outputs: {OUTPUT_DIR}/")