"""
============================================================
SPECIES-AREA CURVE — TANZANIA
============================================================

Builds an area-based species accumulation curve for Tanzania.

Logic:
  1. Assign each RAINBIO record to an H3 hex cell
  2. For each "sample size" N (in hexes), repeatedly:
       - draw N random hexes
       - count unique species across those hexes
  3. Average + 95% CI across draws
  4. Plot cumulative species vs cumulative area

Input:  RAINBIO Tanzania point records CSV
Output: 06_species_area_curve.png + zvalue.txt

Dependencies:
  pip install pandas numpy matplotlib h3
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h3
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points.csv"
OUTPUT_DIR = r"C:/Users/liams/Documents/PhD-Project Data/tanzania_diversity_maps"

HEX_RESOLUTION = 4         # Match your existing richness map (~5,122 km²/hex)
N_DRAWS        = 50        # Number of random draws per sample size
N_STEPS        = 30        # Number of points along the curve

os.makedirs(OUTPUT_DIR, exist_ok=True)

# H3 res 4 ≈ 1770 km² is documented; your run reported ~5,122 km².
# H3 area scales by resolution. Use h3 library directly to be safe.
HEX_AREA_KM2 = h3.cell_area(h3.latlng_to_cell(-6.0, 35.0, HEX_RESOLUTION),
                             unit="km^2")

print("=" * 60)
print("TANZANIA — SPECIES-AREA CURVE")
print("=" * 60)
print(f"    H3 resolution      : {HEX_RESOLUTION}")
print(f"    Approx hex area    : {HEX_AREA_KM2:,.0f} km²")
print(f"    Bootstrap draws    : {N_DRAWS}")


# ============================================================
# LOAD DATA & ASSIGN HEXES
# ============================================================

print("\n[1] Loading data...")

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip()

lat_col = "decimalLatitude"
lon_col = "decimalLongitude"
sp_col  = "species"

df = df.dropna(subset=[lat_col, lon_col, sp_col])
print(f"    Records loaded     : {len(df):,}")
print(f"    Unique species     : {df[sp_col].nunique():,}")

print("\n[2] Assigning records to H3 hexes...")
df["hex_id"] = df.apply(
    lambda r: h3.latlng_to_cell(r[lat_col], r[lon_col], HEX_RESOLUTION),
    axis=1
)
unique_hexes = df["hex_id"].unique()
n_hexes = len(unique_hexes)
print(f"    Hexes with records : {n_hexes:,}")
print(f"    Total area covered : ~{n_hexes * HEX_AREA_KM2:,.0f} km²")


# ============================================================
# BUILD ACCUMULATION CURVE
# ============================================================

print(f"\n[3] Building accumulation curve ({N_DRAWS} draws × {N_STEPS} points)...")

# Sample sizes (in number of hexes) — log-spaced for nicer curve shape
sample_sizes = np.unique(
    np.round(np.logspace(0, np.log10(n_hexes), N_STEPS)).astype(int)
)

species_counts = np.zeros((N_DRAWS, len(sample_sizes)))

# Pre-build hex → species lookup for speed
hex_species = df.groupby("hex_id")[sp_col].apply(set).to_dict()

rng = np.random.default_rng(seed=42)
for d in range(N_DRAWS):
    shuffled = rng.permutation(unique_hexes)
    for i, n in enumerate(sample_sizes):
        chosen = shuffled[:n]
        seen = set()
        for h in chosen:
            seen.update(hex_species[h])
        species_counts[d, i] = len(seen)

mean_count  = species_counts.mean(axis=0)
lower_count = np.percentile(species_counts, 2.5, axis=0)
upper_count = np.percentile(species_counts, 97.5, axis=0)
areas       = sample_sizes * HEX_AREA_KM2


# ============================================================
# FIT POWER LAW (S = c × A^z)
# ============================================================

print("\n[4] Fitting species-area power law (S = c·A^z)...")

# log-log linear regression for robust z estimate
log_a = np.log10(areas)
log_s = np.log10(mean_count)
slope, intercept = np.polyfit(log_a, log_s, 1)
z_value = slope
c_value = 10 ** intercept

print(f"    z (slope)          : {z_value:.3f}")
print(f"    c (intercept)      : {c_value:.2f}")
print(f"    Tropical norm range: 0.20 – 0.35")
if 0.20 <= z_value <= 0.35:
    print(f"    Interpretation     : within typical tropical range — sensible")
elif z_value < 0.20:
    print(f"    Interpretation     : low — may indicate spatial under-sampling")
else:
    print(f"    Interpretation     : high — strong spatial turnover")


# ============================================================
# PLOT
# ============================================================

print("\n[5] Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Linear scale ---
ax = axes[0]
ax.fill_between(areas, lower_count, upper_count, alpha=0.2, color="darkgreen",
                label="95% Confidence Interval")
ax.plot(areas, mean_count, color="darkgreen", linewidth=2.5,
        label=f"Tanzania observed")
ax.axhline(df[sp_col].nunique(), color="black", linestyle=":", alpha=0.5,
           label=f"Total RAINBIO-tz species: {df[sp_col].nunique():,}")
ax.set_xlabel("Cumulative area (km²)")
ax.set_ylabel("Cumulative species count")
ax.set_title("Species-Area Curve", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# --- Log-log scale with power law fit ---
ax = axes[1]
ax.loglog(areas, mean_count, "o-", color="darkgreen", markersize=5,
          label=f"Observed mean")
ax.fill_between(areas, lower_count, upper_count, alpha=0.2, color="darkgreen")
fit_line = c_value * areas ** z_value
ax.loglog(areas, fit_line, "--", color="red", linewidth=1.5,
          label=f"Arrhenius Power law : S = {c_value:.2f} × A^{z_value:.3f}")
ax.set_xlabel("Cumulative area sampled (km², log)")
ax.set_ylabel("Cumulative species (log)")
ax.set_title(f"Log-log Species-Area Curve\nz = {z_value:.3f}",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/06_species_area_curve.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"    Saved: {out_path}")
plt.close()


# ============================================================
# SAVE Z-VALUE & SUMMARY
# ============================================================

with open(f"{OUTPUT_DIR}/species_area_summary.txt", "w") as f:
    f.write("Tanzania Species-Area Curve Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Records              : {len(df):,}\n")
    f.write(f"Unique species       : {df[sp_col].nunique():,}\n")
    f.write(f"Hexes with records   : {n_hexes:,}\n")
    f.write(f"Hex area             : ~{HEX_AREA_KM2:,.0f} km² (H3 res {HEX_RESOLUTION})\n")
    f.write(f"Total area           : ~{n_hexes * HEX_AREA_KM2:,.0f} km²\n\n")
    f.write(f"Power law fit S = c × A^z:\n")
    f.write(f"  z (slope)          : {z_value:.3f}\n")
    f.write(f"  c (intercept)      : {c_value:.2f}\n")
    f.write(f"  Tropical norm      : 0.20 – 0.35\n")

print(f"    Summary saved: {OUTPUT_DIR}/species_area_summary.txt")
print("\n" + "=" * 60)
print(f"DONE — z-value = {z_value:.3f}")
print("=" * 60)