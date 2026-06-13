"""
============================================================
TRUE SPECIES-AREA RELATIONSHIP — TANZANIA
============================================================

A proper species-area curve using nested H3 hexes at multiple
resolutions. Unlike the accumulation curve, this measures how
species count genuinely scales with sampling unit size.

Design:
  1. Assign each RAINBIO record to H3 hexes at multiple resolutions
     (e.g., res 3, 4, 5, 6, 7). Each resolution gives a different
     hex size.
  2. At each resolution, randomly sample hexes containing records.
  3. For each hex, count species within it.
  4. Plot mean species count vs hex area, with 95% CI.
  5. Fit Arrhenius power law: S = c × A^z

This is a true SAR: each point measures species per sampling unit
of a given size, not cumulative species across pooled units.

Input:  RAINBIO Tanzania point records CSV
Output: 07_true_species_area_curve.png + summary text

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

INPUT_FILE = r"C:\Users\liams\Documents\PhD-Project Data\tanzania\tanzania_points.csv"
OUTPUT_DIR = r"C:\Users\liams\Documents\PhD-Project Data\tanzania_diversity_maps"

# H3 resolutions to use — each step finer divides hexes by ~7
# Resolution 3 = ~12,400 km², res 7 = ~5 km²
# Wider range = stronger curve, but coarser end can be unreliable if few hexes
RESOLUTIONS = [3, 4, 5, 6, 7, 8]

# How many random hexes to sample at each resolution
N_SAMPLES_PER_RESOLUTION = 200

# Minimum records per hex to include — avoids empty/near-empty hexes
# distorting the curve at fine resolutions
MIN_RECORDS_PER_HEX = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("TANZANIA — TRUE SPECIES-AREA RELATIONSHIP")
print("=" * 60)


# ============================================================
# LOAD DATA
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


# ============================================================
# ASSIGN HEXES AT MULTIPLE RESOLUTIONS
# ============================================================

print(f"\n[2] Assigning hexes at {len(RESOLUTIONS)} resolutions...")

resolution_data = {}

for res in RESOLUTIONS:
    print(f"    Resolution {res}: ", end="")

    # Assign each record to a hex at this resolution
    df[f"hex_r{res}"] = df.apply(
        lambda r: h3.latlng_to_cell(r[lat_col], r[lon_col], res),
        axis=1
    )

    # Group records by hex, count records and unique species
    hex_summary = (
        df.groupby(f"hex_r{res}")
        .agg(
            n_records=(sp_col, "size"),
            n_species=(sp_col, "nunique"),
        )
        .reset_index()
    )

    # Filter out hexes with very few records (unreliable)
    hex_summary = hex_summary[hex_summary["n_records"] >= MIN_RECORDS_PER_HEX]

    # Get the actual hex area for this resolution at Tanzania latitudes
    sample_cell = hex_summary[f"hex_r{res}"].iloc[0]
    hex_area = h3.cell_area(sample_cell, unit="km^2")

    resolution_data[res] = {
        "n_hexes": len(hex_summary),
        "hex_area_km2": hex_area,
        "species_counts": hex_summary["n_species"].values,
    }

    print(f"{len(hex_summary):,} hexes (≥{MIN_RECORDS_PER_HEX} records), "
          f"~{hex_area:.1f} km² each")


# ============================================================
# RANDOM SAMPLING AT EACH RESOLUTION
# ============================================================

print(f"\n[3] Random sampling — {N_SAMPLES_PER_RESOLUTION} hexes per resolution...")

rng = np.random.default_rng(seed=42)

curve_data = []
for res, info in resolution_data.items():
    species_counts = info["species_counts"]
    n_avail = len(species_counts)

    if n_avail == 0:
        print(f"    Resolution {res}: skipped — no usable hexes")
        continue

    # Sample with replacement if fewer hexes than N_SAMPLES, else without
    n_to_draw = min(N_SAMPLES_PER_RESOLUTION, n_avail)
    sampled = rng.choice(species_counts, size=n_to_draw, replace=False)

    curve_data.append({
        "resolution": res,
        "area_km2": info["hex_area_km2"],
        "mean_species": np.mean(sampled),
        "median_species": np.median(sampled),
        "lower": np.percentile(sampled, 2.5),
        "upper": np.percentile(sampled, 97.5),
        "n_hexes_used": n_to_draw,
        "n_hexes_available": n_avail,
    })

curve_df = pd.DataFrame(curve_data).sort_values("area_km2")
print("\n    Curve data:")
print(curve_df.to_string(index=False))


# ============================================================
# FIT ARRHENIUS POWER LAW: S = c × A^z
# ============================================================

print("\n[4] Fitting Arrhenius power law...")

log_a = np.log10(curve_df["area_km2"].values)
log_s = np.log10(curve_df["mean_species"].values)

# Linear regression in log-log space
slope, intercept = np.polyfit(log_a, log_s, 1)
z_value = slope
c_value = 10 ** intercept

# R² of the fit
predicted = slope * log_a + intercept
ss_res = np.sum((log_s - predicted) ** 2)
ss_tot = np.sum((log_s - log_s.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot

print(f"    z (slope)          : {z_value:.3f}")
print(f"    c (intercept)      : {c_value:.2f}")
print(f"    R² of log-log fit  : {r_squared:.3f}")
print(f"    Tropical norm range: 0.20 – 0.35")

if 0.15 <= z_value <= 0.40:
    interpretation = "within typical published range for tropical floras"
elif z_value < 0.15:
    interpretation = "low — may indicate sampling effects dominating real turnover"
else:
    interpretation = "high — strong spatial turnover or sampling design effect"
print(f"    Interpretation     : {interpretation}")


# ============================================================
# PLOT
# ============================================================

print("\n[5] Plotting...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Linear scale ---
ax = axes[0]
ax.errorbar(curve_df["area_km2"], curve_df["mean_species"],
            yerr=[curve_df["mean_species"] - curve_df["lower"],
                  curve_df["upper"] - curve_df["mean_species"]],
            fmt="o", color="darkblue", markersize=8, capsize=4,
            label="Mean species per hex (95% range)")

ax.set_xlabel("Hex area (km²)")
ax.set_ylabel("Species per hex")
ax.set_title("Species-Area Relationship — Tanzania (linear)",
             fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# --- Log-log scale with power law fit ---
ax = axes[1]

# Plot data
ax.errorbar(curve_df["area_km2"], curve_df["mean_species"],
            yerr=[curve_df["mean_species"] - curve_df["lower"],
                  curve_df["upper"] - curve_df["mean_species"]],
            fmt="o", color="darkblue", markersize=8, capsize=4,
            label="Observed mean")

# Plot power law fit
x_fit = np.logspace(np.log10(curve_df["area_km2"].min()),
                    np.log10(curve_df["area_km2"].max()), 100)
y_fit = c_value * x_fit ** z_value
ax.loglog(x_fit, y_fit, "--", color="red", linewidth=2,
          label=f"S = {c_value:.2f} × A^{z_value:.3f}\n(R² = {r_squared:.3f})")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Hex area (km², log)")
ax.set_ylabel("Species per hex (log)")
ax.set_title(
    f"Log-log Species-Area Relationship\nz = {z_value:.3f} ",
    fontweight="bold"
)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/07_true_species_area_curve.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"    Saved: {out_path}")
plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(f"{OUTPUT_DIR}/true_species_area_summary.txt", "w") as f:
    f.write("Tanzania — True Species-Area Relationship\n")
    f.write("=" * 50 + "\n\n")
    f.write("Design:\n")
    f.write("  Nested H3 hexes at multiple resolutions, each\n")
    f.write("  representing a different sampling unit size. Mean\n")
    f.write("  species count per hex computed at each resolution.\n\n")
    f.write(f"Resolutions used    : {RESOLUTIONS}\n")
    f.write(f"Min records per hex : {MIN_RECORDS_PER_HEX}\n")
    f.write(f"Hexes per resolution: {N_SAMPLES_PER_RESOLUTION} (or all avail)\n\n")
    f.write("Curve data:\n")
    f.write(curve_df.to_string(index=False))
    f.write(f"\n\nPower law fit S = c × A^z:\n")
    f.write(f"  z (slope)         : {z_value:.3f}\n")
    f.write(f"  c (intercept)     : {c_value:.2f}\n")
    f.write(f"  R² of log-log fit : {r_squared:.3f}\n")
    f.write(f"  Tropical norm     : 0.20 – 0.35\n")
    f.write(f"  Interpretation    : {interpretation}\n")

print(f"    Summary saved: {OUTPUT_DIR}/true_species_area_summary.txt")
print("\n" + "=" * 60)
print(f"DONE — z = {z_value:.3f} (R² = {r_squared:.3f})")
print("=" * 60)