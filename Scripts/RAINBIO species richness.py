"""
============================================================
TANZANIA — SPECIES RICHNESS, RECORDS DISTRIBUTION,
            AND SPECIES-AREA CURVES
============================================================

For Mark meeting — three connected deliverables:

  1. Hexagonal grid species richness map for Tanzania
  2. Records per species distribution + reliability flags
  3. Species-area curves overall and by habitat

Inputs:
  - tanzania_points.csv  (or full Tanzania records)

Outputs (in 'tanzania_diversity_maps/'):
  - 01_hex_richness_map.png           — main richness map
  - 02_hex_records_density_map.png    — records per hex (effort)
  - 03_hex_richness_per_record.png    — bias-corrected richness
  - 04_records_per_species.png        — distribution + threshold flags
  - 05_species_area_curves.png        — accumulation curves
  - hex_diversity_grid.geojson        — hex grid with stats
  - reliable_species_list.csv         — species with >= threshold records

Dependencies:
  pip install pandas numpy matplotlib geopandas shapely h3
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\liams\Documents\PhD-Project\Data\RAINBIO\tanzania_points.csv"
OUTPUT_DIR = r"C:\Users\liams\Documents\PhD-Project\Data\RAINBIO\tanzania_diversity_maps"

HEX_RESOLUTION = 4         # H3 resolution: 4 ≈ 1770 km² hex (~42km edge)
                           # 5 ≈ 252 km² hex (~16km edge) — finer, slower
                           # 3 ≈ 12,393 km² hex — coarser, faster
RELIABILITY_THRESHOLD = 5  # Min records per species for SDM reliability
N_ACCUMULATION_DRAWS = 100 # Bootstrap iterations for accumulation curves

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("TANZANIA DIVERSITY ANALYSIS")
print("=" * 70)


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("\n[1] LOADING RAINBIO TANZANIA DATA")
print("-" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip()

# Auto-detect columns
lat_col = next((c for c in df.columns
                if c.lower() in ("decimallatitude", "decimallat", "latitude", "lat")), None)
lon_col = next((c for c in df.columns
                if c.lower() in ("decimallongitude", "decimallong", "longitude", "lon")), None)
species_col = next((c for c in df.columns
                    if c.lower() in ("species", "scientificname")), None)
habit_col = next((c for c in df.columns
                  if c.lower() in ("a_habit", "habit")), None)
family_col = next((c for c in df.columns
                   if c.lower() in ("family",)), None)

print(f"    Records              : {len(df):,}")
print(f"    Lat / Lon columns    : {lat_col} / {lon_col}")
print(f"    Species / Family     : {species_col} / {family_col}")
print(f"    Habit column         : {habit_col}")

# Drop rows missing essentials
df = df.dropna(subset=[lat_col, lon_col, species_col])
print(f"    After dropping NAs   : {len(df):,}")
print(f"    Unique species       : {df[species_col].nunique():,}")


# ============================================================
# STEP 2 — BUILD HEXAGONAL GRID OVER TANZANIA
# ============================================================

print(f"\n[2] BUILDING HEXAGONAL GRID (H3 resolution {HEX_RESOLUTION})")
print("-" * 70)

try:
    import h3
    USE_H3 = True
    print(f"    Using H3 library — uniform-area hexagons")
except ImportError:
    USE_H3 = False
    print(f"    H3 not installed — falling back to manual hex grid")
    print(f"    For better results: pip install h3")


def assign_h3(row):
    """Assign each record to an H3 hexagon."""
    return h3.latlng_to_cell(row[lat_col], row[lon_col], HEX_RESOLUTION)


def manual_hex_grid(bounds, hex_size_deg=0.5):
    """Build a manual hexagonal grid as a fallback."""
    lon_min, lat_min, lon_max, lat_max = bounds
    hex_w = hex_size_deg * np.sqrt(3)
    hex_h = hex_size_deg * 1.5
    hexes = []
    row = 0
    lat = lat_min - hex_size_deg
    while lat <= lat_max + hex_size_deg:
        lon_offset = (hex_w / 2) if row % 2 else 0
        lon = lon_min - hex_size_deg + lon_offset
        while lon <= lon_max + hex_size_deg:
            angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
            verts = [(lon + hex_size_deg * np.cos(a),
                      lat + hex_size_deg * np.sin(a)) for a in angles]
            hexes.append(Polygon(verts))
            lon += hex_w
        lat += hex_h
        row += 1
    return hexes


if USE_H3:
    print(f"    Assigning records to H3 hexagons...")
    df["hex_id"] = df.apply(assign_h3, axis=1)

    # Aggregate per hex
    hex_stats = (
        df.groupby("hex_id")
        .agg(
            n_records=(species_col, "size"),
            n_species=(species_col, "nunique"),
            n_families=(family_col, "nunique") if family_col else (species_col, "nunique"),
        )
        .reset_index()
    )

    # Build polygons for each hex
    def hex_polygon(hex_id):
        boundary = h3.cell_to_boundary(hex_id)
        # H3 returns (lat, lon), shapely needs (lon, lat)
        return Polygon([(lon, lat) for lat, lon in boundary])

    hex_stats["geometry"] = hex_stats["hex_id"].apply(hex_polygon)
    hex_gdf = gpd.GeoDataFrame(hex_stats, geometry="geometry", crs="EPSG:4326")

    # Hex area (km²) — H3 res 4 ≈ 1770 km²
    h3_areas = {3: 12393.43, 4: 1770.35, 5: 252.90, 6: 36.13}
    hex_area_km2 = h3_areas.get(HEX_RESOLUTION, 1770.35)

else:
    # Fallback manual grid
    bounds = (df[lon_col].min(), df[lat_col].min(),
              df[lon_col].max(), df[lat_col].max())
    hex_polys = manual_hex_grid(bounds, hex_size_deg=0.4)
    hex_gdf = gpd.GeoDataFrame({"hex_id": range(len(hex_polys))},
                                geometry=hex_polys, crs="EPSG:4326")

    points_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(points_gdf, hex_gdf, predicate="within")
    hex_stats = (
        joined.groupby("hex_id")
        .agg(
            n_records=(species_col, "size"),
            n_species=(species_col, "nunique"),
            n_families=(family_col, "nunique") if family_col else (species_col, "nunique"),
        )
        .reset_index()
    )
    hex_gdf = hex_gdf.merge(hex_stats, on="hex_id", how="left").fillna(0)
    hex_gdf = hex_gdf[hex_gdf["n_records"] > 0]
    hex_area_km2 = (0.4 ** 2) * np.sqrt(3) * 1.5 * 12321  # rough

# Bias-corrected richness — richness per log(records)
hex_gdf["richness_per_log_records"] = (
    hex_gdf["n_species"] / np.log10(hex_gdf["n_records"] + 1)
)

print(f"    Hexagons with data   : {len(hex_gdf):,}")
print(f"    Approx area per hex  : ~{hex_area_km2:,.0f} km²")
print(f"    Total records mapped : {hex_gdf['n_records'].sum():,}")
print(f"    Richness range       : {hex_gdf['n_species'].min():.0f} – {hex_gdf['n_species'].max():.0f} species")


# ============================================================
# STEP 3 — TANZANIA OUTLINE (FAO GAUL OR FALLBACK)
# ============================================================

print("\n[3] LOADING TANZANIA BOUNDARY")
print("-" * 70)

# Try Natural Earth via geopandas, then fall back to bounding box
try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    tanzania = world[world["name"] == "Tanzania"]
    if len(tanzania) > 0:
        print(f"    Loaded Tanzania boundary from Natural Earth")
    else:
        raise ValueError("Tanzania not found")
except Exception as e:
    print(f"    Natural Earth unavailable ({e}) — using bounding box")
    bbox = Polygon([
        (29.5, -11.8), (40.5, -11.8),
        (40.5, -0.95), (29.5, -0.95),
    ])
    tanzania = gpd.GeoDataFrame({"name": ["Tanzania"]},
                                 geometry=[bbox], crs="EPSG:4326")


# ============================================================
# STEP 4 — FIGURE 1: HEXAGONAL SPECIES RICHNESS MAP
# ============================================================

print("\n[4] BUILDING SPECIES RICHNESS MAP")
print("-" * 70)

fig, ax = plt.subplots(figsize=(11, 10))

# Tanzania outline
tanzania.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)

# Hexagons coloured by richness
hex_gdf.plot(
    ax=ax,
    column="n_species",
    cmap="YlGnBu",
    edgecolor="white",
    linewidth=0.2,
    legend=True,
    legend_kwds={"label": "Species richness", "shrink": 0.6, "orientation": "vertical"},
    zorder=2,
)

ax.set_xlim(28.5, 41.5)
ax.set_ylim(-12.5, 0)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    f"Tanzania — Plant Species Richness (RAINBIO)\n"
    f"H3 resolution {HEX_RESOLUTION} (~{hex_area_km2:,.0f} km²/hex), "
    f"{len(hex_gdf):,} hexagons with records",
    fontsize=13, fontweight="bold",
)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_hex_richness_map.png", dpi=200, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/01_hex_richness_map.png")
plt.close()


# ============================================================
# STEP 5 — FIGURE 2: RECORD DENSITY MAP (SAMPLING EFFORT)
# ============================================================

print("\n[5] BUILDING RECORD DENSITY (SAMPLING EFFORT) MAP")
print("-" * 70)

fig, ax = plt.subplots(figsize=(11, 10))

tanzania.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)

hex_gdf.plot(
    ax=ax,
    column="n_records",
    cmap="OrRd",
    edgecolor="white",
    linewidth=0.2,
    legend=True,
    legend_kwds={"label": "Records (log scale)", "shrink": 0.6},
    norm=LogNorm(vmin=max(1, hex_gdf["n_records"].min()),
                 vmax=hex_gdf["n_records"].max()),
    zorder=2,
)

ax.set_xlim(28.5, 41.5)
ax.set_ylim(-12.5, 0)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    "Tanzania — Sampling Effort (Records per Hex)\n"
    "Heavy sampling biases richness — compare against Figure 1",
    fontsize=13, fontweight="bold",
)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_hex_records_density_map.png", dpi=200, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/02_hex_records_density_map.png")
plt.close()


# ============================================================
# STEP 6 — FIGURE 3: BIAS-CORRECTED RICHNESS
# ============================================================

print("\n[6] BUILDING BIAS-CORRECTED RICHNESS MAP")
print("-" * 70)

fig, ax = plt.subplots(figsize=(11, 10))

tanzania.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)

hex_gdf.plot(
    ax=ax,
    column="richness_per_log_records",
    cmap="PuBuGn",
    edgecolor="white",
    linewidth=0.2,
    legend=True,
    legend_kwds={"label": "Species per log(records+1)", "shrink": 0.6},
    zorder=2,
)

ax.set_xlim(28.5, 41.5)
ax.set_ylim(-12.5, 0)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    "Tanzania — Bias-corrected Richness\n"
    "Species per log(records) — controls for uneven sampling effort",
    fontsize=13, fontweight="bold",
)
ax.grid(alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_hex_richness_per_record.png", dpi=200, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/03_hex_richness_per_record.png")
plt.close()

# Save hex grid as geojson
hex_gdf.drop(columns=[c for c in ["geometry"] if c not in hex_gdf.columns]).to_file(
    f"{OUTPUT_DIR}/hex_diversity_grid.geojson", driver="GeoJSON"
)
print(f"    Saved: {OUTPUT_DIR}/hex_diversity_grid.geojson")


# ============================================================
# STEP 7 — RECORDS PER SPECIES DISTRIBUTION
# ============================================================

print("\n[7] RECORDS PER SPECIES DISTRIBUTION")
print("-" * 70)

records_per_species = df.groupby(species_col).size().sort_values(ascending=False)

# Reliability classification
n_total = len(records_per_species)
n_singleton = (records_per_species == 1).sum()
n_unreliable = (records_per_species < RELIABILITY_THRESHOLD).sum()
n_reliable = (records_per_species >= RELIABILITY_THRESHOLD).sum()
n_well_sampled = (records_per_species >= 20).sum()

print(f"    Total species          : {n_total:,}")
print(f"    Singletons (1 record)  : {n_singleton:,} ({n_singleton/n_total*100:.1f}%)")
print(f"    Unreliable (<{RELIABILITY_THRESHOLD} records): {n_unreliable:,} ({n_unreliable/n_total*100:.1f}%)")
print(f"    Reliable (≥{RELIABILITY_THRESHOLD} records)   : {n_reliable:,} ({n_reliable/n_total*100:.1f}%)")
print(f"    Well-sampled (≥20)     : {n_well_sampled:,} ({n_well_sampled/n_total*100:.1f}%)")
print(f"    Median records/species : {records_per_species.median():.0f}")
print(f"    Mean records/species   : {records_per_species.mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Linear scale histogram
ax = axes[0]
capped = records_per_species.clip(upper=50)
ax.hist(capped, bins=50, color="steelblue", edgecolor="white")
ax.axvline(RELIABILITY_THRESHOLD, color="red", linestyle="--", linewidth=2,
           label=f"Reliability threshold ({RELIABILITY_THRESHOLD} records)")
ax.axvline(records_per_species.median(), color="green", linestyle=":", linewidth=2,
           label=f"Median: {records_per_species.median():.0f}")
ax.set_xlabel("Records per species (capped at 50)")
ax.set_ylabel("Number of species")
ax.set_title("Records per Species — Linear Scale", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)

# Log scale rank-abundance
ax = axes[1]
ranks = np.arange(1, len(records_per_species) + 1)
ax.loglog(ranks, records_per_species.values, "o", markersize=2, color="darkblue", alpha=0.5)
ax.axhline(RELIABILITY_THRESHOLD, color="red", linestyle="--", linewidth=2,
           label=f"Reliability threshold ({RELIABILITY_THRESHOLD})")
ax.set_xlabel("Species rank (log)")
ax.set_ylabel("Records per species (log)")
ax.set_title(f"Rank-Abundance Curve — {n_total:,} species\n"
             f"{n_unreliable:,} ({n_unreliable/n_total*100:.0f}%) below threshold",
             fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_records_per_species.png", dpi=200, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/04_records_per_species.png")
plt.close()

# Export reliable species list
reliable = records_per_species[records_per_species >= RELIABILITY_THRESHOLD]
reliable_df = pd.DataFrame({
    "species": reliable.index,
    "n_records": reliable.values,
})
if family_col:
    fam_lookup = df.groupby(species_col)[family_col].first()
    reliable_df["family"] = reliable_df["species"].map(fam_lookup)
if habit_col:
    habit_lookup = df.groupby(species_col)[habit_col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    reliable_df["habit"] = reliable_df["species"].map(habit_lookup)

reliable_df.to_csv(f"{OUTPUT_DIR}/reliable_species_list.csv", index=False)
print(f"    Saved: {OUTPUT_DIR}/reliable_species_list.csv ({len(reliable_df):,} species)")


# ============================================================
# STEP 8 — SPECIES-AREA CURVES
# ============================================================

print("\n[8] BUILDING SPECIES-AREA CURVES")
print("-" * 70)


def species_accumulation(records_df, species_column, n_draws=100, n_steps=50):
    """
    Build a species accumulation curve.
    Returns mean and 95% bands of species count vs sample size.
    """
    n_records = len(records_df)
    if n_records < 10:
        return None
    sample_sizes = np.unique(np.linspace(1, n_records, n_steps).astype(int))
    species_counts = np.zeros((n_draws, len(sample_sizes)))

    species_array = records_df[species_column].values
    for d in range(n_draws):
        shuffled = np.random.permutation(species_array)
        for i, s in enumerate(sample_sizes):
            species_counts[d, i] = len(np.unique(shuffled[:s]))

    return {
        "sample_sizes": sample_sizes,
        "mean": species_counts.mean(axis=0),
        "lower": np.percentile(species_counts, 2.5, axis=0),
        "upper": np.percentile(species_counts, 97.5, axis=0),
    }


def area_accumulation(records_df, lat_c, lon_c, species_column, n_draws=20):
    """
    Build a species-area curve by aggregating random hex subsets.
    """
    if "hex_id" not in records_df.columns:
        return None
    hex_ids = records_df["hex_id"].unique()
    n_hexes = len(hex_ids)
    sample_sizes = np.unique(np.linspace(1, n_hexes, 30).astype(int))
    counts = np.zeros((n_draws, len(sample_sizes)))

    for d in range(n_draws):
        shuffled = np.random.permutation(hex_ids)
        for i, s in enumerate(sample_sizes):
            subset_hex = shuffled[:s]
            mask = records_df["hex_id"].isin(subset_hex)
            counts[d, i] = records_df.loc[mask, species_column].nunique()

    return {
        "n_hexes": sample_sizes,
        "area_km2": sample_sizes * hex_area_km2,
        "mean": counts.mean(axis=0),
        "lower": np.percentile(counts, 2.5, axis=0),
        "upper": np.percentile(counts, 97.5, axis=0),
    }


# Overall accumulation curve (records-based)
print(f"    Computing overall species accumulation...")
overall_curve = species_accumulation(df, species_col, n_draws=N_ACCUMULATION_DRAWS)

# Per-habitat accumulation curves
print(f"    Computing per-habitat accumulation curves...")
habitat_curves = {}
if habit_col and df[habit_col].notna().any():
    top_habits = df[habit_col].value_counts().head(5).index
    for habit in top_habits:
        habit_df = df[df[habit_col] == habit]
        if len(habit_df) >= 50:
            habitat_curves[habit] = species_accumulation(
                habit_df, species_col, n_draws=N_ACCUMULATION_DRAWS
            )

# Area-based accumulation
if USE_H3:
    print(f"    Computing species-area curve (hex-based)...")
    area_curve = area_accumulation(df, lat_col, lon_col, species_col, n_draws=20)
else:
    area_curve = None

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 8a — Overall + per-habitat (records-based)
ax = axes[0]
if overall_curve:
    ax.fill_between(overall_curve["sample_sizes"],
                    overall_curve["lower"], overall_curve["upper"],
                    alpha=0.2, color="black")
    ax.plot(overall_curve["sample_sizes"], overall_curve["mean"],
            color="black", linewidth=2.5, label=f"All Tanzania ({df[species_col].nunique():,} species)")

colors = plt.cm.tab10(np.linspace(0, 1, len(habitat_curves)))
for (habit, curve), color in zip(habitat_curves.items(), colors):
    ax.fill_between(curve["sample_sizes"], curve["lower"], curve["upper"],
                    alpha=0.15, color=color)
    n_sp = int(curve["mean"][-1])
    ax.plot(curve["sample_sizes"], curve["mean"],
            color=color, linewidth=1.8, label=f"{habit} ({n_sp:,} species)")

ax.set_xlabel("Number of records (random draw)")
ax.set_ylabel("Cumulative species count")
ax.set_title("Species Accumulation by Records — Overall and by Habit",
             fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# 8b — Area-based curve
ax = axes[1]
if area_curve is not None:
    ax.fill_between(area_curve["area_km2"], area_curve["lower"],
                    area_curve["upper"], alpha=0.2, color="darkgreen")
    ax.plot(area_curve["area_km2"], area_curve["mean"],
            color="darkgreen", linewidth=2.5,
            label=f"Tanzania ({df[species_col].nunique():,} species total)")
    ax.set_xlabel("Area sampled (km², log)")
    ax.set_ylabel("Cumulative species count")
    ax.set_xscale("log")
    ax.set_title(f"Species-Area Curve — Hex-based\n"
                 f"Approaching saturation? Compare end slope.",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")
else:
    ax.text(0.5, 0.5, "Area-based curve unavailable\n(H3 library required)",
            ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_species_area_curves.png", dpi=200, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/05_species_area_curves.png")
plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\n    All outputs saved to:")
print(f"    {OUTPUT_DIR}\n")
print(f"    Files:")
print(f"      01_hex_richness_map.png         — main richness map")
print(f"      02_hex_records_density_map.png  — sampling effort (compare to 01)")
print(f"      03_hex_richness_per_record.png  — bias-corrected richness")
print(f"      04_records_per_species.png      — records distribution")
print(f"      05_species_area_curves.png      — accumulation curves")
print(f"      hex_diversity_grid.geojson      — hex grid with stats")
print(f"      reliable_species_list.csv       — {len(reliable_df):,} species ≥{RELIABILITY_THRESHOLD} records")
print()
print("    For the meeting with Mark:")
print(f"      • {n_total:,} total species; {n_reliable:,} ({n_reliable/n_total*100:.0f}%) have ≥{RELIABILITY_THRESHOLD} records")
print(f"      • Singletons: {n_singleton:,} ({n_singleton/n_total*100:.0f}%) — likely unreliable")
print(f"      • Hex grid: {len(hex_gdf):,} hexagons with records, ~{hex_area_km2:,.0f} km² each")
print(f"      • Richness range across hexagons: {hex_gdf['n_species'].min():.0f}–{hex_gdf['n_species'].max():.0f}")
print("=" * 70)