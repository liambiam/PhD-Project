"""
============================================================
ALPHAEARTH EXPLORATORY ANALYSIS — TANZANIA
============================================================
Foundation model embeddings for plant functional ecology PhD

What this script does:
  1. Connects to Google Earth Engine
  2. Loads the AlphaEarth Satellite Embedding dataset for Tanzania
  3. Visualises embeddings as RGB composites
  4. Samples embeddings at points across Tanzania
  5. Clusters Tanzania into ecological zones using embeddings
  6. Overlays RAINBIO species data with embedding clusters
  7. Compares embedding stability across years (resilience proxy)
  8. Exports results for presentation to supervisors

Setup (one-time):
  pip install earthengine-api geemap geopandas pandas numpy scikit-learn
  pip install matplotlib seaborn rasterio folium

  Then authenticate (only once):
  >>> import ee
  >>> ee.Authenticate()

Run order: top to bottom. Each section is self-contained.
============================================================
"""

import ee
import geemap
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID    = "phdproject-494613"   # <-- set your GEE project
RAINBIO_CSV   = "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/tanzania_points.csv"  # <-- your filtered RAINBIO data
OUTPUT_DIR    = "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/alphaearth_outputs"

YEAR          = 2024     # AlphaEarth annual layer to use
N_SAMPLE_PTS  = 5000     # Number of points to sample for clustering
N_CLUSTERS    = 8        # Number of ecological zones to cluster into
RESILIENCE_YEARS = [2018, 2024]  # Years to compare for stability analysis

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SECTION 1 — INITIALISE EARTH ENGINE
# ============================================================

print("=" * 60)
print("[1] INITIALISING EARTH ENGINE")
print("=" * 60)

try:
    ee.Initialize(project=PROJECT_ID)
    print(f"    Connected to GEE project: {PROJECT_ID}")
except Exception as e:
    print(f"    Authentication needed. Run: ee.Authenticate()")
    print(f"    Error: {e}")
    raise


# ============================================================
# SECTION 2 — DEFINE TANZANIA STUDY AREA
# ============================================================

print("\n" + "=" * 60)
print("[2] DEFINING TANZANIA STUDY AREA")
print("=" * 60)

# Use FAO GAUL boundaries for Tanzania
countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
tanzania = countries.filter(ee.Filter.eq("ADM0_NAME", "United Republic of Tanzania"))

bounds = tanzania.geometry().bounds().getInfo()
print(f"    Tanzania bounding box: {bounds['coordinates']}")
print(f"    Area: ~947,300 km² (continental Tanzania)")


# ============================================================
# SECTION 3 — LOAD ALPHAEARTH EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print(f"[3] LOADING ALPHAEARTH EMBEDDINGS FOR {YEAR}")
print("=" * 60)

# The Satellite Embedding dataset
embeddings = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

# Filter to Tanzania and the chosen year
ae_tanzania = (
    embeddings
    .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
    .filterBounds(tanzania)
    .mosaic()
    .clip(tanzania)
)

# AlphaEarth has 64 embedding dimensions named A00..A63
band_names = [f"A{i:02d}" for i in range(64)]
print(f"    Bands: {len(band_names)} embedding dimensions (A00–A63)")
print(f"    Resolution: 10m × 10m")
print(f"    Temporal aggregation: full year {YEAR}")


# ============================================================
# SECTION 4 — VISUALISE EMBEDDINGS AS RGB COMPOSITES
# ============================================================

print("\n" + "=" * 60)
print("[4] CREATING RGB VISUALISATIONS")
print("=" * 60)

# Three different RGB combinations to explore the embedding space
# Different bands highlight different ecological structure
rgb_combinations = {
    "rgb_combo_1": {"bands": ["A01", "A16", "A09"], "label": "Standard composite"},
    "rgb_combo_2": {"bands": ["A03", "A23", "A47"], "label": "Vegetation structure"},
    "rgb_combo_3": {"bands": ["A05", "A30", "A55"], "label": "Phenological signal"},
}

# Build an interactive map
print("    Building interactive map (will open in browser)...")
Map = geemap.Map(center=[-6.0, 35.0], zoom=6)

# Add each RGB combo as a layer
for name, combo in rgb_combinations.items():
    vis_params = {
        "bands": combo["bands"],
        "min": -0.3,
        "max": 0.3,
    }
    Map.addLayer(ae_tanzania, vis_params, f"{name}: {combo['label']}")

Map.addLayer(tanzania, {"color": "yellow"}, "Tanzania border", False)

# Save the map as HTML for sharing
map_path = f"{OUTPUT_DIR}/tanzania_alphaearth_map.html"
Map.to_html(map_path)
print(f"    Interactive map saved: {map_path}")
print(f"    Open this file in a browser to explore visually")


# ============================================================
# SECTION 5 — SAMPLE EMBEDDINGS AT RANDOM POINTS
# ============================================================

print("\n" + "=" * 60)
print(f"[5] SAMPLING {N_SAMPLE_PTS} POINTS ACROSS TANZANIA")
print("=" * 60)

# Generate stratified random sample across Tanzania
sample = ae_tanzania.sample(
    region=tanzania.geometry(),
    scale=1000,            # 1km — coarser than native 10m for tractability
    numPixels=N_SAMPLE_PTS,
    geometries=True,
    seed=42,
)

# Pull data to client (this is the slow step)
print("    Extracting samples from Earth Engine — this can take 1-3 mins...")
sample_features = sample.getInfo()["features"]
print(f"    Retrieved {len(sample_features)} sample points")

# Convert to dataframe
records = []
for f in sample_features:
    props = f["properties"]
    coords = f["geometry"]["coordinates"]
    record = {"lon": coords[0], "lat": coords[1]}
    for b in band_names:
        record[b] = props.get(b, np.nan)
    records.append(record)

df = pd.DataFrame(records).dropna()
print(f"    Final sample after removing nulls: {len(df)}")
print(f"    Embedding dimensions per point: {len(band_names)}")

# Save raw embeddings
df.to_csv(f"{OUTPUT_DIR}/tanzania_embeddings_sample.csv", index=False)
print(f"    Saved: {OUTPUT_DIR}/tanzania_embeddings_sample.csv")


# ============================================================
# SECTION 6 — DIMENSIONALITY REDUCTION & VISUALISATION
# ============================================================

print("\n" + "=" * 60)
print("[6] DIMENSIONALITY REDUCTION (PCA)")
print("=" * 60)

# Standardise embeddings before PCA
X = df[band_names].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Run PCA — first 10 components for diagnostics
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_std)

# Variance explained
var_explained = pca.explained_variance_ratio_
cum_var = np.cumsum(var_explained)
print(f"    Variance explained by first 10 PCs:")
for i, (v, c) in enumerate(zip(var_explained, cum_var)):
    print(f"      PC{i+1}: {v:.3f} ({c:.3f} cumulative)")

# Plot PCA results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, 11), var_explained, color="steelblue", edgecolor="white")
axes[0].plot(range(1, 11), cum_var, "ro-", linewidth=1.5, markersize=6, label="Cumulative")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained")
axes[0].set_title("PCA Scree Plot — AlphaEarth Embeddings")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Geographic plot of PC1
sc = axes[1].scatter(df["lon"], df["lat"], c=X_pca[:, 0], cmap="viridis",
                      s=3, alpha=0.7)
plt.colorbar(sc, ax=axes[1], label="PC1 score")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("PC1 (largest source of variation) — Spatial Pattern")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_pca_analysis.png", dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/01_pca_analysis.png")
plt.close()


# ============================================================
# SECTION 7 — UNSUPERVISED CLUSTERING INTO ECOLOGICAL ZONES
# ============================================================

print("\n" + "=" * 60)
print(f"[7] CLUSTERING TANZANIA INTO {N_CLUSTERS} ECOLOGICAL ZONES")
print("=" * 60)

# K-means on the standardised embeddings
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_std)

# Cluster sizes
cluster_counts = df["cluster"].value_counts().sort_index()
print(f"    Cluster sizes:")
for c, n in cluster_counts.items():
    print(f"      Cluster {c}: {n:,} points ({n/len(df)*100:.1f}%)")

# Plot clusters on map
fig, ax = plt.subplots(figsize=(10, 8))
colours = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))

for c in range(N_CLUSTERS):
    cluster_pts = df[df["cluster"] == c]
    ax.scatter(cluster_pts["lon"], cluster_pts["lat"],
               c=[colours[c]], s=4, alpha=0.7, label=f"Zone {c}")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Tanzania — Data-driven Ecological Zones from AlphaEarth Embeddings\n"
             f"K-means clustering on 64-dim embedding space, k={N_CLUSTERS}")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
ax.set_aspect("equal")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_ecological_zones.png", dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/02_ecological_zones.png")
plt.close()

# Save cluster assignments
df.to_csv(f"{OUTPUT_DIR}/tanzania_embeddings_clustered.csv", index=False)


# ============================================================
# SECTION 8 — RAINBIO OVERLAY (IF AVAILABLE)
# ============================================================

print("\n" + "=" * 60)
print("[8] RAINBIO SPECIES OVERLAY")
print("=" * 60)

if os.path.exists(RAINBIO_CSV):
    print(f"    Loading RAINBIO data from {RAINBIO_CSV}...")
    rainbio = pd.read_csv(RAINBIO_CSV, low_memory=False)
    rainbio = rainbio.dropna(subset=["decimalLatitude", "decimalLongitude"])

    # Convert to GeoDataFrame for spatial join
    rainbio_gdf = gpd.GeoDataFrame(
        rainbio,
        geometry=gpd.points_from_xy(rainbio["decimalLongitude"], rainbio["decimalLatitude"]),
        crs="EPSG:4326",
    )

    # Convert sample points to GeoDataFrame
    sample_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    # Spatial join — assign each RAINBIO record to its nearest cluster
    print("    Performing spatial join (nearest neighbour)...")
    joined = gpd.sjoin_nearest(rainbio_gdf, sample_gdf[["cluster", "geometry"]],
                                max_distance=0.1, how="left")

    print(f"    {joined['cluster'].notna().sum():,} RAINBIO records matched to a cluster")

    # Species richness per cluster
    richness_per_cluster = joined.groupby("cluster")["species"].nunique().sort_values(ascending=False)
    records_per_cluster = joined.groupby("cluster").size().sort_values(ascending=False)

    print(f"\n    Species richness per ecological zone:")
    for c, n in richness_per_cluster.items():
        print(f"      Zone {int(c)}: {n:,} unique species, {records_per_cluster[c]:,} records")

    # Family composition per cluster
    family_top = joined.groupby(["cluster", "family"]).size().reset_index(name="n")
    family_pivot = family_top.pivot(index="family", columns="cluster", values="n").fillna(0)

    # Top 15 families overall
    top_families = family_pivot.sum(axis=1).sort_values(ascending=False).head(15).index
    family_top_pivot = family_pivot.loc[top_families]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(family_top_pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(family_top_pivot.columns)))
    ax.set_xticklabels([f"Zone {int(c)}" for c in family_top_pivot.columns])
    ax.set_yticks(range(len(family_top_pivot.index)))
    ax.set_yticklabels(family_top_pivot.index)
    ax.set_title("Top 15 Plant Families × Ecological Zone\n(record count, RAINBIO)")
    plt.colorbar(im, ax=ax, label="Number of records")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_family_by_zone.png", dpi=150, bbox_inches="tight")
    print(f"    Saved: {OUTPUT_DIR}/03_family_by_zone.png")
    plt.close()

    # Combined map: zones + species records
    fig, ax = plt.subplots(figsize=(11, 9))
    for c in range(N_CLUSTERS):
        cluster_pts = df[df["cluster"] == c]
        ax.scatter(cluster_pts["lon"], cluster_pts["lat"],
                   c=[colours[c]], s=10, alpha=0.5, label=f"Zone {c}")
    # Overlay species records (small, dark)
    rb_subset = rainbio.sample(min(50000, len(rainbio)), random_state=42)
    ax.scatter(rb_subset["decimalLongitude"], rb_subset["decimalLatitude"],
               c="black", s=0.5, alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Tanzania — AlphaEarth Ecological Zones with RAINBIO Records (black)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_zones_with_records.png", dpi=150, bbox_inches="tight")
    print(f"    Saved: {OUTPUT_DIR}/04_zones_with_records.png")
    plt.close()
else:
    print(f"    RAINBIO CSV not found at {RAINBIO_CSV} — skipping overlay")
    print(f"    (Update RAINBIO_CSV path at top of script to include this step)")


# ============================================================
# SECTION 9 — TEMPORAL STABILITY (RESILIENCE PROXY)
# ============================================================

print("\n" + "=" * 60)
print(f"[9] TEMPORAL STABILITY: {RESILIENCE_YEARS[0]} vs {RESILIENCE_YEARS[1]}")
print("=" * 60)

print("    Loading embeddings for both years...")
ae_year1 = (embeddings
            .filterDate(f"{RESILIENCE_YEARS[0]}-01-01", f"{RESILIENCE_YEARS[0]+1}-01-01")
            .filterBounds(tanzania)
            .mosaic()
            .clip(tanzania))

ae_year2 = (embeddings
            .filterDate(f"{RESILIENCE_YEARS[1]}-01-01", f"{RESILIENCE_YEARS[1]+1}-01-01")
            .filterBounds(tanzania)
            .mosaic()
            .clip(tanzania))

# Cosine similarity between the two years' embeddings
# = a per-pixel measure of how much the pixel changed
print("    Computing per-pixel cosine similarity between years...")

def cosine_similarity(img1, img2, band_names):
    """Cosine similarity between two embedding images, band-wise."""
    dot = ee.Image(0)
    norm1_sq = ee.Image(0)
    norm2_sq = ee.Image(0)
    for b in band_names:
        v1 = img1.select(b)
        v2 = img2.select(b)
        dot = dot.add(v1.multiply(v2))
        norm1_sq = norm1_sq.add(v1.multiply(v1))
        norm2_sq = norm2_sq.add(v2.multiply(v2))
    norm1 = norm1_sq.sqrt()
    norm2 = norm2_sq.sqrt()
    return dot.divide(norm1.multiply(norm2)).rename("similarity")

similarity = cosine_similarity(ae_year1, ae_year2, band_names)

# Sample similarity at our existing points
sim_sample = similarity.sampleRegions(
    collection=sample,
    scale=1000,
    geometries=False,
)

print("    Extracting similarity scores...")
sim_features = sim_sample.getInfo()["features"]
sim_values = [f["properties"].get("similarity") for f in sim_features]
sim_values = [v for v in sim_values if v is not None]

print(f"    Mean similarity ({RESILIENCE_YEARS[0]} vs {RESILIENCE_YEARS[1]}): {np.mean(sim_values):.4f}")
print(f"    Std similarity:    {np.std(sim_values):.4f}")
print(f"    Range: {min(sim_values):.4f} — {max(sim_values):.4f}")
print(f"    (Closer to 1 = more stable; lower = more change)")

# Plot histogram of stability
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sim_values, bins=50, color="darkgreen", edgecolor="white", alpha=0.7)
ax.axvline(np.mean(sim_values), color="red", linestyle="--",
           label=f"Mean: {np.mean(sim_values):.3f}")
ax.set_xlabel(f"Cosine similarity ({RESILIENCE_YEARS[0]} → {RESILIENCE_YEARS[1]})")
ax.set_ylabel("Number of pixels")
ax.set_title(f"Tanzania — AlphaEarth Embedding Stability {RESILIENCE_YEARS[0]} → {RESILIENCE_YEARS[1]}\n"
             f"Higher similarity = more stable pixel = candidate high-resilience location")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_temporal_stability.png", dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/05_temporal_stability.png")
plt.close()


# ============================================================
# SECTION 10 — SUMMARY OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("[10] EXPLORATION COMPLETE — OUTPUTS SUMMARY")
print("=" * 60)

outputs = [
    ("Interactive map (HTML)", "tanzania_alphaearth_map.html"),
    ("Raw embeddings sample", "tanzania_embeddings_sample.csv"),
    ("Embeddings + cluster labels", "tanzania_embeddings_clustered.csv"),
    ("PCA analysis figure", "01_pca_analysis.png"),
    ("Ecological zones map", "02_ecological_zones.png"),
    ("Family-by-zone heatmap", "03_family_by_zone.png"),
    ("Zones + RAINBIO overlay", "04_zones_with_records.png"),
    ("Temporal stability histogram", "05_temporal_stability.png"),
]

for label, filename in outputs:
    full_path = f"{OUTPUT_DIR}/{filename}"
    exists = "✓" if os.path.exists(full_path) else "✗"
    print(f"    [{exists}] {label:35s} {full_path}")

print("\n" + "=" * 60)
print("NEXT STEPS FOR YOUR MEETING WITH MARK:")
print("=" * 60)
print("  1. Open the interactive HTML map and explore visually")
print("  2. Compare ecological zones map vs RAINBIO habitat classifications")
print("  3. Note where species richness aligns/diverges from embedding clusters")
print("  4. Look at temporal stability — are stable pixels in protected areas?")
print("  5. Bring all PNG outputs to the meeting — screen-share friendly")
print("=" * 60)
