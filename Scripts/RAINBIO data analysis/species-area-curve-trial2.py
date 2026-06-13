import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

# ============================================
# 1. LOAD YOUR DATA
# ============================================

df = pd.read_csv(r"C:\Users\liams\Documents\PhD-Project Data\tanzania\tanzania_points.csv")

# Make sure column names match
# CHANGE if needed
species_col = "species"
lat_col = "decimalLatitude"
lon_col = "decimalLongitude"

# ============================================
# 2. CONVERT TO GEODATAFRAME
# ============================================

geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# ============================================
# 3. PROJECT TO METRIC CRS (important!)
# ============================================

gdf = gdf.to_crs(epsg=3395)  # meters

# ============================================
# 4. CREATE GRID (adjust resolution here)
# ============================================

grid_size = 50000  # 50 km

xmin, ymin, xmax, ymax = gdf.total_bounds

cols = np.arange(xmin, xmax, grid_size)
rows = np.arange(ymin, ymax, grid_size)

polygons = []
for x in cols:
    for y in rows:
        polygons.append(
            box(x, y, x + grid_size, y + grid_size)
        )

grid = gpd.GeoDataFrame(geometry=polygons, crs=gdf.crs)

# ============================================
# 5. SPATIAL JOIN (points → grid cells)
# ============================================

joined = gpd.sjoin(gdf, grid, how="left", predicate="within")

# Each grid cell gets an ID
joined["cell_id"] = joined["index_right"]

# ============================================
# 6. SPECIES PER CELL
# ============================================

cell_species = joined.groupby("cell_id")[species_col].apply(set)

# Remove empty cells
cell_species = cell_species.dropna()

cells = list(cell_species.index)

# ============================================
# 7. SPECIES–AREA CURVE
# ============================================

n_reps = 50  # number of random permutations
results = []

for _ in range(n_reps):
    np.random.shuffle(cells)
    
    seen_species = set()
    species_counts = []
    area_counts = []
    
    for i, cell in enumerate(cells):
        seen_species.update(cell_species[cell])
        
        species_counts.append(len(seen_species))
        area_counts.append((i + 1) * grid_size * grid_size / 1e6)  # km²
    
    results.append(species_counts)

# Convert to array
results = np.array(results)

# Mean + std
mean_species = results.mean(axis=0)
std_species = 0

# ============================================
# 8. PLOT
# ============================================

plt.figure(figsize=(6, 5))

plt.plot(area_counts, mean_species, label="Mean")
plt.fill_between(
    area_counts,
    mean_species - std_species,
    mean_species + std_species,
    alpha=0.3
)

plt.xlabel("Area (km²)")
plt.ylabel("Cumulative species richness")
plt.title("Species–Area Curve (Tanzania RainBio)")
plt.legend()

plt.show()