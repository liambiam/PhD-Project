"""
Tanzania RAINBIO — occupancy rate & species-richness maps, overall and split
by habitat (habitat_name) and growth form (a_habit).

What it computes
----------------
1. The DENOMINATOR: all H3 cells whose centre falls inside Tanzania, from a
   boundary polygon. This is what 'occupied' is measured against.
2. Occupancy rate = occupied cells / total Tanzania cells, overall and per group.
3. Species-richness choropleth maps (species per occupied cell), overall and
   per group, drawn as H3 hexagons clipped to Tanzania.

Boundary source (pick ONE; the script tries them in order)
----------------------------------------------------------
A. LOCAL FILE  -> set BOUNDARY_PATH to a Tanzania boundary (.shp/.gpkg/.geojson).
   This is the most reliable route (no network, no Python-3.14 package issues).
   Get one from GADM (https://gadm.org, level 0) or Natural Earth.
B. geopandas built-in / naturalearth (older geopandas only) — fallback.

If neither is available the script still reports occupied-cell counts and draws
point-based maps; it just can't compute the occupancy DENOMINATOR without a
boundary. It will tell you which mode it ran in.

CONFIG flagged inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import h3

# ----------------------------------------------------------------------
# CONFIG
CSV_PATH    = r"C:\Users\liams\Documents\PhD-Project Data\tanzania\tanzania_points_with_habitat_labels.csv"
OUTDIR      = r"C:\Users\liams\Documents\PhD-Project Data\Occupancy Maps"
BOUNDARY_PATH = r"C:\Users\liams\Documents\PhD-Project Data\GADM TZ Shape\gadm41_TZA_0.shp"   # <- set to your Tanzania boundary file, e.g.
                      #    r"C:\Users\liams\Documents\PhD-Project Data\boundary\gadm41_TZA_0.shp"
HABITAT_COL = "habitat_name.x"   # habitat column (your file has .x and .y); change if needed
USE_MAJOR_HABITAT = True         # collapse 'Forest – ...' to 'Forest' etc.
GROWTH_COL  = "a_habit"
H3_RES      = 5        # resolution for occupancy + maps
MIN_RECORDS = 20       # skip groups with fewer records than this
CMAP        = "viridis"
# ----------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def major_habitat(name):
    if pd.isna(name):
        return name
    s = str(name)
    for sep in [" – ", " — ", " - ", "–", "—"]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()


def safe_filename(value, maxlen=70):
    import re
    s = str(value)
    s = re.sub(r'[<>:"/\\|?*]', "", s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w\-.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "group"


def cell_to_poly_xy(cell):
    """Return (lons, lats) of an H3 cell boundary for plotting."""
    boundary = h3.cell_to_boundary(cell)   # list of (lat, lng)
    lats = [p[0] for p in boundary]
    lons = [p[1] for p in boundary]
    return lons, lats


# ----------------------------------------------------------------------
# Boundary -> set of all Tanzania cells (the denominator)
# ----------------------------------------------------------------------
def tanzania_cells(res):
    """Return (set_of_all_TZ_cells, boundary_geom_or_None)."""
    geom = None

    # Route A: local boundary file
    if BOUNDARY_PATH and os.path.exists(BOUNDARY_PATH):
        import geopandas as gpd
        gdf = gpd.read_file(BOUNDARY_PATH).to_crs(4326)
        geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
        print(f"  boundary: loaded from {os.path.basename(BOUNDARY_PATH)}")

    # Route B: naturalearth via geopandas (older versions only)
    if geom is None:
        try:
            import geopandas as gpd
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            tz = world[world["name"] == "Tanzania"]
            if len(tz):
                geom = tz.to_crs(4326).union_all() if hasattr(tz, "union_all") else tz.unary_union
                print("  boundary: naturalearth_lowres (geopandas built-in)")
        except Exception as e:
            print(f"  (naturalearth route unavailable: {str(e)[:60]})")

    if geom is None:
        print("  !! No boundary available — occupancy DENOMINATOR cannot be computed.")
        print("     Set BOUNDARY_PATH to a Tanzania boundary file (GADM level 0).")
        return None, None

    # fill the polygon with H3 cells (centre-inside)
    all_cells = set(h3.geo_to_cells(geom, res))
    print(f"  Tanzania contains {len(all_cells):,} H3 cells at res {res}")
    return all_cells, geom


# ----------------------------------------------------------------------
# Occupancy table
# ----------------------------------------------------------------------
def occupancy_table(df, all_cells, group_col=None):
    """Occupancy rate overall or per group."""
    total = len(all_cells) if all_cells else np.nan
    rows = []

    def occ(sub, label):
        occupied = set(sub["cell"]) & all_cells if all_cells else set(sub["cell"])
        n_occ = len(occupied)
        rows.append({
            "group": label,
            "records": len(sub),
            "species": sub["species"].nunique(),
            "occupied_cells": n_occ,
            "total_TZ_cells": total,
            "occupancy_rate": (n_occ / total) if all_cells else np.nan,
        })

    occ(df, "ALL")
    if group_col:
        for g in df[group_col].dropna().unique():
            sub = df[df[group_col] == g]
            if len(sub) >= MIN_RECORDS:
                occ(sub, str(g))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Richness map (H3 hexagons coloured by species count)
# ----------------------------------------------------------------------
def richness_map(df, all_cells, geom, title, outpath, clip_to=None):
    sub = df if clip_to is None else df[df[clip_to[0]] == clip_to[1]]
    if len(sub) < MIN_RECORDS:
        return
    rich = sub.groupby("cell")["species"].nunique()
    if all_cells:
        rich = rich[rich.index.isin(all_cells)]
    if rich.empty:
        return

    # build hexagon polygons
    polys, vals = [], []
    for cell, v in rich.items():
        lons, lats = cell_to_poly_xy(cell)
        polys.append(list(zip(lons, lats)))
        vals.append(v)
    vals = np.array(vals)

    fig, ax = plt.subplots(figsize=(7, 7))
    # boundary outline
    if geom is not None:
        try:
            import geopandas as gpd
            gpd.GeoSeries([geom], crs=4326).boundary.plot(
                ax=ax, color="0.4", linewidth=0.6)
        except Exception:
            pass

    pc = PolyCollection(polys, array=vals, cmap=CMAP,
                        edgecolors="none", alpha=0.9)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    cb = fig.colorbar(pc, ax=ax, shrink=0.7)
    cb.set_label("Species richness per cell")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outpath), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["species", "decimalLatitude", "decimalLongitude"])
    df["cell"] = [h3.latlng_to_cell(lat, lon, H3_RES)
                  for lat, lon in zip(df["decimalLatitude"], df["decimalLongitude"])]
    if USE_MAJOR_HABITAT and HABITAT_COL in df.columns:
        df["habitat_major"] = df[HABITAT_COL].apply(major_habitat)
        hab_col = "habitat_major"
    else:
        hab_col = HABITAT_COL
    print(f"  {len(df):,} records | {df['species'].nunique():,} species "
          f"| {df['cell'].nunique():,} occupied cells (res {H3_RES})")

    print("Building Tanzania cell denominator...")
    all_cells, geom = tanzania_cells(H3_RES)

    # ---- occupancy tables
    print("Occupancy by growth form...")
    occ_growth = occupancy_table(df, all_cells, GROWTH_COL)
    occ_growth.to_csv(os.path.join(OUTDIR, "occupancy_by_growth.csv"), index=False)
    print(occ_growth.to_string(index=False))

    print("\nOccupancy by habitat...")
    occ_hab = occupancy_table(df, all_cells, hab_col)
    occ_hab.to_csv(os.path.join(OUTDIR, "occupancy_by_habitat.csv"), index=False)
    print(occ_hab.to_string(index=False))

    # ---- maps
    print("\nRichness map: overall...")
    richness_map(df, all_cells, geom,
                 f"Species richness — Tanzania (res {H3_RES})",
                 "richness_ALL.png")

    print("Richness maps: per growth form...")
    for g in df[GROWTH_COL].dropna().unique():
        richness_map(df, all_cells, geom,
                     f"Species richness — {g} (res {H3_RES})",
                     f"richness_growth_{safe_filename(g)}.png",
                     clip_to=(GROWTH_COL, g))

    print("Richness maps: per habitat...")
    for g in df[hab_col].dropna().unique():
        richness_map(df, all_cells, geom,
                     f"Species richness — {g} (res {H3_RES})",
                     f"richness_habitat_{safe_filename(g)}.png",
                     clip_to=(hab_col, g))

    print(f"\nDone. Outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
