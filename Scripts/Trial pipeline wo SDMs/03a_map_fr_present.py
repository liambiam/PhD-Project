"""
Diagnostic maps of present-day diversity metrics.

Plots a configurable set of per-cell metrics from fr_present.csv as filled
grid cells, with the Tanzania border overlaid.

Standalone — not part of the numbered pipeline. Run after Stage 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from config import OUTPUT_DIR, TANZANIA_SHP, GRID_RES_DEG, MIN_SPECIES_PER_CELL

# ---------------------------------------------------------------------------
# Which metrics to map. (column, colourmap, label, clip_percentiles)
# Drop or reorder freely. SD omitted by default (redundant with n_species
# under equal weights); add it back if you want completeness.
# ---------------------------------------------------------------------------
METRICS = [
    ("FR",          "viridis", "Functional redundancy (FR)",      (0.02, 0.98)),
    ("FD",          "plasma",  "Functional diversity (Rao's Q)",  (0.02, 0.98)),
   # ("U",           "cividis", "Functional uniqueness (U)",       (0.02, 0.98)),
    ("FD_FR_ratio", "magma",   "FD / FR ratio",                   (0.02, 0.98)),
    ("n_species",   "magma",   "Species per cell", (0.0, 0.95)),
]


def load_border():
    """Return the TZ border GeoDataFrame, or None, printing why if it fails."""
    try:
        import geopandas as gpd
    except ImportError:
        print("Border skipped: geopandas not installed "
              "(pip install geopandas).")
        return None
    try:
        border = gpd.read_file("C:\\Users\\liams\\Documents\\PhD-Project Data\\GADM TZ Shape\\gadm41_TZA_0.shp")
        print(f"Border loaded: {TANZANIA_SHP}")
        return border
    except Exception as e:
        print(f"Border skipped: could not read {TANZANIA_SHP}\n  {e}")
        return None


def draw_border(ax, border):
    if border is not None:
        border.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=3)


def draw_cells(ax, df, value_col, cmap, clip):
    """Draw grid cells as filled squares sized to GRID_RES_DEG."""
    vals = df[value_col].to_numpy()
    lo = np.nanquantile(vals, clip[0])
    hi = np.nanquantile(vals, clip[1])
    norm = Normalize(vmin=lo, vmax=hi)
    cmap_obj = plt.get_cmap(cmap)

    half = GRID_RES_DEG / 2.0
    patches = []
    colors = []
    for lon, lat, v in zip(df["lon_centre"], df["lat_centre"], vals):
        patches.append(Rectangle((lon - half, lat - half),
                                 GRID_RES_DEG, GRID_RES_DEG))
        colors.append(cmap_obj(norm(v)))
    pc = PatchCollection(patches, facecolors=colors,
                         edgecolors="white", linewidths=0.2, zorder=2)
    ax.add_collection(pc)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    return sm


def main():
    fr = pd.read_csv(OUTPUT_DIR / "fr_present.csv")
    valid = fr.dropna(subset=["FR"]).copy()
    print(f"{len(valid)} cells with valid metrics")

    border = load_border()

    n = len(METRICS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # Shared extent so all panels line up
    pad = GRID_RES_DEG
    xlim = (valid["lon_centre"].min() - pad, valid["lon_centre"].max() + pad)
    ylim = (valid["lat_centre"].min() - pad, valid["lat_centre"].max() + pad)

    for ax, (col, cmap, label, clip) in zip(axes, METRICS):
        if col not in valid.columns:
            ax.set_visible(False)
            print(f"Skipping {col}: not in fr_present.csv")
            continue
        draw_border(ax, border)
        sm = draw_cells(ax, valid, col, cmap, clip)
        plt.colorbar(sm, ax=ax, label=label, shrink=0.8)
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")

    # Hide any unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Present-day diversity metrics  "
        f"({len(valid)} cells, >= {MIN_SPECIES_PER_CELL} species, "
        f"{GRID_RES_DEG} deg grid)",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "map_metrics_present.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.show()


if __name__ == "__main__":
    main()