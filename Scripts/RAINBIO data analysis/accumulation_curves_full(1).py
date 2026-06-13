"""
Tanzania RAINBIO occurrence data — accumulation curves & species-area relationship.

Sampling unit = H3 hex cell (point data, not plots).

Outputs
-------
1. Sample-based rarefaction (accumulation) curves:
     - by growth habit (a_habit)          -> one figure, all classes overlaid
     - by habitat (habitat_name)          -> one figure, all habitats overlaid
     - total (all species)                 -> one figure
     - separate single-group figures for every class and every habitat
2. Chao2 asymptotic richness estimate per group (PRINTED TABLE ONLY, not on plots).
3. Species-area relationship (log-log) using nested H3 resolutions -> figure + z exponent.
4. Trees-only, split by habitat (Neil's suggestion).

Expects columns:
  species, decimalLatitude, decimalLongitude, family, a_habit, habitat_name

Rarefaction note
----------------
Accumulation curves are sample-based rarefaction computed here by permuting cell
order N_PERMS times and averaging cumulative richness. The shaded band is +/-1 SD
ACROSS PERMUTATIONS (a spread, not a formal CI). For true CIs and extrapolation,
use iNEXT in R; the printed Chao2 value gives the asymptotic target.

CONFIG values you may want to change are flagged inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h3

# ----------------------------------------------------------------------
# CONFIG
CSV_PATH    = r"C:\Users\liams\Documents\PhD-Project Data\tanzania\tanzania_points_with_habitat_labels.csv"
OUTDIR      = r"C:\Users\liams\Documents\PhD-Project Data\Accumulation Curves\Broad habitats"
HABITAT_COL = "habitat_name.x"   # full habitat column in the CSV
USE_MAJOR_HABITAT = True       # True: collapse to major habitat (text before dash) for grouping
H3_RES      = 7        # H3 res (integer 0-15). 6 ~36 km2/cell. Higher = finer/noisier.
N_PERMS     = 100      # permutations for rarefaction averaging
MIN_RECORDS = 20       # skip groups with fewer records than this (avoids noisy stubs)
SAR_RES     = [3, 4, 5, 6, 7]   # nested H3 resolutions for the species-area relationship
RNG_SEED    = 42
# ----------------------------------------------------------------------

rng = np.random.default_rng(RNG_SEED)
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------
# Data loading / binning
# ----------------------------------------------------------------------
def major_habitat(name):
    """Collapse a full habitat name to its major class: the text before the
    first dash (en-dash or hyphen). 'Savanna – Dry' -> 'Savanna',
    'Forest - Subtropical/Tropical Moist Lowland' -> 'Forest'.
    Names without a dash are returned unchanged."""
    if pd.isna(name):
        return name
    s = str(name)
    # split on en-dash, em-dash, or hyphen-with-spaces (avoids splitting
    # hyphenated single words); fall back to plain hyphen if needed
    for sep in [" – ", " — ", " - ", "–", "—"]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()


def load(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["species", "decimalLatitude", "decimalLongitude"])
    if USE_MAJOR_HABITAT:
        df["habitat_major"] = df[HABITAT_COL].apply(major_habitat)
    return df


def add_cells(df, res):
    return [h3.latlng_to_cell(lat, lon, res)
            for lat, lon in zip(df["decimalLatitude"], df["decimalLongitude"])]


def safe_filename(value, maxlen=80):
    """Make a string safe for a Windows filename: strip characters Windows
    forbids (< > : " / \\ | ? *), collapse whitespace to underscores, replace
    other non-word punctuation, and truncate. Plot titles keep the original
    text; only the filename is sanitised."""
    import re
    s = str(value)
    s = re.sub(r'[<>:"/\\|?*]', "", s)   # forbidden on Windows
    s = re.sub(r"\s+", "_", s.strip())    # whitespace -> underscore
    s = re.sub(r"[^\w\-.]", "_", s)       # any remaining odd char (dashes, parens, >) -> _
    s = re.sub(r"_+", "_", s).strip("_")  # collapse repeats
    return s[:maxlen] or "group"


# ----------------------------------------------------------------------
# Rarefaction (sample-based, over cells)
# ----------------------------------------------------------------------
def rarefaction(sub, cell_col="cell", n_perms=N_PERMS):
    """Return (x, mean, sd) cumulative species over number of cells."""
    cell_species = sub.groupby(cell_col)["species"].apply(set)
    n_cells = len(cell_species)
    if n_cells == 0:
        return None
    sets = list(cell_species.values)
    curves = np.zeros((n_perms, n_cells), dtype=int)
    for p in range(n_perms):
        order = rng.permutation(n_cells)
        seen = set()
        for i, idx in enumerate(order):
            seen |= sets[idx]
            curves[p, i] = len(seen)
    x = np.arange(1, n_cells + 1)
    return x, curves.mean(axis=0), curves.std(axis=0)


# ----------------------------------------------------------------------
# Chao2 asymptotic richness estimator (incidence-based) — PRINTED ONLY
# ----------------------------------------------------------------------
def chao2(sub, cell_col="cell"):
    inc = sub.groupby("species")[cell_col].nunique()
    S_obs = int((inc > 0).sum())
    Q1 = (inc == 1).sum()
    Q2 = (inc == 2).sum()
    m = sub[cell_col].nunique()
    if m < 2:
        return S_obs, np.nan
    correction = ((m - 1) / m) * (Q1 * (Q1 - 1)) / (2 * (Q2 + 1))
    return S_obs, S_obs + correction


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot_overlay(df, group_col, title, outpath):
    """All groups in one figure, overlaid, sorted by richness.
    Legend shows observed richness only (Chao2 kept for the printed table)."""
    groups = list(df[group_col].dropna().unique())
    groups.sort(key=lambda g: df.loc[df[group_col] == g, "species"].nunique(),
                reverse=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("tab10")
    table = []
    ci = 0
    for g in groups:
        sub = df[df[group_col] == g]
        if len(sub) < MIN_RECORDS:
            continue
        res = rarefaction(sub)
        if res is None:
            continue
        x, mean, sd = res
        s_obs, s_chao = chao2(sub)
        color = cmap(ci % 10); ci += 1
        ax.plot(x, mean, color=color, lw=2, label=f"{g} ({s_obs} spp)")
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
        table.append((g, s_obs, round(s_chao, 1), len(sub)))

    ax.set_xlabel(f"Number of H3 cells sampled (res {H3_RES})")
    ax.set_ylabel("Cumulative species richness")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outpath), dpi=150)
    plt.close(fig)
    return table


def plot_single(df, group_col, group_val, outpath):
    """One group, its own figure (no Chao2 line). Title = group value."""
    sub = df[df[group_col] == group_val]
    if len(sub) < MIN_RECORDS:
        return
    res = rarefaction(sub)
    if res is None:
        return
    x, mean, sd = res
    s_obs, _ = chao2(sub)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, mean, color="#1f77b4", lw=2, label="Observed (rarefied)")
    ax.fill_between(x, mean - sd, mean + sd, color="#1f77b4", alpha=0.18)
    ax.set_xlabel(f"Number of H3 cells sampled (res {H3_RES})")
    ax.set_ylabel("Cumulative species richness")
    ax.set_title(f"{group_val}  (observed {s_obs} spp)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outpath), dpi=150)
    plt.close(fig)


def plot_total(df, outpath):
    """Total accumulation, all species (no Chao2 line)."""
    x, mean, sd = rarefaction(df)
    s_obs, s_chao = chao2(df)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, mean, color="black", lw=2, label="Observed (rarefied)")
    ax.fill_between(x, mean - sd, mean + sd, color="black", alpha=0.12)
    ax.set_xlabel(f"Number of H3 cells sampled (res {H3_RES})")
    ax.set_ylabel("Cumulative species richness")
    ax.set_title(f"Total species accumulation — Tanzania (observed {s_obs} spp)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outpath), dpi=150)
    plt.close(fig)
    return s_obs, s_chao


# ----------------------------------------------------------------------
# Species-area relationship (log-log), nested H3 resolutions
# ----------------------------------------------------------------------
def species_area(df, resolutions=SAR_RES, outpath="species_area_relationship.png"):
    areas, mean_S = [], []
    for r in resolutions:
        cells = add_cells(df, r)
        tmp = df.assign(_cell=cells)
        s_per_cell = tmp.groupby("_cell")["species"].nunique()
        mean_S.append(s_per_cell.mean())
        areas.append(h3.average_hexagon_area(r, unit="km^2"))

    areas = np.array(areas); mean_S = np.array(mean_S)
    logA, logS = np.log10(areas), np.log10(mean_S)
    z, c = np.polyfit(logA, logS, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(logA, logS, color="#2ca02c", zorder=3)
    xfit = np.linspace(logA.min(), logA.max(), 50)
    ax.plot(xfit, z * xfit + c, color="#2ca02c", lw=1.5, label=f"z = {z:.3f}")
    for r, xa, ys in zip(resolutions, logA, logS):
        ax.annotate(f"res {r}", (xa, ys), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("log10 cell area (km$^2$)")
    ax.set_ylabel("log10 mean species per cell")
    ax.set_title("Species-area relationship (nested H3 grains)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, outpath), dpi=150)
    plt.close(fig)
    return z


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading...")
    df = load(CSV_PATH)
    df["cell"] = add_cells(df, H3_RES)
    print(f"  {len(df):,} records | {df['species'].nunique():,} species "
          f"| {df['cell'].nunique():,} occupied cells (res {H3_RES})")

    # which column to group habitat curves on
    hab_col = "habitat_major" if USE_MAJOR_HABITAT else HABITAT_COL
    print(f"  grouping habitats on '{hab_col}' "
          f"({df[hab_col].nunique()} classes)")

    print("By growth habit (overlay)...")
    t_class = plot_overlay(df, "a_habit",
                           "Species accumulation by growth habit — Tanzania",
                           "accumulation_by_class.png")
    print("By habitat (overlay)...")
    t_hab = plot_overlay(df, hab_col,
                         "Species accumulation by habitat — Tanzania",
                         "accumulation_by_habitat.png")

    print("Total...")
    s_obs, s_chao = plot_total(df, "accumulation_total.png")

    print("Per-class single figures...")
    for g in df["a_habit"].dropna().unique():
        safe = safe_filename(g)
        plot_single(df, "a_habit", g, f"accum_class_{safe}.png")
    print("Per-habitat single figures...")
    for g in df[hab_col].dropna().unique():
        safe = safe_filename(g)
        plot_single(df, hab_col, g, f"accum_habitat_{safe}.png")

    print("Species-area relationship...")
    z = species_area(df)

    print("Trees-only, by habitat...")
    trees = df[df["a_habit"].astype(str).str.lower().str.contains("tree", na=False)]
    if len(trees):
        plot_overlay(trees, hab_col,
                     "Tree species accumulation by habitat — Tanzania",
                     "accumulation_trees_by_habitat.png")

    print("\n=== Chao2 summary (table only) ===")
    print(f"{'group':45s} {'obs':>6s} {'Chao2':>8s} {'records':>8s}")
    print(f"{'TOTAL':45s} {s_obs:>6d} {s_chao:>8.1f} {len(df):>8d}")
    for label, tab in [("CLASS", t_class), ("HABITAT", t_hab)]:
        print(f"-- by {label} --")
        for g, obs, ch, n in tab:
            print(f"{str(g):45s} {obs:>6d} {ch:>8.1f} {n:>8d}")
    print(f"\nSpecies-area z exponent (all species): {z:.3f}")
    print(f"\nFigures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
