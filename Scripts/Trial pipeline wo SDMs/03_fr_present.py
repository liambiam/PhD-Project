"""
Stage 3: Functional redundancy of present-day communities.

For each occupied grid cell, computes the four diversity indices following
Ricotta et al. (2016):

  SD  = Simpson's diversity         (on species weights)
  FD  = Rao's quadratic entropy     (weights + trait dissimilarity)
  U   = FD / SD                     (functional uniqueness)
  FR  = 1 - U                       (functional redundancy)

Plus the FD/FR ratio retained for interpretation (functional breadth
relative to functional overlap, as in Martin-Fores et al. 2026).

Inputs
------
- assemblage_present.csv (Stage 1): cell_id, species, weight
- trait_distance.npz     (Stage 2): distance matrix + species index
- grid_cells.csv         (Stage 1): cell_id, lon_centre, lat_centre, climate

Outputs (to OUTPUT_DIR)
-----------------------
- fr_present.csv : per-cell SD, FD, U, FR, FD_FR_ratio, n_species

Design notes
------------
- Weights here are all 1.0 (presence-only trial). Simpson's and Rao's both
  use *relative* abundances, so the code normalises weights to sum to 1
  within each cell. With equal weights this reduces to the unweighted form,
  but the code path is identical for when SDM suitability weights arrive.
- Cells with fewer than MIN_SPECIES_PER_CELL species get NaN for all
  indices: Rao's Q is too unstable to interpret with 2-3 species.
- A species present in a cell but absent from the trait matrix (shouldn't
  happen, since the assemblage was filtered to trait-complete species) is
  dropped with a warning rather than crashing.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, MIN_SPECIES_PER_CELL, VERBOSE


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[03_fr_present] {msg}")


def simpson_diversity(p: np.ndarray) -> float:
    """Simpson's diversity index (Gini-Simpson): 1 - sum(p_i^2).

    p is a vector of relative abundances summing to 1.
    """
    return 1.0 - np.sum(p ** 2)


def rao_quadratic_entropy(p: np.ndarray, D: np.ndarray) -> float:
    """Rao's quadratic entropy: sum_i sum_j p_i p_j d_ij.

    p is relative abundances (length S); D is the S x S trait distance
    submatrix for the species present, aligned to p's order.
    """
    return float(p @ D @ p)


def compute_cell_indices(
    species_list: list[str],
    weights: np.ndarray,
    species_to_idx: dict[str, int],
    D_full: np.ndarray,
) -> dict[str, float] | None:
    """Compute SD, FD, U, FR, FD/FR for one cell. Returns None if too few species."""
    # Map species to matrix indices; drop any not found (defensive).
    idx = []
    keep = []
    for i, sp in enumerate(species_list):
        j = species_to_idx.get(sp)
        if j is not None:
            idx.append(j)
            keep.append(i)
    if len(idx) < MIN_SPECIES_PER_CELL:
        return None

    w = weights[keep].astype(np.float64)
    p = w / w.sum()  # relative abundances

    D_sub = D_full[np.ix_(idx, idx)]

    SD = simpson_diversity(p)
    FD = rao_quadratic_entropy(p, D_sub)

    # Functional uniqueness U = FD / SD. Guard against SD == 0
    # (only possible if a single species, already excluded by the floor).
    U = FD / SD if SD > 0 else np.nan
    FR = 1.0 - U if np.isfinite(U) else np.nan

    # FD / FR ratio. Guard against FR == 0 (all species functionally identical).
    FD_FR = FD / FR if (np.isfinite(FR) and FR > 0) else np.nan

    return {
        "n_species": len(idx),
        "SD": SD,
        "FD": FD,
        "U": U,
        "FR": FR,
        "FD_FR_ratio": FD_FR,
    }


def main() -> None:
    asm_path = OUTPUT_DIR / "assemblage_present.csv"
    dist_path = OUTPUT_DIR / "trait_distance.npz"
    grid_path = OUTPUT_DIR / "grid_cells.csv"

    for p in (asm_path, dist_path, grid_path):
        if not p.exists():
            sys.exit(f"Missing input: {p}. Run earlier stages first.")

    log(f"Reading {asm_path.name}")
    asm = pd.read_csv(asm_path)

    log(f"Reading {dist_path.name}")
    npz = np.load(dist_path, allow_pickle=True)
    D_full = npz["distance"]
    species_arr = npz["species"]
    species_to_idx = {sp: i for i, sp in enumerate(species_arr)}
    log(f"  Trait matrix: {D_full.shape[0]} species")

    log(f"Reading {grid_path.name}")
    grid = pd.read_csv(grid_path)

    # --- Per-cell computation ---------------------------------------------
    results = []
    n_too_few = 0
    n_missing_species = 0

    for cell_id, sub in asm.groupby("cell_id"):
        species_list = sub["species"].tolist()
        weights = sub["weight"].to_numpy()

        # Track species not in trait matrix
        n_missing_species += sum(
            1 for sp in species_list if sp not in species_to_idx
        )

        out = compute_cell_indices(species_list, weights, species_to_idx, D_full)
        if out is None:
            n_too_few += 1
            continue
        out["cell_id"] = cell_id
        results.append(out)

    if n_missing_species:
        log(f"  Warning: {n_missing_species} species occurrences not in trait "
            f"matrix (dropped per cell)")

    res_df = pd.DataFrame(results)
    log(f"  {len(res_df):,} cells with valid FR "
        f"(>= {MIN_SPECIES_PER_CELL} species)")
    log(f"  {n_too_few:,} occupied cells below species floor (NaN)")

    # --- Join coordinates + climate for mapping ---------------------------
    out = grid.merge(res_df, on="cell_id", how="left")
    out.to_csv(OUTPUT_DIR / "fr_present.csv", index=False)

    # --- Summary diagnostics ----------------------------------------------
    valid = res_df.dropna(subset=["FR"])
    if len(valid):
        log("  Present-day index summary (valid cells):")
        for col in ["n_species", "SD", "FD", "U", "FR", "FD_FR_ratio"]:
            v = valid[col]
            log(f"    {col:>12}: mean={v.mean():.3f}  "
                f"range=[{v.min():.3f}, {v.max():.3f}]")

    log(f"  Wrote fr_present.csv ({len(out):,} cells)")
    log("Stage 3 complete.")


if __name__ == "__main__":
    main()