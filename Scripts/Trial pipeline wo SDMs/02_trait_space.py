"""
Stage 2: Build the species × species functional trait distance matrix.

Inputs
------
- species_pool.csv (from Stage 1): one row per pool species with mean
  LHS trait values (height, SLA, seed_mass) and a record count.

Outputs (to OUTPUT_DIR)
-----------------------
- trait_distance.npz  : compressed numpy archive with two arrays:
                          'distance' (N x N float64 Gower distance matrix)
                          'species'  (N-length array of species names,
                                     in the same order as matrix rows/cols)
- trait_diagnostics.csv : per-trait summary (n, min, max, mean, sd)
                          on log10-transformed values, for sanity checking.

Design notes
------------
- Traits are log10-transformed before distance calculation. LHS traits are
  all approximately log-normal across plants; raw values give a few giant
  trees and heavy seeds undue leverage. Log-transform is standard in
  community trait analysis (Bruelheide et al. 2018).
- Gower distance is used because it normalises each trait to [0, 1] before
  averaging across traits, so traits with different units / variances
  contribute equally. With three continuous traits and no missingness
  (we've already filtered to complete LHS), Gower reduces to mean absolute
  difference on min–max scaled values.
- This module produces the trait space ONCE for the whole pool. Stages 3
  and 5 slice it per cell — they never recompute distances.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TRAIT_COLUMNS, VERBOSE


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[02_trait_space] {msg}")


def gower_distance_continuous(X: np.ndarray) -> np.ndarray:
    """Gower distance for an N x P matrix of continuous traits.

    Each column is min–max scaled to [0, 1]; pairwise distance is then
    the mean absolute difference across columns. Returns an N x N matrix.
    """
    X = X.astype(np.float64)
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    col_range = col_max - col_min
    # Guard against zero-range columns (shouldn't happen with real LHS data,
    # but a single-value trait would otherwise produce NaN).
    col_range[col_range == 0] = 1.0
    Xs = (X - col_min) / col_range

    # Pairwise mean absolute difference across traits.
    # |x_i - x_j| averaged over P, computed via broadcasting.
    diff = np.abs(Xs[:, None, :] - Xs[None, :, :])  # N x N x P
    D = diff.mean(axis=2)
    # Force exact symmetry and zero diagonal (numerical hygiene).
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def main() -> None:
    pool_path = OUTPUT_DIR / "species_pool.csv"
    if not pool_path.exists():
        sys.exit(f"Missing Stage 1 output: {pool_path}. Run 01_assemblages.py first.")

    log(f"Reading {pool_path.name}")
    pool = pd.read_csv(pool_path)
    log(f"  {len(pool):,} species in pool")

    missing_cols = [c for c in TRAIT_COLUMNS if c not in pool.columns]
    if missing_cols:
        sys.exit(f"species_pool.csv missing trait columns: {missing_cols}")

    # Drop any species missing a trait (defensive; Stage 1 should have done this).
    pool = pool.dropna(subset=TRAIT_COLUMNS).reset_index(drop=True)
    log(f"  {len(pool):,} species with complete traits")

    # Defensive against non-positive values before log transform.
    bad = (pool[TRAIT_COLUMNS] <= 0).any(axis=1)
    if bad.any():
        log(f"  Warning: {bad.sum()} species had non-positive trait values; dropping.")
        pool = pool[~bad].reset_index(drop=True)

    # --- Log transform ----------------------------------------------------
    log("  Log10-transforming traits")
    X = np.log10(pool[TRAIT_COLUMNS].to_numpy(dtype=np.float64))

    # Diagnostics on log-transformed values
    diag = pd.DataFrame({
        "trait": TRAIT_COLUMNS,
        "n":     [X.shape[0]] * len(TRAIT_COLUMNS),
        "min":   X.min(axis=0),
        "max":   X.max(axis=0),
        "mean":  X.mean(axis=0),
        "sd":    X.std(axis=0, ddof=1),
    })
    diag.to_csv(OUTPUT_DIR / "trait_diagnostics.csv", index=False)
    log("  Log-trait summary:")
    for _, r in diag.iterrows():
        log(f"    {r['trait']:>10}: mean={r['mean']:.2f}  "
            f"sd={r['sd']:.2f}  range=[{r['min']:.2f}, {r['max']:.2f}]")

    # --- Gower distance ---------------------------------------------------
    log(f"  Computing {len(pool)} x {len(pool)} Gower distance matrix")
    D = gower_distance_continuous(X)
    log(f"    distance range: [{D.min():.4f}, {D.max():.4f}]")
    log(f"    mean off-diagonal distance: {D[np.triu_indices_from(D, k=1)].mean():.4f}")

    # --- Save -------------------------------------------------------------
    out_path = OUTPUT_DIR / "trait_distance.npz"
    np.savez_compressed(
        out_path,
        distance=D,
        species=pool["species"].to_numpy(),
    )
    log(f"  Wrote {out_path.name} "
        f"({out_path.stat().st_size / 1024:.1f} KB on disk)")
    log("Stage 2 complete.")


if __name__ == "__main__":
    main()