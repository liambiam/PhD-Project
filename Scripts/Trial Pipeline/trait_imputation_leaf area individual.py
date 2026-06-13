"""
Trait gap-filling for RAINBIO / TRY Tanzania plant traits  (Python route)
================================================================================
Fills missing height / SLA / seed_mass using iterative Random-Forest imputation
(a missForest-style approach), borrowing strength from (a) correlations among
the traits and (b) the taxonomic hierarchy (genus, family) supplied as
predictors. Includes a hold-out VALIDATION that masks known values, imputes
them, and scores per-trait recovery (RMSE / R2 on the log scale).

This is the pragmatic stand-in for BHPMF (which needs an awkward R/C toolchain).
It is NOT identical: BHPMF uses the taxonomy as a Bayesian hierarchical prior
and returns per-value posterior uncertainty. Here taxonomy enters as engineered
predictors, and "uncertainty" is approximated by the spread across imputation
repeats. Keep BHPMF as the field-standard comparison once R is sorted -- the
validation harness below is method-agnostic and will score BHPMF the same way.

Trait coverage expected (your numbers): 224 species with 3 traits, 914 with 2,
2514 with 1. The 914 + 2514 all get imputed in the final run.

USAGE: set USE_SYNTHETIC=False and DATA_PATH, then `python trait_imputation.py`.
================================================================================
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------------- 
# CONFIG
# ----------------------------------------------------------------------------- 
USE_SYNTHETIC = False  # using the real rainbio_with_try_traits.csv
DATA_PATH = r"C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/rainbio_with_try_traits.csv"
OUTDIR = r"C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/outputs_imputation"
SEED = 42

SPECIES_COL = "species"
FAMILY_COL = "family"
TRAIT_COLS = ["height", "SLA", "seed_mass"]   # LHS triplet to impute

N_VALID_FOLDS = 5        # cross-validation folds for the hold-out test
HOLDOUT_FRAC = 0.2       # fraction of known values masked per fold
N_IMPUTE_REPEATS = 10    # repeats for the final run -> uncertainty estimate
RF_TREES = 200
# RUNTIME NOTE: on your full ~3650 species this is heavy. RF iterative
# imputation scales with species x folds x repeats x trees. If a full run is
# slow, first do a quick pass with N_VALID_FOLDS=3, N_IMPUTE_REPEATS=3,
# RF_TREES=80 to get accuracy numbers, then raise them for the final figures.
# Switching the estimator to ExtraTreesRegressor or HistGradientBoosting is a
# fast drop-in if RandomForest is the bottleneck.

rng = np.random.default_rng(SEED)
os.makedirs(OUTDIR, exist_ok=True)


# -----------------------------------------------------------------------------
# DATA: load real CSV or synthetic stand-in (same schema)
# -----------------------------------------------------------------------------
def _make_synthetic(n_species=3650):
    """~3650 species with the 224/914/2514-style coverage gradient, plus
    genuine genus/family structure so taxonomic borrowing has signal to use."""
    fams = [f"Family{i}" for i in range(40)]
    rows = []
    for i in range(n_species):
        fam = fams[i % len(fams)]
        genus = f"{fam}_Genus{(i // 3) % 120}"
        sp = f"{genus} sp{i}"
        # family-level trait means -> species drawn around them (hierarchy signal)
        fam_seed = hash(fam) % 1000 / 1000.0
        h = float(np.exp(rng.normal(0.5 + 2 * fam_seed, 0.6)))
        sla = float(np.exp(rng.normal(2.0 + fam_seed, 0.4)))
        sm = float(np.exp(rng.normal(0.5 + 3 * fam_seed, 1.0)))
        # impose a coverage gradient like the real data
        r = rng.random()
        if r < 0.06:        pass                      # ~all 3 (the "224")
        elif r < 0.30:      sla = np.nan              # 2 traits
        else:               sla = np.nan; h = np.nan  # 1 trait
        rows.append(dict(species=sp, family=fam, height=h, SLA=sla, seed_mass=sm))
    return pd.DataFrame(rows)


def load_species_table():
    if USE_SYNTHETIC:
        print("[load] SYNTHETIC data (schema-matched).")
        sp = _make_synthetic()
    else:
        print(f"[load] reading {DATA_PATH}")
        raw = pd.read_csv(DATA_PATH)
        for c in TRAIT_COLS:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        # collapse occurrence rows -> one row per species (first non-NA)
        agg = {c: (lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
               for c in TRAIT_COLS}
        agg[FAMILY_COL] = (lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        sp = raw.groupby(SPECIES_COL).agg(agg).reset_index()

    # derive genus from binomial
    sp["genus"] = sp[SPECIES_COL].str.split().str[0]
    n_present = sp[TRAIT_COLS].notna().sum(axis=1)
    print(f"[load] {len(sp)} species | 3 traits: {(n_present==3).sum()} | "
          f"2: {(n_present==2).sum()} | 1: {(n_present==1).sum()} | "
          f"0: {(n_present==0).sum()}")

    # exclude species with NO observed traits: nothing to borrow from, so any
    # "imputation" for them is just the global/taxon mean -> not real data.
    keep = n_present > 0
    n_drop = int((~keep).sum())
    sp = sp[keep].reset_index(drop=True)
    print(f"[load] dropped {n_drop} species with zero observed traits; "
          f"{len(sp)} species go forward to imputation")
    return sp


# -----------------------------------------------------------------------------
# TAXONOMIC PREDICTORS (leakage-safe)
# -----------------------------------------------------------------------------
def _taxon_means(train_df, level):
    """Mean of each log-trait within each taxon (genus/family), from TRAIN only.
    Returns dict: {level_value: {trait: mean_log_value}} plus global fallback."""
    means = {}
    for tax, g in train_df.groupby(level):
        means[tax] = {t: g[f"log_{t}"].mean() for t in TRAIT_COLS}
    global_mean = {t: train_df[f"log_{t}"].mean() for t in TRAIT_COLS}
    return means, global_mean


def _add_taxon_features(df, gmeans, gglob, fmeans, fglob):
    """Attach genus- and family-level mean-trait predictors to each row."""
    out = df.copy()
    for t in TRAIT_COLS:
        out[f"genus_{t}"] = [gmeans.get(gn, gglob)[t] for gn in df["genus"]]
        out[f"family_{t}"] = [fmeans.get(fm, fglob)[t] for fm in df["family"]]
    return out


def _impute_once(train_df, full_df, seed):
    """Fit RF iterative imputer using taxonomic features built from train_df,
    return the imputed log-trait matrix for full_df."""
    gmeans, gglob = _taxon_means(train_df, "genus")
    fmeans, fglob = _taxon_means(train_df, "family")
    feat = _add_taxon_features(full_df, gmeans, gglob, fmeans, fglob)

    log_traits = [f"log_{t}" for t in TRAIT_COLS]
    tax_feats = [f"{lvl}_{t}" for lvl in ("genus", "family") for t in TRAIT_COLS]
    X = feat[log_traits + tax_feats].values

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=RF_TREES, max_depth=12,
                                        n_jobs=-1, random_state=seed),
        max_iter=10, random_state=seed, sample_posterior=False)
    Xi = imputer.fit_transform(X)
    return Xi[:, :len(TRAIT_COLS)]      # imputed log-traits (first 3 cols)


# -----------------------------------------------------------------------------
# VALIDATION: mask known values, impute, score per trait
# -----------------------------------------------------------------------------
def validate(sp):
    df = sp.copy()
    for t in TRAIT_COLS:
        df[f"log_{t}"] = np.log(df[t])

    # cells we actually know (can validate on)
    known = [(i, t) for i in df.index for t in TRAIT_COLS
             if np.isfinite(df.at[i, f"log_{t}"])]
    known = np.array(known, dtype=object)
    print(f"[valid] {len(known)} known trait-values available to test on")

    records = []
    for fold in range(N_VALID_FOLDS):
        n_mask = int(HOLDOUT_FRAC * len(known))
        pick = rng.choice(len(known), n_mask, replace=False)
        masked = df.copy()
        truth = []
        for k in pick:
            i, t = known[k]; i = int(i)
            truth.append((i, t, df.at[i, f"log_{t}"]))
            masked.at[i, f"log_{t}"] = np.nan

        # train set = rows still fully observed after masking are not required;
        # taxon means are built from all currently-observed cells in `masked`
        train = masked.dropna(subset=[f"log_{t}" for t in TRAIT_COLS], how="all")
        im=_impute_once(train, masked, seed=SEED + fold)
        col = {t: j for j, t in enumerate(TRAIT_COLS)}
        for (i, t, true_log) in truth:
            records.append(dict(fold=fold, trait=t, true_log=true_log,
                                pred_log=im[df.index.get_loc(i), col[t]]))
        print(f"[valid] fold {fold+1}/{N_VALID_FOLDS} done ({n_mask} masked)")

    res = pd.DataFrame(records)
    res["true"] = np.exp(res.true_log); res["pred"] = np.exp(res.pred_log)
    return res


def score(res):
    rows = []
    for t, d in res.groupby("trait"):
        rmse = np.sqrt(np.mean((d.pred_log - d.true_log) ** 2))
        ss_res = np.sum((d.pred_log - d.true_log) ** 2)
        ss_tot = np.sum((d.true_log - d.true_log.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(dict(trait=t, n=len(d), RMSE_log=round(rmse, 3),
                         R2_log=round(r2, 3)))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# FINAL RUN: impute full matrix + uncertainty from repeats
# -----------------------------------------------------------------------------
def final_impute(sp):
    df = sp.copy()
    for t in TRAIT_COLS:
        df[f"log_{t}"] = np.log(df[t])
    train = df.dropna(subset=[f"log_{t}" for t in TRAIT_COLS], how="all")

    stack = []
    for r in range(N_IMPUTE_REPEATS):
        stack.append(_impute_once(train, df, seed=SEED + 100 + r))
    arr = np.stack(stack)                    # (repeats, n_species, 3) log scale
    mean_log = arr.mean(axis=0)
    sd_log = arr.std(axis=0)                 # spread across repeats ~ uncertainty

    out = df[[SPECIES_COL, "genus", "family"]].copy()
    observed = df[TRAIT_COLS].notna().values
    for j, t in enumerate(TRAIT_COLS):
        out[t] = np.exp(mean_log[:, j])
        out[f"{t}_sd_log"] = sd_log[:, j]
        out[f"{t}_imputed"] = ~observed[:, j]
        # keep observed values exactly, don't overwrite with imputation
        out.loc[observed[:, j], t] = df.loc[observed[:, j], t].values
    return out


def main():
    sp = load_species_table()
    print("\n[valid] running hold-out validation ...")
    res = validate(sp)
    summary = score(res)
    print("\n===== PER-TRAIT IMPUTATION ACCURACY =====")
    print(summary.to_string(index=False))
    summary.to_csv(f"{OUTDIR}/validation_summary.csv", index=False)
    res.to_csv(f"{OUTDIR}/validation_predictions.csv", index=False)

    print("\n[final] imputing full trait matrix ...")
    filled = final_impute(sp)
    n_imp = filled[[f"{t}_imputed" for t in TRAIT_COLS]].sum().sum()
    n_complete = (~filled[[f"{t}_imputed" for t in TRAIT_COLS]].any(axis=1)).sum()
    print(f"[final] {len(filled)} species; {int(n_imp)} values imputed; "
          f"{n_complete} species were already complete")
    filled.to_csv(f"{OUTDIR}/traits_gapfilled.csv", index=False)
    print(f"[done] wrote {OUTDIR}/traits_gapfilled.csv")
    return summary, filled


if __name__ == "__main__":
    main()
