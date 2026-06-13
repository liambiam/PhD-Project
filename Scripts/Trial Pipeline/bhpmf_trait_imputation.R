################################################################################
# BHPMF trait gap-filling for RAINBIO / TRY Tanzania plant traits
# ==============================================================================
# Fills missing height / SLA / seed_mass values using Bayesian Hierarchical
# Probabilistic Matrix Factorisation (Schrodt et al. 2015, Glob. Ecol. Biogeogr.),
# borrowing strength across the species -> genus -> family taxonomic hierarchy.
#
# Includes a hold-out VALIDATION: known values from the complete-trait species
# are artificially masked, imputed, and compared to truth (RMSE / R2 per trait).
#
# Trait coverage in your data (for reference):
#   224 species with all 3 traits   <- validation pool (we know the truth here)
#   914 species with 2 traits
#  2514 species with 1 trait
# BHPMF imputes the whole matrix at once, so the 914 + 2514 get filled too.
#
#-------------------------------------------------------------------------------
# !! SETUP -- READ BEFORE RUNNING !!
#-------------------------------------------------------------------------------
# BHPMF was archived from CRAN (2017) and needs the C compiler from an older R.
# The maintained guidance is to use R 3.4.4. Recommended install route:
#
#   # 1. Install dependencies first
#   install.packages(c("Matrix", "matrixStats", "Rcpp", "RcppArmadillo"))
#
#   # 2a. From the GitHub mirror (preferred):
#   install.packages("devtools")
#   devtools::install_github("fisw10/BHPMF")
#
#   # 2b. OR from the CRAN archive tarball if GitHub fails:
#   url <- "https://cran.r-project.org/src/contrib/Archive/BHPMF/BHPMF_1.0.tar.gz"
#   install.packages(url, repos = NULL, type = "source")
#
# Windows: you need Rtools matching your R version on PATH so the C++ compiles.
# Mac: install Xcode + command line tools (xcode-select --install).
#
# BHPMF writes intermediate + output files to disk (tmp.dir / out files), so it
# needs a WRITABLE working directory. Set TMP_DIR and OUT_DIR below.
################################################################################

## ---- 0. CONFIG --------------------------------------------------------------
DATA_PATH <- "C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/rainbio_LHS_traits.csv"
TMP_DIR   <- "C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/bhpmf_tmp"     # scratch
OUT_DIR   <- "C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/bhpmf_out"     # results
SET_SEED  <- 42

TRAIT_COLS <- c("height", "SLA", "seed_mass")   # the LHS triplet to impute
SPECIES_COL <- "species"
FAMILY_COL  <- "family"                          # used for the hierarchy

# BHPMF tuning (defaults are reasonable; raise iterations for final runs)
N_GIBBS_BURN  <- 500     # burn-in samples
N_GIBBS_USED  <- 1000    # retained samples
N_LATENT      <- 10      # latent dimensions of the factorisation
N_FOLDS_VALID <- 5       # cross-validation folds for the hold-out test
HOLDOUT_FRAC  <- 0.2     # fraction of known values masked per fold (validation)

suppressPackageStartupMessages({
  library(BHPMF)
})

dir.create(TMP_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
set.seed(SET_SEED)

## ---- 1. LOAD & PREPARE ------------------------------------------------------
cat("[load] reading", DATA_PATH, "\n")
raw <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

# traits are denormalised per occurrence row -> collapse to ONE row per species.
# (within a species the trait values are constant; we take the first non-NA.)
first_nonNA <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[!is.na(x)]
  if (length(x) == 0) NA_real_ else x[1]
}

sp_tab <- aggregate(
  raw[, TRAIT_COLS],
  by = list(species = raw[[SPECIES_COL]]),
  FUN = first_nonNA
)
# attach family (first non-empty per species) for the hierarchy
fam_lookup <- aggregate(
  raw[[FAMILY_COL]],
  by = list(species = raw[[SPECIES_COL]]),
  FUN = function(x) { x <- x[x != "" & !is.na(x)]; if (length(x)) x[1] else NA }
)
names(fam_lookup)[2] <- "family"
sp_tab <- merge(sp_tab, fam_lookup, by = "species", all.x = TRUE)

# derive genus from the binomial (first token of the species name)
sp_tab$genus <- sub("\\s.*$", "", sp_tab$species)

# coverage report (should echo your 224 / 914 / 2514 split)
n_present <- rowSums(!is.na(sp_tab[, TRAIT_COLS]))
cat(sprintf("[load] %d species total | 3 traits: %d | 2 traits: %d | 1 trait: %d | 0: %d\n",
            nrow(sp_tab), sum(n_present == 3), sum(n_present == 2),
            sum(n_present == 1), sum(n_present == 0)))

## ---- 2. BUILD BHPMF INPUTS --------------------------------------------------
# BHPMF wants:
#   X         : numeric trait matrix (rows = observations, cols = traits), NAs ok
#   hierarchy : matrix of IDs, one row per observation, columns = levels
#               here: species_id, genus, family  (finest -> coarsest)
# IMPORTANT: traits are right-skewed (lognormal) -> log-transform before BHPMF,
# back-transform after. BHPMF assumes roughly Gaussian latent structure.

build_inputs <- function(tab) {
  X <- as.matrix(tab[, TRAIT_COLS])
  X <- log(X)                              # log-transform (all traits > 0)
  hierarchy <- data.frame(
    plant_id = seq_len(nrow(tab)),         # unique row id (level 1, finest)
    species  = tab$species,
    genus     = tab$genus,
    family    = tab$family,
    stringsAsFactors = FALSE
  )
  # BHPMF needs complete hierarchy (no NA in taxonomy); drop rows missing family
  ok <- !is.na(hierarchy$family) & !is.na(hierarchy$genus)
  list(X = X[ok, , drop = FALSE],
       hierarchy = hierarchy[ok, , drop = FALSE],
       kept = ok)
}

inp <- build_inputs(sp_tab)
cat(sprintf("[prep] %d species with complete taxonomy go into BHPMF\n",
            nrow(inp$X)))

## ---- 3. VALIDATION: hold-out known values, impute, score --------------------
# Strategy: among cells we actually KNOW (non-NA), randomly mask a fraction,
# run BHPMF, and compare imputed vs true on the log scale (then back-transform
# for an interpretable RMSE). Repeat over folds. This estimates per-trait
# accuracy and tells you which imputed traits to trust downstream.
#
# CAVEAT to state in writing: the cells we can validate on belong to the
# better-studied species. Real missingness is biased toward rarer species, so
# these numbers likely OVERESTIMATE accuracy for the hardest cases.

run_bhpmf <- function(X, hierarchy, tmp, out) {
  # wrapper around BHPMF::GapFilling; returns the mean-imputed matrix (log scale)
  unlink(list.files(tmp, full.names = TRUE))      # clean scratch between runs
  mean_file <- file.path(out, "mean_gap_filled.txt")
  std_file  <- file.path(out, "std_gap_filled.txt")
  GapFilling(
    X = X,
    hierarchy.info = hierarchy,
    prediction.level = 4,                 # predict at finest (species) level
    used.num.hierarchy.levels = 3,        # use species/genus/family
    mean.gap.filled.output.path = mean_file,
    std.gap.filled.output.path  = std_file,
    rmse.plot.test.data = FALSE,
    number.latent.features = N_LATENT,
    burn = N_GIBBS_BURN, gaps = 2, num.samples = N_GIBBS_USED,
    tmp.dir = tmp, verbose = FALSE
  )
  as.matrix(read.table(mean_file, header = FALSE))
}

validate <- function(inp, n_folds = N_FOLDS_VALID, frac = HOLDOUT_FRAC) {
  X <- inp$X; H <- inp$hierarchy
  known_idx <- which(!is.na(X), arr.ind = TRUE)   # all observed cells
  results <- vector("list", n_folds)

  for (f in seq_len(n_folds)) {
    cat(sprintf("[valid] fold %d/%d\n", f, n_folds))
    n_mask <- floor(frac * nrow(known_idx))
    mask <- known_idx[sample(nrow(known_idx), n_mask), , drop = FALSE]

    Xtest <- X
    truth <- numeric(nrow(mask))
    for (i in seq_len(nrow(mask))) {
      truth[i] <- X[mask[i, 1], mask[i, 2]]
      Xtest[mask[i, 1], mask[i, 2]] <- NA       # hide it
    }

    filled <- run_bhpmf(Xtest, H, TMP_DIR, OUT_DIR)
    pred <- numeric(nrow(mask))
    for (i in seq_len(nrow(mask))) pred[i] <- filled[mask[i, 1], mask[i, 2]]

    fold_df <- data.frame(
      fold = f,
      trait = TRAIT_COLS[mask[, 2]],
      true_log = truth, pred_log = pred,
      true = exp(truth), pred = exp(pred)        # back-transformed
    )
    results[[f]] <- fold_df
  }
  do.call(rbind, results)
}

cat("[valid] starting cross-validated hold-out test (this is the slow part)\n")
val <- validate(inp)

# per-trait accuracy on the back-transformed scale
score <- function(df) {
  by_trait <- split(df, df$trait)
  out <- lapply(names(by_trait), function(tr) {
    d <- by_trait[[tr]]
    rmse_log <- sqrt(mean((d$pred_log - d$true_log)^2))
    r2_log   <- 1 - sum((d$pred_log - d$true_log)^2) /
                    sum((d$true_log - mean(d$true_log))^2)
    data.frame(trait = tr, n = nrow(d),
               RMSE_log = round(rmse_log, 3),
               R2_log = round(r2_log, 3))
  })
  do.call(rbind, out)
}
val_summary <- score(val)
cat("\n[valid] ===== PER-TRAIT IMPUTATION ACCURACY =====\n")
print(val_summary)
write.csv(val_summary, file.path(OUT_DIR, "validation_summary.csv"), row.names = FALSE)
write.csv(val, file.path(OUT_DIR, "validation_predictions.csv"), row.names = FALSE)

## ---- 4. FINAL RUN: impute the full matrix -----------------------------------
cat("\n[final] imputing full trait matrix (all species at once)\n")
filled_log <- run_bhpmf(inp$X, inp$hierarchy, TMP_DIR, OUT_DIR)

# also pull the per-cell standard deviation (uncertainty) BHPMF wrote out
std_log <- as.matrix(read.table(file.path(OUT_DIR, "std_gap_filled.txt"),
                                header = FALSE))

# back-transform to natural units and tag which cells were imputed vs observed
imputed_mask <- is.na(inp$X)
out_tab <- inp$hierarchy[, c("species", "genus", "family")]
for (j in seq_along(TRAIT_COLS)) {
  tr <- TRAIT_COLS[j]
  out_tab[[tr]]            <- exp(filled_log[, j])    # value (observed or imputed)
  out_tab[[paste0(tr, "_sd")]]      <- std_log[, j]   # uncertainty (log scale)
  out_tab[[paste0(tr, "_imputed")]] <- imputed_mask[, j]  # TRUE if filled
}

out_path <- file.path(OUT_DIR, "traits_gapfilled.csv")
write.csv(out_tab, out_path, row.names = FALSE)
cat(sprintf("[final] wrote %s  (%d species, %d trait-values imputed)\n",
            out_path, nrow(out_tab), sum(imputed_mask)))

## ---- 5. WHAT YOU GET --------------------------------------------------------
# OUT_DIR/validation_summary.csv     -> RMSE & R2 per trait (trust assessment)
# OUT_DIR/validation_predictions.csv -> every held-out true-vs-pred pair
# OUT_DIR/traits_gapfilled.csv       -> full imputed table + per-cell SD +
#                                       a *_imputed flag so you can, e.g., keep
#                                       only imputations with SD below a chosen
#                                       threshold when promoting species to the
#                                       "complete-trait" pool for the pipeline.
#
# NEXT: filter traits_gapfilled.csv to species whose imputations are reliable
# (low SD / traits that validated well) and feed that enlarged pool into the
# functional-diversity stages of pilot_pipeline.py.
cat("\n[done] BHPMF gap-filling complete.\n")
