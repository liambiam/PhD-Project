###############################################################################
# Trial SDM comparison: WorldClim vs AlphaEarth embeddings
# Species: Vangueria infausta | Region: East Africa (display clipped to Tanzania)
# Design: 2x2  {WorldClim, AlphaEarth} x {RandomForest, MaxEnt}
#         + AlphaEarth-full-dim RF as a sensitivity check
# Fair-comparison spine: identical occurrences, background, folds, grid, metrics.
# The ONLY thing that changes between models is the predictor set (and learner).
#
# Author scaffold for Liam. Not run against your data here — set the CONFIG
# block, run the PREFLIGHT check, then execute top to bottom.
###############################################################################

## ---------------------------------------------------------------------------
## 0. PACKAGES  (install once; uncomment)
## ---------------------------------------------------------------------------
# install.packages(c("terra","sf","dplyr","randomForest","maxnet",
#                     "blockCV","ecospat","pROC","usdm"))

suppressPackageStartupMessages({
  library(terra); library(sf); library(dplyr)
  library(randomForest); library(maxnet); library(blockCV)
  library(ecospat); library(pROC)            # <- usdm removed
})

set.seed(42)



## ---------------------------------------------------------------------------
## 1. CONFIG  -- the only block you should normally edit
## ---------------------------------------------------------------------------
SPECIES        <- "Vangueria infausta"

# --- input paths (defaults follow your usual layout; adjust as needed) ------
OCC_CSV        <- "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/rainbio_published/published_database/RAINBIO.csv"
SP_COL         <- "tax_sp_level"        # column holding the species binomial
LON_COL        <- "decimalLongitude"    # longitude column
LAT_COL        <- "decimalLatitude"     # latitude column

WC_DIR     <- "C:/Users/liams/Documents/PhD-Project Data/worldclim/climate/wc2.1_30s/"
WC_PATTERN <- "wc2.1_30s_bio_%d.tif"

WC_BIOS        <- c(1, 4, 6, 12, 15, 17)          # pre-chosen, collinearity-checked below

# AlphaEarth: produced by aef_export.js (Earth Engine -> Drive -> local GeoTIFF).
# 64 bands, mean 2018-2022, exported at 1 km over the modelling extent.
AEF_TIF        <- "C:/Users/liams/Documents/PhD-Project Data/AlphaEarth/AEF_mean_2018_2022_EAfrica_1km.tif"

GADM_TZA       <- "C:/Users/liams/Documents/PhD-Project Data/GADM TZ Shape/gadm41_TZA_0.shp"
OUT_DIR        <- "C:/Users/liams/Documents/PhD-Project Data/SDM_trial_outputs/"

# --- analysis settings ------------------------------------------------------
GRID_RES_DEG   <- 1/30     # ~1 km. Set to 1/60 (~2 km) for a faster first run.
BUFFER_DEG     <- 2.5       # buffer around occurrences -> modelling extent
N_BACKGROUND   <- 10000     # background points, shared across ALL models
N_AEF_PCS      <- 8         # matched dimensionality for the primary AEF run
AEF_VAR_KEEP   <- 0.95      # variance retained for the AEF-full sensitivity run
K_FOLDS        <- 5         # spatial CV folds
BLOCK_SIZE_M   <- 200000    # spatial block size in metres (~200 km); tune this
THRESH_PCT     <- 0.10      # 10th-percentile training-presence threshold for binary maps

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ---------------------------------------------------------------------------
## 2. PREFLIGHT  -- fail early with a clear message if inputs are missing
## ---------------------------------------------------------------------------
preflight <- c(OCC_CSV = file.exists(OCC_CSV),
               WC_DIR  = dir.exists(WC_DIR),
               AEF_TIF = file.exists(AEF_TIF),
               GADM    = file.exists(GADM_TZA))
print(preflight)
if (any(!preflight))
  stop("Missing input(s): ", paste(names(preflight)[!preflight], collapse = ", "),
       ". Fix paths in CONFIG (AEF_TIF comes from running aef_export.js first).")

## ---------------------------------------------------------------------------
## 3. OCCURRENCES  -- load, filter to species, light cleaning, thin
## ---------------------------------------------------------------------------
occ_raw <- read.csv(OCC_CSV, stringsAsFactors = FALSE)
stopifnot(all(c(SP_COL, LON_COL, LAT_COL) %in% names(occ_raw)))

occ <- occ_raw %>%
  filter(.data[[SP_COL]] == SPECIES) %>%
  transmute(lon = as.numeric(.data[[LON_COL]]),
            lat = as.numeric(.data[[LAT_COL]])) %>%
  filter(!is.na(lon), !is.na(lat),
         abs(lon) <= 180, abs(lat) <= 90,
         !(lon == 0 & lat == 0)) %>%   # drop the classic 0,0 error
  distinct()

message(sprintf("Raw occurrences for %s: %d", SPECIES, nrow(occ)))
if (nrow(occ) < 30)
  warning("Fewer than 30 cleaned occurrences -- treat results as indicative only.")

## ---------------------------------------------------------------------------
## 4. MODELLING EXTENT + 1 km TEMPLATE  (drives everything downstream)
## ---------------------------------------------------------------------------
## 4. MODELLING EXTENT + 1 km TEMPLATE
occ_v <- terra::vect(occ, geom = c("lon","lat"), crs = "EPSG:4326")
e     <- terra::ext(occ_v)

# expand by BUFFER_DEG on all sides
ext_mod <- terra::ext(e[1] - BUFFER_DEG, e[2] + BUFFER_DEG,
                      e[3] - BUFFER_DEG, e[4] + BUFFER_DEG)

# cap to the Africa window (explicit min/max -- no intersect())
afr <- c(-20, 55, -36, 20)
ext_mod <- terra::ext(max(ext_mod[1], afr[1]), min(ext_mod[2], afr[2]),
                      max(ext_mod[3], afr[3]), min(ext_mod[4], afr[4]))

template <- terra::rast(ext_mod, resolution = GRID_RES_DEG, crs = "EPSG:4326")
message(sprintf("Modelling extent: lon [%.1f, %.1f], lat [%.1f, %.1f]; grid cells: %s",
                ext_mod[1], ext_mod[2], ext_mod[3], ext_mod[4],
                format(terra::ncell(template), big.mark = ",")))

## ---------------------------------------------------------------------------
## 5. WORLDCLIM  -- load chosen bios, crop, resample to template, check VIF
## ---------------------------------------------------------------------------
wc_files <- file.path(WC_DIR, sprintf(WC_PATTERN, WC_BIOS))
stopifnot(all(file.exists(wc_files)))
wc_raw   <- rast(wc_files)
names(wc_raw) <- paste0("bio", WC_BIOS)
wc       <- resample(crop(wc_raw, ext_mod, snap = "out"), template, method = "bilinear")

# Collinearity report (informational -- prune WC_BIOS if VIF is high, e.g. >10)
vif_tab <- tryCatch(usdm::vif(as.data.frame(wc, na.rm = TRUE)), error = function(e) NULL)
if (!is.null(vif_tab)) { message("WorldClim VIF:"); print(vif_tab) }

## ---------------------------------------------------------------------------
## 6. ALPHAEARTH  -- load 64 bands, resample, PCA -> top PCs (+ full-dim set)
## ---------------------------------------------------------------------------
aef_raw <- rast(AEF_TIF)
message(sprintf("AlphaEarth bands loaded: %d", nlyr(aef_raw)))
aef     <- resample(crop(aef_raw, ext_mod, snap = "out"), template, method = "bilinear")

# PCA fitted on a sample of grid cells (the prediction universe), then applied
# to the full grid so the PCs are defined consistently everywhere.
samp <- spatSample(aef, size = 50000, method = "random", na.rm = TRUE, values = TRUE)
pca         <- prcomp(samp, center = TRUE, scale. = TRUE)
var_cum     <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
n_full      <- which(var_cum >= AEF_VAR_KEEP)[1]
message(sprintf("AEF PCs: %d explain %.0f%% var; keeping %d (primary), %d (full)",
                N_AEF_PCS, 100 * var_cum[N_AEF_PCS], N_AEF_PCS, n_full))

aef_pcs     <- terra::predict(aef, pca)          # all PCs as a raster stack
names(aef_pcs) <- paste0("PC", seq_len(nlyr(aef_pcs)))
aef_pc8     <- aef_pcs[[1:N_AEF_PCS]]            # primary, dimension-matched
aef_full    <- aef_pcs[[1:n_full]]               # sensitivity

## ---------------------------------------------------------------------------
## 7. COMMON LAND MASK  -- fill coastal NA gaps then mask
## ---------------------------------------------------------------------------
wc_filled  <- focal(wc,      w = 3, fun = "mean", na.policy = "only", na.rm = TRUE)
aef_filled <- focal(aef_pc8, w = 3, fun = "mean", na.policy = "only", na.rm = TRUE)

mask <- !is.na(wc_filled[[1]]) & !is.na(aef_filled[[1]])
mask[mask == 0] <- NA

wc       <- terra::mask(wc_filled,  mask)
aef_pc8  <- terra::mask(aef_filled, mask)
aef_full <- terra::mask(aef_full,   mask)

# Which of the 153 thinned occurrences fall outside the mask?
mask_vals <- terra::extract(mask, occ_thin[, c("lon","lat")], ID = FALSE)[, 1]
message("Occurrences outside shared mask: ", sum(is.na(mask_vals) | mask_vals == 0))

# Where are they?
lost_occ <- occ_thin[is.na(mask_vals) | mask_vals == 0, ]
print(lost_occ)


## ---------------------------------------------------------------------------
## 8. THIN OCCURRENCES  -- one point per grid cell (done ONCE)
## ---------------------------------------------------------------------------
cells   <- cellFromXY(template, as.matrix(occ[, c("lon","lat")]))
occ_thin <- occ[!is.na(cells) & !duplicated(cells), ]
message(sprintf("Thinned occurrences (1 per cell): %d -> %d",
                nrow(occ), nrow(occ_thin)))

## ---------------------------------------------------------------------------
## 9. BACKGROUND  -- single shared set; extract BOTH predictor sets at it
## ---------------------------------------------------------------------------
bg <- spatSample(mask, size = N_BACKGROUND, method = "random",
                 na.rm = TRUE, xy = TRUE, values = TRUE)
bg <- as.data.frame(bg)[, c("x","y")]; names(bg) <- c("lon","lat")

pa_xy <- rbind(
  data.frame(occ = 1, occ_thin[, c("lon","lat")]),
  data.frame(occ = 0, bg)
)

ex_wc   <- terra::extract(wc,       pa_xy[, c("lon","lat")], ID = FALSE)
ex_pc8  <- terra::extract(aef_pc8,  pa_xy[, c("lon","lat")], ID = FALSE)
ex_full <- terra::extract(aef_full, pa_xy[, c("lon","lat")], ID = FALSE)

# Diagnose which predictor set is causing NA losses
na_wc   <- rowSums(is.na(ex_wc[pa_xy$occ == 1, ]))
na_pc8  <- rowSums(is.na(ex_pc8[pa_xy$occ == 1, ]))
na_full <- rowSums(is.na(ex_full[pa_xy$occ == 1, ]))

message("Presence NAs — WorldClim: ", sum(na_wc > 0),
        " | AEF-PC8: ", sum(na_pc8 > 0),
        " | AEF-full: ", sum(na_full > 0))

dat <- cbind(pa_xy, ex_wc, ex_pc8, ex_full)
dat <- dat[stats::complete.cases(dat), ]      # drop any point with NA predictors
message(sprintf("Modelling rows after NA drop: %d (pres=%d, bg=%d)",
                nrow(dat), sum(dat$occ == 1), sum(dat$occ == 0)))

# Column groups for each predictor set
cols_wc   <- names(wc)
cols_pc8  <- names(aef_pc8)
cols_full <- names(aef_full)

## ---------------------------------------------------------------------------
## 10. SPATIAL-BLOCK CV  -- folds assigned ONCE, shared across all models
## ---------------------------------------------------------------------------
dat_sf <- st_as_sf(dat, coords = c("lon","lat"), crs = 4326)
sb <- cv_spatial(x = dat_sf, column = "occ", r = template,
                 size = BLOCK_SIZE_M, k = K_FOLDS,
                 selection = "random", iteration = 50, progress = FALSE,
                 plot = FALSE)
dat$fold <- sb$folds_ids

## ---------------------------------------------------------------------------
## 11. HELPERS  -- fit / predict / metrics
## ---------------------------------------------------------------------------
fit_model <- function(method, train, preds) {
  if (method == "RF") {
    n1 <- sum(train$occ == 1)
    randomForest(x = train[, preds, drop = FALSE],
                 y = factor(train$occ),
                 ntree = 1000,
                 strata = factor(train$occ),
                 sampsize = c("0" = n1, "1" = n1))   # downsample background -> balanced
  } else {                                            # MaxEnt via maxnet
    maxnet(p = train$occ, data = train[, preds, drop = FALSE],
           f = maxnet.formula(train$occ, train[, preds, drop = FALSE],
                              classes = "default"))
  }
}

predict_vec <- function(method, model, newdata, preds) {
  if (method == "RF") {
    as.numeric(predict(model, newdata[, preds, drop = FALSE], type = "prob")[, "1"])
  } else {
    as.numeric(predict(model, newdata[, preds, drop = FALSE],
                       type = "cloglog", clamp = TRUE))
  }
}

eval_fold <- function(obs, pred) {
  pred <- pmin(pmax(pred, 1e-6), 1 - 1e-6)
  auc  <- as.numeric(pROC::auc(response = obs, predictor = pred, quiet = TRUE))
  boyce <- tryCatch(
    ecospat::ecospat.boyce(fit = pred, obs = pred[obs == 1],
                           PEplot = FALSE)$cor,
    error = function(e) NA_real_)
  cal <- tryCatch(
    coef(glm(obs ~ qlogis(pred), family = binomial))[2],  # Miller's calibration slope
    error = function(e) NA_real_)
  c(AUC = auc, Boyce = boyce, CalSlope = unname(cal))
}

## ---------------------------------------------------------------------------
## 12. RUN THE 2x2 (+ sensitivity) WITH SHARED FOLDS
## ---------------------------------------------------------------------------
configs <- list(
  list(name = "RF_WorldClim",         method = "RF",     preds = cols_wc),
  list(name = "RF_AlphaEarth",        method = "RF",     preds = cols_pc8),
  list(name = "MaxEnt_WorldClim",     method = "MaxEnt", preds = cols_wc),
  list(name = "MaxEnt_AlphaEarth",    method = "MaxEnt", preds = cols_pc8),
  list(name = "RF_AlphaEarth_full",   method = "RF",     preds = cols_full)  # sensitivity
)

cv_results <- list()
for (cfg in configs) {
  fold_metrics <- matrix(NA_real_, nrow = K_FOLDS, ncol = 3,
                         dimnames = list(NULL, c("AUC","Boyce","CalSlope")))
  for (f in seq_len(K_FOLDS)) {
    tr <- dat[dat$fold != f, ]; te <- dat[dat$fold == f, ]
    if (sum(te$occ == 1) < 5) next               # skip folds with too few test presences
    m  <- fit_model(cfg$method, tr, cfg$preds)
    p  <- predict_vec(cfg$method, m, te, cfg$preds)
    fold_metrics[f, ] <- eval_fold(te$occ, p)
  }
  cv_results[[cfg$name]] <- data.frame(
    model = cfg$name,
    AUC_mean      = mean(fold_metrics[,"AUC"],      na.rm = TRUE),
    AUC_sd        = sd(fold_metrics[,"AUC"],        na.rm = TRUE),
    Boyce_mean    = mean(fold_metrics[,"Boyce"],    na.rm = TRUE),
    Boyce_sd      = sd(fold_metrics[,"Boyce"],      na.rm = TRUE),
    CalSlope_mean = mean(fold_metrics[,"CalSlope"], na.rm = TRUE)
  )
}
metrics <- do.call(rbind, cv_results); rownames(metrics) <- NULL
print(metrics)
write.csv(metrics, file.path(OUT_DIR, "cv_metrics.csv"), row.names = FALSE)

## ---------------------------------------------------------------------------
## 13. FINAL MODELS ON ALL DATA  -> predict to the full 1 km grid
## ---------------------------------------------------------------------------
predstacks <- list(WorldClim = wc, AlphaEarth = aef_pc8)
fit_full_and_predict <- function(method, preds, stack) {
  m <- fit_model(method, dat, preds)
  if (method == "RF") {
    terra::predict(stack, m, type = "prob", index = 2, na.rm = TRUE)
  } else {
    terra::predict(stack, m, type = "cloglog", clamp = TRUE, na.rm = TRUE)
  }
}
rast_rf_wc  <- fit_full_and_predict("RF",     cols_wc,  predstacks$WorldClim)
rast_rf_ae  <- fit_full_and_predict("RF",     cols_pc8, predstacks$AlphaEarth)
rast_mx_wc  <- fit_full_and_predict("MaxEnt", cols_wc,  predstacks$WorldClim)
rast_mx_ae  <- fit_full_and_predict("MaxEnt", cols_pc8, predstacks$AlphaEarth)

suit <- c(rast_rf_wc, rast_rf_ae, rast_mx_wc, rast_mx_ae)
names(suit) <- c("RF_WC","RF_AE","MaxEnt_WC","MaxEnt_AE")
writeRaster(suit, file.path(OUT_DIR, "suitability_EAfrica.tif"), overwrite = TRUE)

## ---------------------------------------------------------------------------
## 14. AGREEMENT / DISAGREEMENT  (compare WC vs AE WITHIN each learner)
## ---------------------------------------------------------------------------
# Continuous difference maps (AE - WC): warm = AEF predicts more suitable
diff_rf <- rast_rf_ae - rast_rf_wc
diff_mx <- rast_mx_ae - rast_mx_wc

# Binary agreement using a 10th-percentile training-presence threshold per model
thr <- function(r) {
  v <- terra::extract(r, occ_thin[, c("lon","lat")], ID = FALSE)[, 1]
  as.numeric(quantile(v, probs = THRESH_PCT, na.rm = TRUE))
}
bin_rf_wc <- rast_rf_wc >= thr(rast_rf_wc)
bin_rf_ae <- rast_rf_ae >= thr(rast_rf_ae)
# 0 both-absent, 1 WC-only, 2 AE-only, 3 both-present
agree_rf <- bin_rf_wc + 2 * bin_rf_ae
levels(agree_rf) <- data.frame(
  value = 0:3, class = c("Both absent","WC only","AEF only","Both present"))

## ---------------------------------------------------------------------------
## 15. CLIP TO TANZANIA FOR DISPLAY  (metrics stay on the modelling extent)
## ---------------------------------------------------------------------------
tza <- vect(GADM_TZA)
clip_tza <- function(r) terra::mask(crop(r, tza), tza)

suit_tza    <- clip_tza(suit)
diff_rf_tza <- clip_tza(diff_rf)
diff_mx_tza <- clip_tza(diff_mx)
agree_rf_tza<- clip_tza(agree_rf)

writeRaster(suit_tza,     file.path(OUT_DIR, "suitability_Tanzania.tif"),  overwrite = TRUE)
writeRaster(diff_rf_tza,  file.path(OUT_DIR, "diff_RF_AE_minus_WC_TZA.tif"), overwrite = TRUE)
writeRaster(diff_mx_tza,  file.path(OUT_DIR, "diff_MaxEnt_AE_minus_WC_TZA.tif"), overwrite = TRUE)
writeRaster(agree_rf_tza, file.path(OUT_DIR, "agreement_RF_TZA.tif"),       overwrite = TRUE)

# Quick PNG previews
png(file.path(OUT_DIR, "suitability_Tanzania.png"), 1200, 1000, res = 130)
plot(suit_tza, main = names(suit_tza)); dev.off()

png(file.path(OUT_DIR, "agreement_RF_Tanzania.png"), 900, 1000, res = 130)
plot(agree_rf_tza, main = "RF: WorldClim vs AlphaEarth agreement")
plot(tza, add = TRUE, border = "grey40"); dev.off()

png(file.path(OUT_DIR, "diff_RF_Tanzania.png"), 900, 1000, res = 130)
plot(diff_rf_tza, main = "RF suitability: AlphaEarth - WorldClim")
plot(tza, add = TRUE, border = "grey40"); dev.off()

message("Done. Outputs written to: ", OUT_DIR)

## ---------------------------------------------------------------------------
## OPTIONAL: pull AlphaEarth via rgee instead of the JS export (fragile route)
## ---------------------------------------------------------------------------
# library(rgee); ee_Initialize(drive = TRUE)
# geom <- ee$Geometry$Rectangle(c(ext_mod[1], ext_mod[3], ext_mod[2], ext_mod[4]))
# col  <- ee$ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")$
#           filterDate("2018-01-01","2023-01-01")$filterBounds(geom)
# aef_ee <- col$mean()$clip(geom)
# aef_raw <- ee_as_rast(aef_ee, region = geom, scale = 1000, via = "drive")
###############################################################################
