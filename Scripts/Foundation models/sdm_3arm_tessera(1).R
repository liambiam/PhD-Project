###############################################################################
# Trial SDM comparison: WorldClim vs AlphaEarth vs TESSERA
# Species: Vangueria infausta | Region: East Africa (display clipped to TZ)
# Design: 2x3 {WorldClim, AlphaEarth, TESSERA} x {RandomForest, MaxEnt}
#         + AEF-full & TESSERA-full RF sensitivity checks (metrics only).
#
# TESSERA is sampled at POINTS (not a mosaic): a full 10 m TESSERA grid over
# Tanzania is ~1.6 TB. So the run is split around a Python step:
#   STAGE 1 (R): build presence+background points -> write pa_points.csv, stop.
#   (Python)   : tessera_sample_points.py reads it -> tessera_at_points.csv
#   STAGE 2 (R): read tessera_at_points.csv -> join -> full 2x3 comparison.
#
# WorldClim & AlphaEarth still use rasters (they are small at 1 km). Only the
# TESSERA arm is point-sampled, hence it gets CV metrics but no full-grid
# prediction map in this pilot (added later if it earns its place).
###############################################################################

install.packages(c(
  "terra",
  "sf",
  "dplyr",
  "randomForest",
  "maxnet",
  "blockCV",
  "ecospat",
  "pROC"
))

STAGE <- 2   # <-- set to 1, run; run Python; set to 2, run again.

suppressPackageStartupMessages({
  library(terra); library(sf); library(dplyr)
  library(randomForest); library(maxnet); library(blockCV)
  library(ecospat); library(pROC)
})
set.seed(42)

## ---------------------------------------------------------------------------
## 1. CONFIG
## ---------------------------------------------------------------------------
SPECIES   <- "Vangueria infausta"
OCC_CSV   <- "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/rainbio_published/published_database/RAINBIO.csv"
SP_COL    <- "tax_sp_level"; LON_COL <- "decimalLongitude"; LAT_COL <- "decimalLatitude"

WC_DIR     <- "C:/Users/liams/Documents/PhD-Project Data/worldclim/climate/wc2.1_30s/"
WC_PATTERN <- "wc2.1_30s_bio_%d.tif"
WC_BIOS    <- c(1, 4, 12, 15, 17)   # bio6 dropped (VIF 15.7)

AEF_TIF   <- "C:/Users/liams/Documents/PhD-Project Data/AlphaEarth/AEF_mean_2018_2022_EAfrica_1km.tif"
GADM_TZA  <- "C:/Users/liams/Documents/PhD-Project Data/GADM TZ Shape/gadm41_TZA_0.shp"
OUT_DIR   <- "C:/Users/liams/Documents/PhD-Project Data/SDM_trial_outputs/"

# CSV handoff with the Python TESSERA sampler
PA_CSV    <- "C:/Users/liams/Documents/PhD-Project Data/TESSERA/pa_points.csv"          # STAGE 1 writes
TESS_CSV  <- "C:/Users/liams/Documents/PhD-Project Data/TESSERA/tessera_at_points.csv"  # STAGE 2 reads
RDS_STATE <- "C:/Users/liams/Documents/PhD-Project Data/TESSERA/stage1_state.rds"       # carry STAGE1 objects

GRID_RES_DEG <- 1/30
BUFFER_DEG   <- 2.5
N_BACKGROUND <- 2000   # NOTE: large bg scatters across many TESSERA tiles.
                        # If the Python size gate is too big, drop to ~2000.
N_FM_PCS     <- 8
FM_VAR_KEEP  <- 0.95
K_FOLDS      <- 5
BLOCK_SIZE_M <- 200000
THRESH_PCT   <- 0.10

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(PA_CSV), showWarnings = FALSE, recursive = TRUE)

## ===========================================================================
## STAGE 1: occurrences, predictors (WC+AEF), mask, points -> write pa_points
## ===========================================================================
if (STAGE == 1) {

  ## PREFLIGHT (TESSERA not needed yet)
  pf <- c(OCC_CSV=file.exists(OCC_CSV), WC_DIR=dir.exists(WC_DIR),
          AEF_TIF=file.exists(AEF_TIF), GADM=file.exists(GADM_TZA))
  print(pf); if (any(!pf)) stop("Missing: ", paste(names(pf)[!pf], collapse=", "))

  ## OCCURRENCES
  occ_raw <- read.csv(OCC_CSV, stringsAsFactors = FALSE)
  stopifnot(all(c(SP_COL, LON_COL, LAT_COL) %in% names(occ_raw)))
  occ <- occ_raw %>%
    filter(.data[[SP_COL]] == SPECIES) %>%
    transmute(lon = as.numeric(.data[[LON_COL]]), lat = as.numeric(.data[[LAT_COL]])) %>%
    filter(!is.na(lon), !is.na(lat), abs(lon) <= 180, abs(lat) <= 90,
           !(lon == 0 & lat == 0)) %>% distinct()
  message(sprintf("Raw occurrences: %d", nrow(occ)))

  ## EXTENT + TEMPLATE
  occ_v <- terra::vect(occ, geom = c("lon","lat"), crs = "EPSG:4326")
  e <- terra::ext(occ_v)
  ext_mod <- terra::ext(e[1]-BUFFER_DEG, e[2]+BUFFER_DEG, e[3]-BUFFER_DEG, e[4]+BUFFER_DEG)
  afr <- c(-20, 55, -36, 20)
  ext_mod <- terra::ext(max(ext_mod[1],afr[1]), min(ext_mod[2],afr[2]),
                        max(ext_mod[3],afr[3]), min(ext_mod[4],afr[4]))
  template <- terra::rast(ext_mod, resolution = GRID_RES_DEG, crs = "EPSG:4326")
  message(sprintf("Grid cells: %s", format(terra::ncell(template), big.mark=",")))

  ## WORLDCLIM
  wc_files <- file.path(WC_DIR, sprintf(WC_PATTERN, WC_BIOS))
  stopifnot(all(file.exists(wc_files)))
  wc_raw <- rast(wc_files); names(wc_raw) <- paste0("bio", WC_BIOS)
  wc <- resample(crop(wc_raw, ext_mod, snap="out"), template, method="bilinear")
  vt <- tryCatch(usdm::vif(as.data.frame(wc, na.rm=TRUE)), error=function(e) NULL)
  if (!is.null(vt)) { message("WorldClim VIF:"); print(vt) }

  ## ALPHAEARTH PCA (same helper would serve TESSERA if it were a raster)
  load_fm_pcs <- function(tif, label, n_pcs, var_keep, template, ext_mod, prefix) {
    r_raw <- rast(tif); message(sprintf("%s bands: %d", label, nlyr(r_raw)))
    r <- resample(crop(r_raw, ext_mod, snap="out"), template, method="bilinear")
    samp <- spatSample(r, size=50000, method="random", na.rm=TRUE, values=TRUE)
    pca <- prcomp(samp, center=TRUE, scale.=TRUE)
    vcum <- cumsum(pca$sdev^2)/sum(pca$sdev^2); nfull <- which(vcum>=var_keep)[1]
    message(sprintf("%s PCs: %d explain %.0f%%; keep %d primary, %d full",
                    label, n_pcs, 100*vcum[n_pcs], n_pcs, nfull))
    pcs <- terra::predict(r, pca); names(pcs) <- paste0(prefix, seq_len(nlyr(pcs)))
    list(pc_primary = pcs[[1:n_pcs]], pc_full = pcs[[1:nfull]])
  }
  aef <- load_fm_pcs(AEF_TIF, "AlphaEarth", N_FM_PCS, FM_VAR_KEEP, template, ext_mod, "AEPC")
  aef_pc8 <- aef$pc_primary; aef_full <- aef$pc_full

  ## MASK (WC + AEF; TESSERA coverage handled later via NA drop on its columns)
  wc_f  <- focal(wc,      w=3, fun="mean", na.policy="only", na.rm=TRUE)
  ae_f  <- focal(aef_pc8, w=3, fun="mean", na.policy="only", na.rm=TRUE)
  mask  <- !is.na(wc_f[[1]]) & !is.na(ae_f[[1]]); mask[mask==0] <- NA
  wc <- terra::mask(wc_f, mask); aef_pc8 <- terra::mask(ae_f, mask)
  aef_full <- terra::mask(aef_full, mask)

  ## THIN
  cells <- cellFromXY(template, as.matrix(occ[,c("lon","lat")]))
  occ_thin <- occ[!is.na(cells) & !duplicated(cells), ]
  message(sprintf("Thinned: %d -> %d", nrow(occ), nrow(occ_thin)))

  ## BACKGROUND + PA points
  bg <- spatSample(mask, size=N_BACKGROUND, method="random",
                   na.rm=TRUE, xy=TRUE, values=TRUE)
  bg <- as.data.frame(bg)[,c("x","y")]; names(bg) <- c("lon","lat")
  pa_xy <- rbind(data.frame(occ=1, occ_thin[,c("lon","lat")]),
                 data.frame(occ=0, bg))

  ## Write points for Python; save raster-derived predictors for STAGE 2.
  write.csv(pa_xy, PA_CSV, row.names = FALSE)
  message(sprintf("STAGE 1 wrote %d points -> %s", nrow(pa_xy), PA_CSV))

  ex_wc      <- terra::extract(wc,       pa_xy[,c("lon","lat")], ID=FALSE)
  ex_aef     <- terra::extract(aef_pc8,  pa_xy[,c("lon","lat")], ID=FALSE)
  ex_aef_full<- terra::extract(aef_full, pa_xy[,c("lon","lat")], ID=FALSE)

  saveRDS(list(pa_xy=pa_xy, ex_wc=ex_wc, ex_aef=ex_aef, ex_aef_full=ex_aef_full,
               cols_wc=names(wc), cols_aef=names(aef_pc8), cols_aef_full=names(aef_full),
               template=wrap(template)), RDS_STATE)
  message("STAGE 1 done. Now run tessera_sample_points.py, then set STAGE <- 2.")
}

## ===========================================================================
## STAGE 2: join TESSERA, run 2x3 comparison
## ===========================================================================
if (STAGE == 2) {

  stopifnot(file.exists(RDS_STATE), file.exists(TESS_CSV))
  s <- readRDS(RDS_STATE)
  pa_xy <- s$pa_xy; template <- unwrap(s$template)

  ## Join TESSERA values (matched row order: occ,lon,lat keys)
  tess <- read.csv(TESS_CSV)
  ts_cols <- grep("^TS[0-9]+$", names(tess), value = TRUE)
  stopifnot(length(ts_cols) == 128)
  # align by lon/lat/occ to be safe rather than assuming identical row order
  key <- function(d) paste(d$occ, round(d$lon,6), round(d$lat,6))
  tess <- tess[match(key(pa_xy), key(tess)), ]
  ex_tess <- tess[, ts_cols]

  ## PCA on TESSERA point values (cannot PCA a raster we don't have).
  ## Fit on background rows (the prediction universe proxy), apply to all.
  bg_rows <- pa_xy$occ == 0 & stats::complete.cases(ex_tess)
  pca_t <- prcomp(ex_tess[bg_rows, ], center = TRUE, scale. = TRUE)
  vcum  <- cumsum(pca_t$sdev^2)/sum(pca_t$sdev^2)
  nfull <- which(vcum >= FM_VAR_KEEP)[1]
  message(sprintf("TESSERA PCs: %d explain %.0f%%; keep %d primary, %d full",
                  N_FM_PCS, 100*vcum[N_FM_PCS], N_FM_PCS, nfull))
  ts_scores <- predict(pca_t, ex_tess)
  ex_ts      <- as.data.frame(ts_scores[, 1:N_FM_PCS]); names(ex_ts) <- paste0("TSPC", 1:N_FM_PCS)
  ex_ts_full <- as.data.frame(ts_scores[, 1:nfull]);    names(ex_ts_full) <- paste0("TSPC", 1:nfull)

  ## Assemble modelling frame
  dat <- cbind(pa_xy, s$ex_wc, s$ex_aef, s$ex_aef_full, ex_ts, ex_ts_full)
  dat <- dat[stats::complete.cases(dat), ]
  message(sprintf("Modelling rows: %d (pres=%d, bg=%d)",
                  nrow(dat), sum(dat$occ==1), sum(dat$occ==0)))

  cols_wc <- s$cols_wc; cols_aef <- s$cols_aef; cols_aef_full <- s$cols_aef_full
  cols_ts <- names(ex_ts); cols_ts_full <- names(ex_ts_full)

  ## SPATIAL CV
  dat_sf <- st_as_sf(dat, coords=c("lon","lat"), crs=4326)
  sb <- cv_spatial(x=dat_sf, column="occ", r=template, size=BLOCK_SIZE_M,
                   k=K_FOLDS, selection="random", iteration=50,
                   progress=FALSE, plot=FALSE)
  dat$fold <- sb$folds_ids

  ## HELPERS
  fit_model <- function(method, train, preds) {
    if (method=="RF") {
      n1 <- sum(train$occ==1)
      randomForest(x=train[,preds,drop=FALSE], y=factor(train$occ), ntree=1000,
                   strata=factor(train$occ), sampsize=c("0"=n1,"1"=n1))
    } else {
      maxnet(p=train$occ, data=train[,preds,drop=FALSE],
             f=maxnet.formula(train$occ, train[,preds,drop=FALSE], classes="default"))
    }
  }
  predict_vec <- function(method, model, newdata, preds) {
    if (method=="RF") as.numeric(predict(model, newdata[,preds,drop=FALSE], type="prob")[,"1"])
    else as.numeric(predict(model, newdata[,preds,drop=FALSE], type="cloglog", clamp=TRUE))
  }
  eval_fold <- function(obs, pred) {
    pred <- pmin(pmax(pred,1e-6),1-1e-6)
    auc <- as.numeric(pROC::auc(response=obs, predictor=pred, quiet=TRUE))
    boyce <- tryCatch(ecospat::ecospat.boyce(fit=pred, obs=pred[obs==1], PEplot=FALSE)$cor,
                      error=function(e) NA_real_)
    cal <- tryCatch(coef(glm(obs ~ qlogis(pred), family=binomial))[2],
                    error=function(e) NA_real_)
    c(AUC=auc, Boyce=boyce, CalSlope=unname(cal))
  }

  ## RUN 2x3 + sensitivity
  configs <- list(
    list(name="RF_WorldClim",       method="RF",     preds=cols_wc),
    list(name="RF_AlphaEarth",      method="RF",     preds=cols_aef),
    list(name="RF_TESSERA",         method="RF",     preds=cols_ts),
    list(name="MaxEnt_WorldClim",   method="MaxEnt", preds=cols_wc),
    list(name="MaxEnt_AlphaEarth",  method="MaxEnt", preds=cols_aef),
    list(name="MaxEnt_TESSERA",     method="MaxEnt", preds=cols_ts),
    list(name="RF_AlphaEarth_full", method="RF",     preds=cols_aef_full),
    list(name="RF_TESSERA_full",    method="RF",     preds=cols_ts_full)
  )
  cv_results <- list()
  for (cfg in configs) {
    fm <- matrix(NA_real_, K_FOLDS, 3, dimnames=list(NULL,c("AUC","Boyce","CalSlope")))
    for (f in seq_len(K_FOLDS)) {
      tr <- dat[dat$fold!=f,]; te <- dat[dat$fold==f,]
      if (sum(te$occ==1) < 5) next
      m <- fit_model(cfg$method, tr, cfg$preds)
      p <- predict_vec(cfg$method, m, te, cfg$preds)
      fm[f,] <- eval_fold(te$occ, p)
    }
    cv_results[[cfg$name]] <- data.frame(
      model=cfg$name,
      AUC_mean=mean(fm[,"AUC"],na.rm=TRUE), AUC_sd=sd(fm[,"AUC"],na.rm=TRUE),
      Boyce_mean=mean(fm[,"Boyce"],na.rm=TRUE), Boyce_sd=sd(fm[,"Boyce"],na.rm=TRUE),
      CalSlope_mean=mean(fm[,"CalSlope"],na.rm=TRUE))
  }
  metrics <- do.call(rbind, cv_results); rownames(metrics) <- NULL
  print(metrics)
  write.csv(metrics, file.path(OUT_DIR, "cv_metrics_3arm.csv"), row.names=FALSE)
  message("STAGE 2 done. Metrics -> ", file.path(OUT_DIR, "cv_metrics_3arm.csv"))
  message("NOTE: WC & AEF full-grid prediction maps unchanged from 2-arm run; ",
          "TESSERA map deferred (point-sampled only).")
}
###############################################################################
