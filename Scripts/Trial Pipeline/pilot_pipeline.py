"""
Pilot functional-ecology pipeline for the 224 complete-trait species.
================================================================================
End-to-end skeleton implementing the workflow agreed in supervision:

    1. Species richness        (grid the occurrences, count species per cell)
    2. SDMs                    (presence-background RF per species)
    3. Stack -> predicted richness, compare with raw richness
    4. Functional diversity    (FDis, MNTD on the LHS traits)
    5. FD vs richness          (per-cell correlation + residual map)
    6. Functional redundancy   (richness relative to functional dispersion)

Designed to PROVE THE ROUTINE on a few dozen species, then scale by feeding
more species through the same code (Neil's plan). All six stages run on either
your real RAINBIO/TRY table or a synthetic stand-in with the identical schema.

------------------------------------------------------------------------------
TO RUN ON YOUR REAL DATA
------------------------------------------------------------------------------
Set USE_SYNTHETIC = False and point DATA_PATH at:
    C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/rainbio_LHS_traits.csv
(Use forward slashes or a raw string r"..." on Windows.)

SDM predictors use the 19 WorldClim bioclim layers. Set WORLDCLIM_DIR to the
folder containing wc2.1_10m_bio_1.tif ... wc2.1_10m_bio_19.tif. Requires
rasterio (pip install rasterio). Points on nodata pixels (ocean/coast) are
dropped from training; grid cells with no climate data are masked from output.

The trait columns used are configurable in COLS below. NOTE: the leaf trait in
your file is SLA (specific leaf area), not raw leaf area -- a standard and
defensible substitute in the LHS / leaf-economics framing. height_generative is
kept available as an alternative height measure.
================================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- 
# CONFIG
# ----------------------------------------------------------------------------- 
USE_SYNTHETIC = False   # <- set False and fill DATA_PATH to use your real file
DATA_PATH = r"C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/rainbio_LHS_traits.csv"
OUTDIR = r"C:/Users/liams/Documents/PhD-Project Data/pilot_pipeline_output"
RANDOM_SEED = 42

# Column names as they appear in rainbio_LHS_traits.csv
COLS = dict(
    species="species",
    lat="decimalLatitude",
    lon="decimalLongitude",
    family="family",
    habit="a_habit",
    habitat="habitat_name.x",        # .y is a duplicate join artefact -> dropped
    # --- the working LHS trait triplet ---
    height="height",                 # vegetative height (m)
    leaf="SLA",                      # specific leaf area (leaf economics axis)
    seed="seed_mass",                # seed mass (mg)  -> proxy for seed volume
)
TRAITS = ["height", "leaf", "seed"]  # logical names; mapped via COLS

# Tanzania-ish bounding box (used for grid + synthetic data + background pts)
BBOX = dict(lat_min=-11.8, lat_max=-1.0, lon_min=29.3, lon_max=40.5)

# --- Tanzania boundary (GADM) -----------------------------------------------
# Level-0 country outline, e.g. gadm41_TZA_0.shp. Used to mask grid cells and
# background points to within Tanzania. Set to None to disable masking. 
GADM_PATH = r"C:/Users/liams/Documents/PhD-Project Data/GADM TZ Shape/gadm41_TZA_1.shp"

GRID_RES_DEG = 0.4     # richness/FD grid cell size in degrees
N_BACKGROUND = 5000    # background points per SDM

# --- WorldClim bioclim predictors -------------------------------------------
# Folder holding wc2.1_10m_bio_1.tif ... wc2.1_10m_bio_19.tif
WORLDCLIM_DIR = r"C:/Users/liams/Documents/PhD-Project Data/Worldclim climate data/wc2.1_10m"
WORLDCLIM_PATTERN = "wc2.1_10m_bio_{i}.tif"   # {i} = 1..19
BIO_VARS = list(range(1, 20))                 # bio1..bio19
# When USE_SYNTHETIC is True we fabricate this many pseudo-layers instead.
N_PSEUDO_PREDICTORS = len(BIO_VARS)

rng = np.random.default_rng(RANDOM_SEED)


@dataclass
class PipelineData:
    """Container passed between stages."""
    occ: pd.DataFrame                 # occurrences, complete-trait species only
    traits: pd.DataFrame              # one row per species, LHS traits
    species: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# DATA LOADING  +  SYNTHETIC GENERATOR (identical schema to your CSV)
# -----------------------------------------------------------------------------
def _make_synthetic(n_species=300, mean_occ=40):
    """Build a table with the same columns as rainbio_LHS_traits.csv.

    ~300 species are generated but only a subset will have all three traits
    non-NA, mimicking the real situation where the 'complete-trait' set
    (your 224) is a subsequently-derived subset.
    """
    fams = ["Acanthaceae", "Fabaceae", "Rubiaceae", "Asteraceae", "Poaceae",
            "Euphorbiaceae", "Apocynaceae", "Malvaceae"]
    habits = ["tree", "shrub", "herb", "liana", "graminoid"]
    habitats = ["Forest - Subtropical/tropical moist montane",
                "Forest - Subtropical/tropical dry",
                "Shrubland - Subtropical/tropical high altitude",
                "Savanna - Dry", "Grassland - Subtropical/tropical"]
    rows = []
    for i in range(n_species):
        sp = f"Genus{i//12} species{i%12}"
        fam = fams[i % len(fams)]
        habit = habits[i % len(habits)]
        # species-level trait values (constant within species, as in your file)
        h = float(np.exp(rng.normal(1.0, 0.9)))          # height ~ lognormal
        sla = float(np.exp(rng.normal(2.6, 0.5)))         # SLA
        sm = float(np.exp(rng.normal(1.5, 1.4)))          # seed mass
        # introduce NA gaps to mimic ~partial coverage -> derived complete set
        if rng.random() < 0.25: h = np.nan
        if rng.random() < 0.30: sla = np.nan
        if rng.random() < 0.20: sm = np.nan
        # a spatial niche centre for this species (so SDMs have signal)
        clat = rng.uniform(BBOX["lat_min"], BBOX["lat_max"])
        clon = rng.uniform(BBOX["lon_min"], BBOX["lon_max"])
        n_occ = max(3, int(rng.poisson(mean_occ)))
        for _ in range(n_occ):
            lat = np.clip(rng.normal(clat, 1.2), BBOX["lat_min"], BBOX["lat_max"])
            lon = np.clip(rng.normal(clon, 1.2), BBOX["lon_min"], BBOX["lon_max"])
            hab = habitats[rng.integers(len(habitats))]
            rows.append(dict(
                species=sp, decimalLatitude=round(lat, 4),
                decimalLongitude=round(lon, 4), family=fam, a_habit=habit,
                habitat_iucn=int(rng.integers(100, 400)),
                **{"habitat_name.x": hab, "habitat_name.y": hab},
                height_generative=np.nan,
                SLA=sla, height=h, seed_mass=sm,
            ))
    return pd.DataFrame(rows)


def load_data() -> PipelineData:
    """Load real CSV or synthetic, then derive the complete-trait subset."""
    if USE_SYNTHETIC:
        print("[load] Using SYNTHETIC data (schema-matched stand-in).")
        df = _make_synthetic()
    else:
        print(f"[load] Reading {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

    # coerce 'NA' strings -> real NaN for the trait columns
    for logical in TRAITS:
        col = COLS[logical]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sp_col = COLS["species"]
    trait_cols = [COLS[t] for t in TRAITS]

    # species-level trait table: one row per species (traits constant within sp)
    sp_traits = (df.groupby(sp_col)[trait_cols]
                   .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan))

    # complete-trait species = all three LHS traits present
    complete_mask = sp_traits[trait_cols].notna().all(axis=1)
    complete_species = sp_traits.index[complete_mask].tolist()
    print(f"[load] {len(sp_traits)} species total; "
          f"{len(complete_species)} with all 3 LHS traits (the working pool).")

    occ = df[df[sp_col].isin(complete_species)].copy()
    traits = sp_traits.loc[complete_species].copy()
    traits.columns = TRAITS  # rename to logical names height/leaf/seed

    return PipelineData(occ=occ, traits=traits, species=complete_species)


# -----------------------------------------------------------------------------
# GRID HELPERS
# -----------------------------------------------------------------------------
def _cell_ids(lat, lon, res=GRID_RES_DEG):
    """Rectangular grid cell index. Swap here for H3 if you prefer hexagons:
       import h3; return h3.latlng_to_cell(lat, lon, resolution)."""
    r = np.floor((lat - BBOX["lat_min"]) / res).astype(int)
    c = np.floor((lon - BBOX["lon_min"]) / res).astype(int)
    return r, c


def _cell_centroids(res=GRID_RES_DEG):
    """All grid cell (row,col) -> centroid lat/lon over the bbox."""
    nr = int(np.ceil((BBOX["lat_max"] - BBOX["lat_min"]) / res))
    nc = int(np.ceil((BBOX["lon_max"] - BBOX["lon_min"]) / res))
    cells = {}
    for r in range(nr):
        for c in range(nc):
            cells[(r, c)] = (BBOX["lat_min"] + (r + 0.5) * res,
                             BBOX["lon_min"] + (c + 0.5) * res)
    return cells


# -----------------------------------------------------------------------------
# TANZANIA BOUNDARY MASK  (GADM level-0)
# -----------------------------------------------------------------------------
_BOUNDARY_CACHE = {}


def _tz_boundary():
    """Load and cache the dissolved Tanzania polygon. Returns None in synthetic
    mode or if GADM_PATH is unset, so masking silently no-ops."""
    if USE_SYNTHETIC or not GADM_PATH:
        return None
    if "geom" in _BOUNDARY_CACHE:
        return _BOUNDARY_CACHE["geom"]
    import geopandas as gpd
    if not os.path.exists(GADM_PATH):
        raise FileNotFoundError(
            f"GADM boundary not found: {GADM_PATH}\n"
            f"Set GADM_PATH at the top of the file (e.g. gadm41_TZA_0.shp), "
            f"or set it to None to disable masking.")
    gdf = gpd.read_file(GADM_PATH).to_crs("EPSG:4326")
    geom = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
    _BOUNDARY_CACHE["geom"] = geom
    print("[boundary] loaded Tanzania outline from GADM")
    return geom


def _inside_tz(lat, lon):
    """Boolean mask: which (lat,lon) points fall inside Tanzania.
    Returns all-True (bbox) when no boundary is available."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    geom = _tz_boundary()
    if geom is None:
        return np.ones(len(lat), dtype=bool)
    from shapely import points, contains          # shapely 2.x vectorised
    pts = points(lon, lat)                         # (x=lon, y=lat)
    return contains(geom, pts)


# -----------------------------------------------------------------------------
# STAGE 1 : SPECIES RICHNESS
# -----------------------------------------------------------------------------
def stage1_richness(data: PipelineData) -> pd.DataFrame:
    occ = data.occ
    r, c = _cell_ids(occ[COLS["lat"]].values, occ[COLS["lon"]].values)
    occ = occ.assign(_r=r, _c=c)
    grp = occ.groupby(["_r", "_c"])
    rich = grp[COLS["species"]].nunique().rename("richness")
    effort = grp.size().rename("n_records")
    out = pd.concat([rich, effort], axis=1).reset_index()
    print(f"[stage1] richness over {len(out)} occupied cells; "
          f"max={out.richness.max()}, median={out.richness.median():.1f}")
    return out


# -----------------------------------------------------------------------------
# STAGE 2 : SDMs  (presence-background Random Forest, spatial-block CV)
# -----------------------------------------------------------------------------
_RASTER_CACHE = {}   # path -> opened rasterio dataset (opened once, reused)


def _open_worldclim():
    """Open all 19 bioclim rasters once and cache them. Returns ordered list of
    (bio_index, dataset). Raises a clear error if a file is missing."""
    import rasterio
    if _RASTER_CACHE:
        return _RASTER_CACHE["datasets"]
    datasets = []
    for i in BIO_VARS:
        path = os.path.join(WORLDCLIM_DIR, WORLDCLIM_PATTERN.format(i=i))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"WorldClim layer not found: {path}\n"
                f"Check WORLDCLIM_DIR / WORLDCLIM_PATTERN at the top of the file.")
        datasets.append((i, rasterio.open(path)))
    _RASTER_CACHE["datasets"] = datasets
    print(f"[worldclim] opened {len(datasets)} bioclim rasters")
    return datasets


def _extract_worldclim(lat, lon):
    """Sample all 19 bioclim layers at the given lat/lon points.
    Returns an (n_points, 19) array; nodata pixels become np.nan."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    datasets = _open_worldclim()
    coords = list(zip(lon, lat))           # rasterio.sample expects (x, y)=(lon,lat)
    cols = []
    for _, ds in datasets:
        nodata = ds.nodata
        vals = np.array([v[0] for v in ds.sample(coords)], dtype=float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
        # WorldClim sometimes uses very large negative sentinels
        vals[vals < -3.0e38] = np.nan
        cols.append(vals)
    return np.column_stack(cols)


def _pseudo_predictors(lat, lon):
    """Synthetic stand-in producing N_PSEUDO_PREDICTORS smooth spatial layers,
    used only when USE_SYNTHETIC is True so the pipeline stays testable
    without the rasters present."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    base = [lat, lon, lat * lon, np.sin(lat / 3.0), np.cos(lon / 3.0),
            (lat ** 2 + lon ** 2) ** 0.5]
    feats = [base[k % len(base)] * (1.0 + 0.1 * (k // len(base)))
             for k in range(N_PSEUDO_PREDICTORS)]
    return np.column_stack(feats)


def _predictors(lat, lon):
    """Dispatch: real WorldClim extraction, or synthetic layers."""
    if USE_SYNTHETIC:
        return _pseudo_predictors(lat, lon)
    return _extract_worldclim(lat, lon)


def _spatial_block_auc(X, y, coords, n_blocks=4):
    """Quick spatial-block CV AUC (blocks by longitude bands)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    lon = coords[:, 1]
    edges = np.quantile(lon, np.linspace(0, 1, n_blocks + 1))
    fold = np.clip(np.digitize(lon, edges[1:-1]), 0, n_blocks - 1)
    aucs = []
    for f in range(n_blocks):
        tr, te = fold != f, fold == f
        if y[te].sum() == 0 or y[tr].sum() == 0 or te.sum() < 5:
            continue
        m = RandomForestClassifier(n_estimators=150, max_depth=8,
                                   class_weight="balanced",
                                   random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else np.nan


def stage2_sdms(data: PipelineData, max_species=None):
    """Fit one presence-background RF per species; predict to grid centroids.
    Uses real WorldClim bioclim layers (or synthetic stand-ins). Points on
    nodata pixels (e.g. ocean) are dropped from training; grid cells whose
    predictors are all nodata are masked out of the prediction.
    Returns: dict species -> predicted suitability per cell, plus an AUC table.
    """
    from sklearn.ensemble import RandomForestClassifier

    cells = _cell_centroids()
    cell_keys = list(cells.keys())
    cell_lat = np.array([cells[k][0] for k in cell_keys])
    cell_lon = np.array([cells[k][1] for k in cell_keys])
    Xgrid = _predictors(cell_lat, cell_lon)
    grid_climate = ~np.isnan(Xgrid).any(axis=1)   # cells with full predictor data
    grid_incountry = _inside_tz(cell_lat, cell_lon)  # cells inside Tanzania
    grid_valid = grid_climate & grid_incountry
    # column means (over valid cells) for imputing any stray NaNs at predict time
    col_means = np.nanmean(Xgrid, axis=0)
    Xgrid_filled = np.where(np.isnan(Xgrid), col_means, Xgrid)
    print(f"[stage2] {grid_valid.sum()}/{len(cell_keys)} grid cells valid "
          f"(inside Tanzania + complete climate data)")

    occ = data.occ
    species = data.species if max_species is None else data.species[:max_species]
    preds, auc_rows = {}, []

    for i, sp in enumerate(species):
        pres = occ[occ[COLS["species"]] == sp]
        plat = pres[COLS["lat"]].values.astype(float)
        plon = pres[COLS["lon"]].values.astype(float)
        Xp = _predictors(plat, plon)
        keep_p = ~np.isnan(Xp).any(axis=1)        # drop presences on nodata
        Xp = Xp[keep_p]
        if len(Xp) < 5:                           # too few usable presences
            auc_rows.append(dict(species=sp, n_presence=int(len(Xp)),
                                 n_raw=int(len(plat)), auc=np.nan))
            continue
        plat, plon = plat[keep_p], plon[keep_p]

        # background sample across bbox, dropping ocean/nodata and out-of-country
        blat = rng.uniform(BBOX["lat_min"], BBOX["lat_max"], N_BACKGROUND)
        blon = rng.uniform(BBOX["lon_min"], BBOX["lon_max"], N_BACKGROUND)
        Xb = _predictors(blat, blon)
        keep_b = (~np.isnan(Xb).any(axis=1)) & _inside_tz(blat, blon)
        Xb, blat, blon = Xb[keep_b], blat[keep_b], blon[keep_b]

        X = np.vstack([Xp, Xb])
        y = np.r_[np.ones(len(Xp)), np.zeros(len(Xb))]
        coords = np.vstack([np.c_[plat, plon], np.c_[blat, blon]])

        auc = _spatial_block_auc(X, y, coords)
        m = RandomForestClassifier(n_estimators=200, max_depth=10,
                                   class_weight="balanced",
                                   random_state=RANDOM_SEED, n_jobs=-1)
        m.fit(X, y)
        p = m.predict_proba(Xgrid_filled)[:, 1]
        p[~grid_valid] = np.nan                   # mask nodata cells in output
        preds[sp] = p
        auc_rows.append(dict(species=sp, n_presence=int(len(Xp)),
                             n_raw=int(len(plat)), auc=auc))
        if (i + 1) % 25 == 0:
            print(f"[stage2] fitted {i+1}/{len(species)} SDMs")

    auc_tab = pd.DataFrame(auc_rows)
    print(f"[stage2] {auc_tab.auc.notna().sum()} SDMs fitted; "
          f"mean block-CV AUC={auc_tab.auc.mean():.3f}")
    return dict(preds=preds, cell_keys=cell_keys, auc=auc_tab,
                cell_lat=cell_lat, cell_lon=cell_lon, grid_valid=grid_valid)


# -----------------------------------------------------------------------------
# STAGE 3 : STACK SDMs -> PREDICTED RICHNESS ; COMPARE WITH RAW
# -----------------------------------------------------------------------------
def stage3_compare(sdm_out, richness_df, threshold=0.5):
    """Stack thresholded suitabilities into predicted richness, compare to raw."""
    from scipy.stats import spearmanr
    cell_keys = sdm_out["cell_keys"]
    stack = np.zeros(len(cell_keys))
    for sp, p in sdm_out["preds"].items():
        stack += np.where(np.isnan(p), 0.0, (p >= threshold).astype(float))
    pred = pd.DataFrame({"_r": [k[0] for k in cell_keys],
                         "_c": [k[1] for k in cell_keys],
                         "pred_richness": stack})
    merged = pred.merge(richness_df, on=["_r", "_c"], how="left")
    merged["richness"] = merged["richness"].fillna(0)
    occupied = merged[merged.richness > 0]
    rho, _ = spearmanr(occupied.richness, occupied.pred_richness)
    merged["resid"] = merged.pred_richness - merged.richness
    print(f"[stage3] Spearman(raw, predicted richness) over occupied cells "
          f"= {rho:.3f}")
    return merged, rho


# -----------------------------------------------------------------------------
# TRAIT PREPARATION
# -----------------------------------------------------------------------------
def _prepare_traits(traits: pd.DataFrame):
    """log-transform (traits are right-skewed) then z-scale. Returns scaled df."""
    from sklearn.preprocessing import StandardScaler
    X = traits[TRAITS].copy()
    # log1p on strictly-positive trait values (height, SLA, seed mass all >0)
    X = np.log1p(X.clip(lower=0))
    scaled = pd.DataFrame(StandardScaler().fit_transform(X),
                          index=traits.index, columns=TRAITS)
    return scaled


# -----------------------------------------------------------------------------
# STAGE 4 : FUNCTIONAL DIVERSITY  (FDis and MNTD, presence-only)
# -----------------------------------------------------------------------------
def _fdis(coords):
    """Functional dispersion (Laliberte & Legendre 2010), unweighted:
    mean distance of species to the assemblage centroid in trait space."""
    if len(coords) < 2:
        return np.nan
    centroid = coords.mean(axis=0)
    return float(np.sqrt(((coords - centroid) ** 2).sum(axis=1)).mean())


def _mntd(coords):
    """Mean nearest taxon distance: mean over species of the distance to the
    nearest other species in trait space (presence-only)."""
    n = len(coords)
    if n < 2:
        return np.nan
    from scipy.spatial.distance import cdist
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def stage4_funcdiv(data: PipelineData, richness_df):
    scaled = _prepare_traits(data.traits)
    occ = data.occ
    r, c = _cell_ids(occ[COLS["lat"]].values, occ[COLS["lon"]].values)
    occ = occ.assign(_r=r, _c=c)

    rows = []
    for (rr, cc), g in occ.groupby(["_r", "_c"]):
        sp_here = g[COLS["species"]].unique()
        coords = scaled.loc[scaled.index.intersection(sp_here)].values
        rows.append(dict(_r=rr, _c=cc, n_species=len(sp_here),
                         FDis=_fdis(coords), MNTD=_mntd(coords)))
    fd = pd.DataFrame(rows)
    fd = fd.merge(richness_df[["_r", "_c", "richness"]], on=["_r", "_c"],
                  how="left")
    print(f"[stage4] FD computed for {len(fd)} cells; "
          f"mean FDis={fd.FDis.mean():.3f}, mean MNTD={fd.MNTD.mean():.3f}")
    return fd


# -----------------------------------------------------------------------------
# STAGE 5 : FD vs RICHNESS  (correlation + residuals)
# -----------------------------------------------------------------------------
def stage5_fd_vs_richness(fd_df):
    """Regress FDis on richness; residuals flag functionally over/under-
    dispersed cells (more/less trait diversity than richness predicts)."""
    from scipy.stats import spearmanr
    import numpy as np
    d = fd_df.dropna(subset=["FDis", "richness"])
    d = d[d.richness >= 2]
    rho, p = spearmanr(d.richness, d.FDis)
    # simple linear fit of FDis ~ log(richness) for residual map
    x = np.log(d.richness.values); yv = d.FDis.values
    b1, b0 = np.polyfit(x, yv, 1)
    resid = yv - (b0 + b1 * x)
    out = d.copy()
    out["fd_resid"] = resid
    print(f"[stage5] Spearman(richness, FDis) = {rho:.3f} (p={p:.2g}); "
          f"residuals flag {int((resid>0).sum())} over- / "
          f"{int((resid<0).sum())} under-dispersed cells")
    return out, rho


# -----------------------------------------------------------------------------
# STAGE 6 : FUNCTIONAL REDUNDANCY
# -----------------------------------------------------------------------------
# NOTE FOR LIAM: this is the routine Neil asked you to write yourself, so treat
# the definition below as a STARTING POINT to make your own, not gospel.
#
# Implemented here as the widely-used contrast:
#     FR = species richness  -  functional richness (effective trait groups)
# where "functional richness" is estimated as the number of distinct trait
# clusters present in a cell (species packed into the same trait neighbourhood
# are functionally redundant). High FR => many species sharing trait space
# => more buffering / insurance. We report both an absolute and a normalised
# (0-1) version. Alternative framings to consider: Renyi/Hill-number based
# redundancy (de Bello et al. 2007), or FR = 1 - (FD / richness).
# -----------------------------------------------------------------------------
def stage6_redundancy(data: PipelineData, richness_df, cluster_eps=0.75):
    from sklearn.cluster import AgglomerativeClustering
    scaled = _prepare_traits(data.traits)
    occ = data.occ
    r, c = _cell_ids(occ[COLS["lat"]].values, occ[COLS["lon"]].values)
    occ = occ.assign(_r=r, _c=c)

    rows = []
    for (rr, cc), g in occ.groupby(["_r", "_c"]):
        sp_here = g[COLS["species"]].unique()
        coords = scaled.loc[scaled.index.intersection(sp_here)].values
        n = len(coords)
        if n < 2:
            fric = float(n)
        else:
            cl = AgglomerativeClustering(
                n_clusters=None, distance_threshold=cluster_eps,
                linkage="average")
            labels = cl.fit_predict(coords)
            fric = float(len(set(labels)))      # effective trait groups
        fr_abs = n - fric                        # redundant species count
        fr_norm = fr_abs / n if n > 0 else 0.0   # 0 = all unique, ->1 = packed
        rows.append(dict(_r=rr, _c=cc, n_species=n, func_groups=fric,
                         FR_abs=fr_abs, FR_norm=fr_norm))
    fr = pd.DataFrame(rows)
    print(f"[stage6] redundancy over {len(fr)} cells; "
          f"mean FR_norm={fr.FR_norm.mean():.3f} "
          f"(0=all functionally unique, 1=highly redundant)")
    return fr


# -----------------------------------------------------------------------------
# PLOTTING  (quick diagnostic maps/figures; not publication-final)
# -----------------------------------------------------------------------------
def _incountry_cell_mask():
    """(nr,nc) boolean array: which grid cells have centroids inside Tanzania.
    Cached. All-True when no boundary available."""
    if "cellmask" in _BOUNDARY_CACHE:
        return _BOUNDARY_CACHE["cellmask"]
    nr = int(np.ceil((BBOX["lat_max"] - BBOX["lat_min"]) / GRID_RES_DEG))
    nc = int(np.ceil((BBOX["lon_max"] - BBOX["lon_min"]) / GRID_RES_DEG))
    cells = _cell_centroids()
    lat = np.array([cells[(r, c)][0] for r in range(nr) for c in range(nc)])
    lon = np.array([cells[(r, c)][1] for r in range(nr) for c in range(nc)])
    mask = _inside_tz(lat, lon).reshape(nr, nc)
    _BOUNDARY_CACHE["cellmask"] = mask
    return mask


def _grid_to_array(df, value_col):
    nr = int(np.ceil((BBOX["lat_max"] - BBOX["lat_min"]) / GRID_RES_DEG))
    nc = int(np.ceil((BBOX["lon_max"] - BBOX["lon_min"]) / GRID_RES_DEG))
    arr = np.full((nr, nc), np.nan)
    for _, row in df.iterrows():
        rr, cc = int(row["_r"]), int(row["_c"])
        if 0 <= rr < nr and 0 <= cc < nc:
            arr[rr, cc] = row[value_col]
    arr[~_incountry_cell_mask()] = np.nan       # blank cells outside Tanzania
    return arr


def _draw_boundary(ax):
    """Overlay the Tanzania outline on a map axis, if available."""
    geom = _tz_boundary()
    if geom is None:
        return
    import geopandas as gpd
    gpd.GeoSeries([geom], crs="EPSG:4326").boundary.plot(
        ax=ax, color="black", linewidth=0.6)


def make_figures(richness_df, compare_df, fd_df, fd_resid_df, fr_df,
                 auc_tab, outdir=OUTDIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    extent = [BBOX["lon_min"], BBOX["lon_max"], BBOX["lat_min"], BBOX["lat_max"]]

    def _map(ax, df, col, title, cmap="viridis", clip=True):
        arr = _grid_to_array(df, col)
        if clip and np.isfinite(arr).any():           # 98th-pct clip (Neil's note)
            vmax = np.nanpercentile(arr, 98)
        else:
            vmax = None
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmax=vmax)
        _draw_boundary(ax)
        ax.set_xlim(BBOX["lon_min"], BBOX["lon_max"])
        ax.set_ylim(BBOX["lat_min"], BBOX["lat_max"])
        ax.set_title(title, fontsize=9); plt.colorbar(im, ax=ax, shrink=0.8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    _map(axes[0, 0], richness_df, "richness", "Raw species richness (98% clip)")
    _map(axes[0, 1], compare_df, "pred_richness", "SDM-predicted richness")
    _map(axes[0, 2], compare_df, "resid", "Predicted - raw (residual)",
         cmap="RdBu_r", clip=False)
    _map(axes[1, 0], fd_df, "FDis", "Functional dispersion (FDis)")
    _map(axes[1, 1], fd_resid_df, "fd_resid", "FD residual vs richness",
         cmap="RdBu_r", clip=False)
    _map(axes[1, 2], fr_df, "FR_norm", "Functional redundancy (normalised)",
         cmap="magma")
    for ax in axes.ravel():
        ax.set_xlabel("lon", fontsize=7); ax.set_ylabel("lat", fontsize=7)
    fig.suptitle("Pilot pipeline — 224 complete-trait species", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/pipeline_maps.png", dpi=130)
    plt.close(fig)

    # scatter: richness vs FDis, and AUC histogram
    fig2, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    d = fd_df.dropna(subset=["FDis", "richness"])
    a.scatter(d.richness, d.FDis, s=12, alpha=0.6)
    a.set_xlabel("species richness"); a.set_ylabel("FDis")
    a.set_title("FD vs richness")
    if auc_tab.auc.notna().any():
        b.hist(auc_tab.auc.dropna(), bins=15, color="steelblue")
        b.axvline(0.5, color="k", ls="--", lw=1)
        b.set_xlabel("spatial-block CV AUC"); b.set_title("SDM performance")
    fig2.tight_layout(); fig2.savefig(f"{outdir}/diagnostics.png", dpi=130)
    plt.close(fig2)
    print(f"[plot] wrote {outdir}/pipeline_maps.png and diagnostics.png")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main(max_species_sdm=None):
    os.makedirs(OUTDIR, exist_ok=True)
    data = load_data()

    richness_df = stage1_richness(data)
    sdm_out = stage2_sdms(data, max_species=max_species_sdm)
    compare_df, _ = stage3_compare(sdm_out, richness_df)
    fd_df = stage4_funcdiv(data, richness_df)
    fd_resid_df, _ = stage5_fd_vs_richness(fd_df)
    fr_df = stage6_redundancy(data, richness_df)

    # persist tables
    richness_df.to_csv(f"{OUTDIR}/stage1_richness.csv", index=False)
    sdm_out["auc"].to_csv(f"{OUTDIR}/stage2_sdm_auc.csv", index=False)
    compare_df.to_csv(f"{OUTDIR}/stage3_richness_compare.csv", index=False)
    fd_df.to_csv(f"{OUTDIR}/stage4_funcdiv.csv", index=False)
    fd_resid_df.to_csv(f"{OUTDIR}/stage5_fd_vs_richness.csv", index=False)
    fr_df.to_csv(f"{OUTDIR}/stage6_redundancy.csv", index=False)

    make_figures(richness_df, compare_df, fd_df, fd_resid_df, fr_df,
                 sdm_out["auc"])
    print("\n[done] All six stages complete. Outputs in ./%s/" % OUTDIR)
    return dict(data=data, richness=richness_df, sdm=sdm_out,
                compare=compare_df, fd=fd_df, fr=fr_df)


if __name__ == "__main__":
    main()
