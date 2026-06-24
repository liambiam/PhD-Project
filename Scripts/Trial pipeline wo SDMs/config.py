"""
Configuration for the functional redundancy pipeline (trial version).

All paths, grid settings, scenario choices and analysis thresholds live here.
Scripts 01–05 import from this file and nothing else for configuration.

Anything you might want to vary between runs belongs here. Anything hard-coded
in the scripts themselves is a bug — flag it and lift it up.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the PhD data directory (Windows path style retained for Liam's setup)
DATA_ROOT = Path("C:/Users/liams/Documents/PhD-Project Data")

# Inputs
RAINBIO_TRAITS_CSV = DATA_ROOT / "TRY and RAINBIO joined" / "rainbio_LHS_traits.csv"
WORLDCLIM_PRESENT_DIR = DATA_ROOT / "Worldclim climate data" / "wc2.1_10m"
WORLDCLIM_FUTURE_DIR = DATA_ROOT / "Worldclim climate data" / "future"  # placeholder
TANZANIA_SHP = DATA_ROOT / "GADM Tanzania" / "gadm41_TZA_0.shp"

# Outputs
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

# Resolution in degrees. 0.5 matches the pilot; drop to 0.25 or 0.1 later
# once SDM surfaces replace raw rasterisation.
GRID_RES_DEG = 0.25

# Tanzania bounding box (decimal degrees, WGS84). Used to define the grid.
# Slightly padded beyond GADM extent.
TANZANIA_BBOX = {
    "min_lon": 29.0,
    "max_lon": 41.0,
    "min_lat": -12.0,
    "max_lat": -0.5,
}

# Geographic extent used to derive each species' climatic envelope.
# Wider than Tanzania because RAINBIO is continental and envelopes
# should reflect the species' full realised niche, not a Tanzanian slice.
# "africa" uses all RAINBIO records; "east_africa" uses a regional bbox;
# "tanzania" restricts to TZ only (not recommended — envelopes too narrow).
ENVELOPE_EXTENT = "africa"

EAST_AFRICA_BBOX = {
    "min_lon": 28.0,
    "max_lon": 42.0,
    "min_lat": -12.0,
    "max_lat": 5.0,
}

# ---------------------------------------------------------------------------
# Trait & assemblage settings
# ---------------------------------------------------------------------------

# Traits used for the LHS scheme. Column names in rainbio_LHS_traits.csv.
TRAIT_COLUMNS = ["height", "SLA", "seed_mass"]

# Only keep species with values for all of these traits.
REQUIRE_COMPLETE_TRAITS = True

# Minimum RAINBIO records per species to enter the reliable pool.
MIN_RECORDS_PER_SPECIES = 5

# Minimum species per cell for FR to be reported. Below this, Rao's Q is
# too noisy to mean anything; cells get NaN rather than a misleading value.
MIN_SPECIES_PER_CELL = 4

# ---------------------------------------------------------------------------
# Niche envelope percentiles
# ---------------------------------------------------------------------------

# Following Martín-Forés et al. 2026 (Australia paper).
MAT_TOLERANCE_PERCENTILE = 98  # upper thermal limit
MAP_TOLERANCE_PERCENTILE = 2   # lower precipitation limit

# ---------------------------------------------------------------------------
# Future climate scenario
# ---------------------------------------------------------------------------

# Placeholder values — Stage 4 will read these. Fill in once you have a
# specific future WorldClim file in place.
FUTURE_SCENARIO = "ssp370"
FUTURE_HORIZON = "2061-2080"
FUTURE_GCM = "ACCESS-CM2"

# ---------------------------------------------------------------------------
# Reassembly settings
# ---------------------------------------------------------------------------

# Dispersal rule for the gain step in Stage 4.
# "unconstrained" — any climatically suitable cell can gain any species.
# "distance_kernel" — not implemented yet; placeholder for v2.
# "range_adjacent" — not implemented yet; placeholder for v2.
DISPERSAL_RULE = "unconstrained"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
VERBOSE = True