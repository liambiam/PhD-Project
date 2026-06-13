# =========================================================
# TANZANIA FLORA MATCHING (ROBUST VERSION)
# =========================================================

library(readxl)
library(dplyr)
library(stringr)
library(tidystringdist)
library(stringdist)
library(taxize)   # optional but powerful

# =========================================================
# FILES
# =========================================================

kew_file <- "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/List of species from Tanzania.xlsx"
rainbio_file <- "C:/Users/liams/Documents/PhD-Project Data/RAINBIO/tanzania_points.csv"

# =========================================================
# LOAD
# =========================================================

kew <- read_excel(kew_file)
rainbio <- read.csv(rainbio_file)

# =========================================================
# CLEAN FUNCTIONS
# =========================================================

clean <- function(x) {
  x %>%
    str_to_lower() %>%
    str_trim() %>%
    str_replace_all("\\s+", " ")
}

# strict binomial (Genus + species only)
binomial <- function(x) {
  clean(x) %>%
    word(1, 2)
}

# more robust cleaning (removes var/subsp + punctuation noise)
deep_clean <- function(x) {
  x %>%
    clean() %>%
    str_replace_all("\\b(var|subsp|ssp|forma|f\\.)\\b.*$", "") %>%
    str_replace_all("[^a-z ]", "") %>%
    str_squish() %>%
    word(1, 2)
}

# =========================================================
# CREATE DATASETS
# =========================================================

kew$bin <- binomial(kew$taxon_name)
rainbio$bin <- binomial(rainbio$species)

kew$deep <- deep_clean(kew$taxon_name)
rainbio$deep <- deep_clean(rainbio$species)

# unique lists
kew_bin <- unique(kew$bin)
rain_bin <- unique(rainbio$bin)

kew_deep <- unique(kew$deep)
rain_deep <- unique(rainbio$deep)

# =========================================================
# BASIC OVERLAP
# =========================================================

cat("\n====================\n")
cat("BASIC BINOMIAL MATCH\n")
cat("====================\n")

basic_shared <- intersect(kew_bin, rain_bin)

cat("Shared (binomial):", length(basic_shared), "\n")
cat("Kew only:", length(setdiff(kew_bin, rain_bin)), "\n")
cat("RAINBIO only:", length(setdiff(rain_bin, kew_bin)), "\n")

# =========================================================
# IMPROVED CLEAN MATCH
# =========================================================

cat("\n====================\n")
cat("DEEP CLEAN MATCH\n")
cat("====================\n")

deep_shared <- intersect(kew_deep, rain_deep)

cat("Shared (deep clean):", length(deep_shared), "\n")
cat("Kew only:", length(setdiff(kew_deep, rain_deep)), "\n")
cat("RAINBIO only:", length(setdiff(rain_deep, kew_deep)), "\n")

# =========================================================
# FUZZY MATCHING (SPELLING ERRORS)
# =========================================================

cat("\n====================\n")
cat("FUZZY MATCHING (STRING DISTANCE)\n")
cat("====================\n")

# sample matching (FULL cross is expensive; adjust if needed)
max_dist <- 1  # 1 = very strict, 2 = moderate, 3 = loose

fuzzy_matches <- stringdist_inner_join(
  data.frame(kew = kew_deep),
  data.frame(rainbio = rain_deep),
  by = c("kew" = "rainbio"),
  method = "lv",
  max_dist = max_dist,
  distance_col = "dist"
)

cat("Fuzzy matched pairs:", nrow(fuzzy_matches), "\n")

# extract unique matched species
fuzzy_shared <- unique(c(fuzzy_matches$kew, fuzzy_matches$rainbio))

cat("Unique species after fuzzy matching:", length(fuzzy_shared), "\n")

# =========================================================
# SYNONYM RESOLUTION (OPTIONAL - WCVP STYLE)
# =========================================================
# NOTE: this uses GBIF backbone; WCVP is better but harder locally

cat("\n====================\n")
cat("SYNONYM RESOLUTION (GBIF BACKBONE)\n")
cat("====================\n")

resolve_names <- function(x) {

  out <- tryCatch(
    taxize::gnr_resolve(x, best_match_only = TRUE),
    error = function(e) NULL
  )

  return(out)
}

# NOTE: only run on subset first (API heavy)
test_names <- head(kew_deep, 200)

resolved <- resolve_names(test_names)

if (!is.null(resolved)) {
  cat("Resolved sample size:", nrow(resolved), "\n")
  cat("Example resolved names:\n")
  print(head(resolved[, c("user_supplied_name", "matched_name")]))
} else {
  cat("Synonym resolution failed or skipped (API issue or rate limit)\n")
}

# =========================================================
# FINAL COMPARISON SUMMARY
# =========================================================

cat("\n====================\n")
cat("FINAL SUMMARY\n")
cat("====================\n")

summary <- data.frame(
  method = c(
    "raw binomial",
    "deep clean",
    "fuzzy match"
  ),
  shared_species = c(
    length(basic_shared),
    length(deep_shared),
    length(fuzzy_shared)
  )
)

print(summary)

# improvement estimate
cat("\nImprovement from raw to fuzzy:",
    length(fuzzy_shared) - length(basic_shared),
    "species\n")

cat("Done.\n")