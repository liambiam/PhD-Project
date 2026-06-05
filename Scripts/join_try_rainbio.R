# =====================================================================
# Join TRY traits (SLA / plant height / seed mass + variants) onto
# RAINBIO Tanzania occurrence points -> a single CSV of occurrences
# with trait columns appended.
#
# The TRY file is too big to load whole, so we stream it in chunks,
# keep only trait rows, and aggregate to one value per species per
# trait BEFORE joining (the aggregated table is small).
#
# NOTE on names: this does a DIRECT name join (RAINBIO 'species' ==
# TRY 'AccSpeciesName'). That UNDERSTATES coverage because synonyms /
# spelling differences won't match. Fine for a first look; the proper
# version resolves both sides to WCVP first (as in the checklist work).
# =====================================================================

library(readr)
library(dplyr)
library(tidyr)

# ---- CONFIG ---------------------------------------------------------
try_path    <- "C:/Users/liams/Documents/PhD-Project Data/TRY/SLA, PH, SM/50190.txt"
rainbio_csv <- "C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat_labels.csv"
outdir      <- "C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined"
chunk_size  <- 500000
error_threshold <- 4   # drop TRY rows with ErrorRisk >= this (NA kept)
use_stat    <- "median"  # "median" (robust) or "mean" per species per trait
# --------------------------------------------------------------------

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)


# ---- STAGE 1+2: stream TRY, keep trait rows, aggregate -------------
# We do filtering inside the chunk callback so we never hold the whole
# file. Each chunk returns only its trait rows (small), then we
# aggregate the accumulated trait rows at the end.
message("Streaming TRY file and extracting trait rows...")

collect_trait_rows <- function(chunk, pos) {
  chunk |>
    filter(!is.na(TraitID), !is.na(StdValue)) |>
    select(AccSpeciesName, TraitID, TraitName, StdValue, ErrorRisk)
}

trait_rows <- read_tsv_chunked(
  try_path,
  callback   = DataFrameCallback$new(collect_trait_rows),
  chunk_size = chunk_size,
  locale     = locale(encoding = "latin1"),
  show_col_types = FALSE,
  progress   = TRUE
)

message(sprintf("  kept %s trait rows across %s species",
                format(nrow(trait_rows), big.mark = ","),
                format(n_distinct(trait_rows$AccSpeciesName), big.mark = ",")))

# drop high-error-risk measurements
trait_rows <- trait_rows |>
  mutate(ErrorRisk = suppressWarnings(as.numeric(ErrorRisk))) |>
  filter(is.na(ErrorRisk) | ErrorRisk < error_threshold)

# aggregate to one value per species per trait
agg <- trait_rows |>
  group_by(AccSpeciesName, TraitName) |>
  summarise(
    value = if (use_stat == "median") median(StdValue, na.rm = TRUE)
            else mean(StdValue, na.rm = TRUE),
    n_measurements = n(),
    .groups = "drop"
  )

# pivot to wide: one row per species, one column per trait
traits_wide <- agg |>
  select(AccSpeciesName, TraitName, value) |>
  pivot_wider(names_from = TraitName, values_from = value)

# also keep a measurement-count per trait? (optional, commented)
# counts_wide <- agg |> select(AccSpeciesName, TraitName, n_measurements) |>
#   pivot_wider(names_from = TraitName, values_from = n_measurements,
#               names_prefix = "n_")

message(sprintf("  aggregated to %s species x %s trait columns",
                format(nrow(traits_wide), big.mark = ","),
                ncol(traits_wide) - 1))

# save the species x trait table on its own (useful later)
write_csv(traits_wide, file.path(outdir, "try_species_traits_wide.csv"))


# ---- STAGE 3: join onto RAINBIO occurrences ------------------------
message("Joining onto RAINBIO occurrences...")
rb <- read_csv(rainbio_csv, show_col_types = FALSE) |>
  mutate(species = trimws(species))

joined <- rb |>
  left_join(traits_wide, by = c("species" = "AccSpeciesName"))

# how many occurrence rows now have each trait?
trait_cols <- setdiff(names(traits_wide), "AccSpeciesName")
message("\n--- Coverage: RAINBIO OCCURRENCE rows with a trait value ---")
for (tc in trait_cols) {
  n <- sum(!is.na(joined[[tc]]))
  message(sprintf("  %-70s %s rows (%.1f%%)",
                  substr(tc, 1, 70), format(n, big.mark = ","),
                  100 * n / nrow(joined)))
}

# how many unique SPECIES matched?
rb_species   <- unique(rb$species)
matched_spp  <- intersect(rb_species, traits_wide$AccSpeciesName)
message(sprintf("\n--- Coverage: RAINBIO SPECIES matched to >=1 trait ---"))
message(sprintf("  %s of %s species (%.1f%%)",
                format(length(matched_spp), big.mark = ","),
                format(length(rb_species), big.mark = ","),
                100 * length(matched_spp) / length(rb_species)))

# write the main output
out_path <- file.path(outdir, "rainbio_with_try_traits.csv")
write_csv(joined, out_path)
message(sprintf("\nWritten:\n  %s  (occurrences + trait columns)\n  %s  (species x trait table)",
                out_path, file.path(outdir, "try_species_traits_wide.csv")))
