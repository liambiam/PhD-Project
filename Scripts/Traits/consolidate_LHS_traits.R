# =====================================================================
# Consolidate the fragmented TRY trait columns in rainbio_with_try_traits.csv
# into the 3 LHS traits:  SLA, height, seed_mass
#
#   SLA       <- coalesce the 5 "specific leaf area" variants into one,
#                in priority order (petiole excluded > undefined >
#                included > other); drop the empty "of total area".
#   height    <- "Plant height vegetative"  (generative kept separately,
#                optional, since it's a different trait).
#   seed_mass <- "Seed dry mass"
#
# Reports per-trait coverage and how many species have ALL THREE.
# Works from the already-joined CSV, so no need to re-stream TRY.
# =====================================================================

library(readr)
library(dplyr)

# ---- CONFIG ---------------------------------------------------------
joined_csv <- "C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined/rainbio_with_try_traits.csv"
outdir     <- "C:/Users/liams/Documents/PhD-Project Data/TRY and RAINBIO joined"
species_col <- "species"
# --------------------------------------------------------------------

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(joined_csv, show_col_types = FALSE)

# ---- helper: find a column by a substring of its (long) TRY name ----
# returns the actual column name in df, or NA if not present
find_col <- function(df, pattern) {
  hits <- names(df)[grepl(pattern, names(df), ignore.case = TRUE, fixed = FALSE)]
  if (length(hits) == 0) return(NA_character_)
  hits
}

# Identify the SLA variant columns (all contain "specific leaf area").
# Order matters: we coalesce in PRIORITY order so the preferred
# definition wins when a species has more than one.
sla_excluded  <- find_col(df, "specific leaf area.*petiole excluded")
sla_undefined <- find_col(df, "specific leaf area.*undefined")
sla_included  <- find_col(df, "specific leaf area.*petiole included")
sla_other     <- find_col(df, "specific leaf area")   # catch-all remainder

# build an ordered, de-duplicated list of existing SLA columns
sla_priority <- unique(na.omit(c(sla_excluded, sla_undefined, sla_included, sla_other)))
sla_priority <- sla_priority[sla_priority %in% names(df)]
message("SLA columns found (priority order):")
for (c in sla_priority) message("   ", c)

height_veg <- find_col(df, "Plant height vegetative")
height_gen <- find_col(df, "Plant height generative")
seed_mass  <- find_col(df, "Seed dry mass")

# ---- consolidate ----------------------------------------------------
# coalesce SLA variants: first non-NA across the priority-ordered cols
if (length(sla_priority) > 0) {
  df$SLA <- do.call(dplyr::coalesce, df[ , sla_priority, drop = FALSE])
} else {
  df$SLA <- NA_real_
  warning("No SLA columns found!")
}

df$height    <- if (!is.na(height_veg[1])) df[[height_veg[1]]] else NA_real_
df$seed_mass <- if (!is.na(seed_mass[1]))  df[[seed_mass[1]]]  else NA_real_
# keep generative height separately (optional, not part of LHS)
df$height_generative <- if (!is.na(height_gen[1])) df[[height_gen[1]]] else NA_real_

# ---- build a clean output: original cols + 3 LHS traits -------------
# drop the messy long trait columns, keep everything else + the 3 clean ones
long_trait_cols <- unique(na.omit(c(sla_priority, height_veg, height_gen, seed_mass)))
clean <- df |>
  select(-any_of(long_trait_cols)) |>
  relocate(SLA, height, seed_mass, .after = last_col())

write_csv(clean, file.path(outdir, "rainbio_LHS_traits.csv"))

# also a compact species x 3-trait table (one row per species)
species_traits <- clean |>
  group_by(.data[[species_col]]) |>
  summarise(SLA       = first(na.omit(SLA)),
            height    = first(na.omit(height)),
            seed_mass = first(na.omit(seed_mass)),
            .groups = "drop")
write_csv(species_traits, file.path(outdir, "species_LHS_traits.csv"))


# ---- coverage report ------------------------------------------------
n_rows <- nrow(clean)
all_species <- unique(clean[[species_col]])
n_species <- length(all_species)

cov_row <- function(x) sprintf("%s rows (%.1f%%)",
                               format(sum(!is.na(x)), big.mark = ","),
                               100 * sum(!is.na(x)) / n_rows)

message("\n--- OCCURRENCE-ROW coverage (consolidated) ---")
message("  SLA       : ", cov_row(clean$SLA))
message("  height    : ", cov_row(clean$height))
message("  seed_mass : ", cov_row(clean$seed_mass))

# species-level coverage
sp_has <- species_traits |>
  mutate(has_SLA = !is.na(SLA), has_h = !is.na(height), has_sm = !is.na(seed_mass),
         n_traits = has_SLA + has_h + has_sm)

message("\n--- SPECIES coverage (consolidated) ---")
message(sprintf("  total RAINBIO species          : %s", format(n_species, big.mark = ",")))
message(sprintf("  with SLA                       : %s (%.1f%%)",
                format(sum(sp_has$has_SLA), big.mark = ","), 100*mean(sp_has$has_SLA)))
message(sprintf("  with height                    : %s (%.1f%%)",
                format(sum(sp_has$has_h), big.mark = ","), 100*mean(sp_has$has_h)))
message(sprintf("  with seed_mass                 : %s (%.1f%%)",
                format(sum(sp_has$has_sm), big.mark = ","), 100*mean(sp_has$has_sm)))
message(sprintf("  with at least ONE trait        : %s (%.1f%%)",
                format(sum(sp_has$n_traits >= 1), big.mark = ","), 100*mean(sp_has$n_traits >= 1)))
message(sprintf("  with ALL THREE LHS traits      : %s (%.1f%%)  <-- key number",
                format(sum(sp_has$n_traits == 3), big.mark = ","), 100*mean(sp_has$n_traits == 3)))

# breakdown by number of traits held
message("\n  species by number of traits held:")
print(as.data.frame(table(n_traits = sp_has$n_traits)))

message(sprintf("\nWritten:\n  %s  (occurrences + 3 LHS trait cols)\n  %s  (species x 3 traits)",
                file.path(outdir, "rainbio_LHS_traits.csv"),
                file.path(outdir, "species_LHS_traits.csv")))
