# =====================================================================
# REVERSE reconciliation: which KEW (accepted) species are ABSENT from
# RAINBIO?  i.e. the completeness gap in RAINBIO relative to the
# accepted Tanzanian flora.
#
# Mirror of the earlier RAINBIO->Kew funnel, run the other direction.
# Both lists are resolved to WCVP ACCEPTED names first, so that species
# present in RAINBIO under a synonym are NOT falsely counted as absent.
#
# Output: Kew-accepted species with no RAINBIO equivalent, broken down
# by family and (where derivable) growth form, written to CSV + the
# inputs for the Excel step.
#
# Run AFTER the main reconciliation script if you want to reuse its
# `resolved` object for RAINBIO; otherwise this resolves both sides.
# =====================================================================

library(rWCVP)
library(readr)
library(readxl)
library(dplyr)
library(stringr)
library(tidyr)

# ---- CONFIG ---------------------------------------------------------
rainbio_csv <- "C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat_labels.csv"
kew_xlsx    <- "C:/Users/liams/Documents/PhD-Project Data/Kew/List of species from Tanzania.xlsx"
outdir      <- "C:/Users/liams/Documents/PhD-Project Data/Reconciliation"
# --------------------------------------------------------------------
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

clean_name  <- function(x) {
  x <- str_squish(x)
  x <- str_replace_all(x, "\\b(cf|aff)\\.?\\s+", "")
  x <- str_replace_all(x, "\u00d7", "x")
  x
}
to_binomial <- function(x) word(x, 1, 2)

# ---- load -----------------------------------------------------------
message("Loading sources...")
rb <- read.csv(rainbio_csv, stringsAsFactors = FALSE)
kw <- read_excel(kew_xlsx)

rb_names <- rb$species   |> clean_name() |> unique()
rb_names <- rb_names[!is.na(rb_names) & rb_names != ""]
kw_names <- kw$taxon_name |> clean_name() |> unique()
kw_names <- kw_names[!is.na(kw_names) & kw_names != ""]
message(sprintf("  RAINBIO: %d names | Kew: %d names", length(rb_names), length(kw_names)))

# ---- helper: resolve a vector of names to WCVP accepted names -------
resolve_to_accepted <- function(names_vec) {
  m <- wcvp_match_names(data.frame(scientificname = names_vec),
                        name_col = "scientificname",
                        fuzzy = TRUE, progress_bar = TRUE)
  # robustly find the accepted-id column, then join to WCVP names table
  acc_id_col <- intersect(c("wcvp_accepted_id", "accepted_plant_name_id",
                            "wcvp_accepted_plant_name_id"), names(m))[1]
  wcvp_tbl <- rWCVPdata::wcvp_names
  id_to_name <- wcvp_tbl |>
    select(plant_name_id, taxon_name, family) |>
    rename(accepted_id = plant_name_id,
           accepted_name = taxon_name,
           accepted_family = family)
  m |>
    mutate(accepted_id = .data[[acc_id_col]]) |>
    left_join(id_to_name, by = "accepted_id") |>
    mutate(accepted_name = coalesce(accepted_name, scientificname)) |>
    select(input_name = scientificname, accepted_name, accepted_family)
}

# ---- resolve both lists --------------------------------------------
message("\nResolving RAINBIO names to WCVP accepted...")
rb_res <- resolve_to_accepted(rb_names)
message("Resolving Kew names to WCVP accepted...")
kw_res <- resolve_to_accepted(kw_names)

rb_accepted <- unique(na.omit(rb_res$accepted_name))
kw_accepted <- unique(na.omit(kw_res$accepted_name))
message(sprintf("\n  RAINBIO accepted species: %d", length(rb_accepted)))
message(sprintf("  Kew accepted species    : %d", length(kw_accepted)))

# ---- THE REVERSE SET: in Kew, not in RAINBIO ------------------------
kew_only <- setdiff(kw_accepted, rb_accepted)
message(sprintf("\n  Kew-accepted species ABSENT from RAINBIO: %d", length(kew_only)))

# attach family + original Kew metadata (lifeform/growth form if present)
kew_only_df <- tibble(accepted_name = kew_only) |>
  left_join(kw_res |> distinct(accepted_name, accepted_family),
            by = "accepted_name") |>
  # bring across Kew's own lifeform description if the column exists
  left_join(
    kw |>
      mutate(taxon_name = clean_name(taxon_name)) |>
      select(any_of(c("taxon_name", "lifeform_description",
                      "climate_description"))) |>
      rename(kew_name = taxon_name),
    by = c("accepted_name" = "kew_name")
  ) |>
  mutate(genus = word(accepted_name, 1)) |>
  arrange(accepted_family, accepted_name)

write.csv(kew_only_df, file.path(outdir, "kew_species_absent_from_rainbio.csv"),
          row.names = FALSE)

# ---- breakdown by family -------------------------------------------
by_family <- kew_only_df |>
  count(accepted_family, name = "n_missing") |>
  arrange(desc(n_missing))
write.csv(by_family, file.path(outdir, "kew_absent_by_family.csv"), row.names = FALSE)

# ---- breakdown by growth form (from Kew lifeform_description, if any)
by_growth <- NULL
if ("lifeform_description" %in% names(kew_only_df)) {
  by_growth <- kew_only_df |>
    mutate(lifeform_description = ifelse(is.na(lifeform_description),
                                         "(not given)", lifeform_description)) |>
    count(lifeform_description, name = "n_missing") |>
    arrange(desc(n_missing))
  write.csv(by_growth, file.path(outdir, "kew_absent_by_growthform.csv"),
            row.names = FALSE)
}

# ---- summary --------------------------------------------------------
message("\n========== REVERSE RECONCILIATION SUMMARY ==========")
message(sprintf("Kew accepted species           : %d", length(kw_accepted)))
message(sprintf("  also in RAINBIO              : %d", length(intersect(kw_accepted, rb_accepted))))
message(sprintf("  ABSENT from RAINBIO          : %d (%.1f%% of Kew flora)",
                length(kew_only), 100*length(kew_only)/length(kw_accepted)))
message("\nTop families missing from RAINBIO:")
print(head(by_family, 15))
if (!is.null(by_growth)) {
  message("\nMissing species by growth form (Kew lifeform):")
  print(head(by_growth, 15))
}
message(sprintf("\nWritten to: %s", outdir))
message("  kew_species_absent_from_rainbio.csv")
message("  kew_absent_by_family.csv")
if (!is.null(by_growth)) message("  kew_absent_by_growthform.csv")

# keep objects available for the Excel step
assign("kew_only_df", kew_only_df, envir = .GlobalEnv)
assign("by_family", by_family, envir = .GlobalEnv)
assign("by_growth", by_growth, envir = .GlobalEnv)
assign("rev_summary", tibble(
  metric = c("Kew accepted species", "Also in RAINBIO",
             "Absent from RAINBIO", "Percent of Kew flora missing"),
  value  = c(length(kw_accepted), length(intersect(kw_accepted, rb_accepted)),
             length(kew_only),
             round(100*length(kew_only)/length(kw_accepted), 1))
), envir = .GlobalEnv)
