# =====================================================================
# Reconciling RAINBIO vs Kew Tanzanian plant checklists
#
# Strategy (cheap -> expensive):
#   1. Mechanical diagnostic: how much of the "difference" is just
#      formatting / infraspecific rank / genus actually present?
#      (no matching needed - resolves a big chunk for free)
#   2. WCVP matching (rWCVP): resolve remaining RAINBIO names against
#      Kew's own authority (World Checklist of Vascular Plants) so
#      synonyms collapse to accepted names.
#   3. Genuine residual: names that are truly Kew-absent after matching
#      -> small enough to eyeball. Written out for manual inspection.
#
# Kew is treated as the authority (your Kew list is from POWO/WCVP).
# RAINBIO is occurrence-derived and frozen ~2015, so most disagreement
# is expected to be RAINBIO carrying names WCVP has since synonymised.
# =====================================================================

# ---- INSTALL (one-time) --------------------------------------------
# rWCVP is on CRAN; its DATA lives in rWCVPdata on a Kew R-universe repo:
#
#   install.packages("rWCVP")
#   install.packages("rWCVPdata", 
#   repos = c("https://matildabrown.github.io/drat", getOption("repos")))
#
# (rWCVPdata is large; first install downloads the full checklist.)
# --------------------------------------------------------------------

library(rWCVP)
library(readxl)
library(dplyr)
library(stringr)
library(tidyr)

# ---- CONFIG ---------------------------------------------------------
rainbio_csv <- "C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat_labels.csv"
kew_xlsx    <- "C:/Users/liams/Documents/PhD-Project Data/Kew/List of species from Tanzania.xlsx"
outdir      <- "C:/Users/liams/Documents/PhD-Project Data/Reconciliation"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Tanzania WGSRPD level-3 area code, used to ask WCVP "is this species
# actually accepted as occurring in Tanzania?"
TZ_CODE <- "TAN"
# --------------------------------------------------------------------


# ---- light name cleaner --------------------------------------------
clean_name <- function(x) {
  x <- str_squish(x)                          # collapse whitespace
  x <- str_replace_all(x, "\\b(cf|aff)\\.?\\s+", "")  # drop cf./aff.
  x <- str_replace_all(x, "\u00d7", "x")      # hybrid marker
  x
}

# strip to bare binomial (genus + species epithet), dropping infraspecific
to_binomial <- function(x) {
  word(x, 1, 2)
}


# ---- load -----------------------------------------------------------
message("Loading sources...")
rb <- read.csv(rainbio_csv, stringsAsFactors = FALSE)
kw <- read_excel(kew_xlsx)

rb_names <- rb$species |> clean_name() |> unique()
rb_names <- rb_names[!is.na(rb_names) & rb_names != ""]

kw_names <- kw$taxon_name |> clean_name() |> unique()
kw_names <- kw_names[!is.na(kw_names) & kw_names != ""]

message(sprintf("  RAINBIO: %d unique names", length(rb_names)))
message(sprintf("  Kew    : %d unique names", length(kw_names)))


# =====================================================================
# STEP 1 - mechanical diagnostic (no matching)
# =====================================================================
message("\n--- STEP 1: mechanical diagnostic ---")

# 1a. exact raw overlap
shared_raw  <- intersect(rb_names, kw_names)
rb_only_raw <- setdiff(rb_names, kw_names)
message(sprintf("Exact-string shared: %d", length(shared_raw)))
message(sprintf("RAINBIO-only (raw) : %d   <-- this is the discrepancy to explain",
                length(rb_only_raw)))

# 1b. of the RAINBIO-only names, how many match Kew once reduced to binomial?
rb_only_binom <- to_binomial(rb_only_raw)
kw_binom      <- unique(to_binomial(kw_names))
recovered_by_binomial <- rb_only_raw[rb_only_binom %in% kw_binom]
message(sprintf("  ...of which recovered by stripping to binomial: %d",
                length(recovered_by_binomial)))

# 1c. of the STILL-unmatched, how many at least share a GENUS with Kew?
still_unmatched <- setdiff(rb_only_raw, recovered_by_binomial)
rb_genus <- word(still_unmatched, 1)
kw_genus <- unique(word(kw_names, 1))
same_genus  <- still_unmatched[rb_genus %in% kw_genus]   # likely synonym/reclass
diff_genus  <- still_unmatched[!(rb_genus %in% kw_genus)] # genus absent entirely
message(sprintf("  ...still unmatched: %d", length(still_unmatched)))
message(sprintf("       genus present in Kew (likely synonym/reclass): %d", length(same_genus)))
message(sprintf("       genus absent from Kew (real gap or odd name)  : %d", length(diff_genus)))

# save the step-1 breakdown
write.csv(data.frame(rainbio_only_raw = rb_only_raw),
          file.path(outdir, "step1_rainbio_only_raw.csv"), row.names = FALSE)


# =====================================================================
# STEP 2 - WCVP matching of the still-unmatched names
# =====================================================================
message("\n--- STEP 2: WCVP matching (resolving synonyms to accepted) ---")

to_match <- data.frame(scientificname = still_unmatched, stringsAsFactors = FALSE)

# wcvp_match_names: fuzzy + author-aware matching against WCVP
matched <- wcvp_match_names(
  to_match,
  name_col   = "scientificname",
  fuzzy      = TRUE,
  progress_bar = TRUE
)

# resolve each match to its ACCEPTED name (follow synonyms)
# wcvp_accepted_id / the accepted columns give the accepted taxon
resolved <- matched |>
  mutate(
    accepted_name = coalesce(wcvp_accepted_name, wcvp_name),
    is_synonym    = wcvp_status == "Synonym"
  )

# Does the ACCEPTED name now exist in the Kew list?
resolved <- resolved |>
  mutate(accepted_in_kew = accepted_name %in% kw_names |
                           to_binomial(accepted_name) %in% kw_binom)

n_resolved_into_kew <- sum(resolved$accepted_in_kew, na.rm = TRUE)
message(sprintf("WCVP resolved into a Kew-accepted name: %d", n_resolved_into_kew))

write.csv(resolved, file.path(outdir, "step2_wcvp_matched.csv"), row.names = FALSE)


# =====================================================================
# STEP 3 - genuine residual (eyeball this)
# =====================================================================
message("\n--- STEP 3: genuine residual ---")

residual <- resolved |>
  filter(!accepted_in_kew | is.na(accepted_name)) |>
  select(scientificname, match_type = match_type,
         wcvp_name, wcvp_status, accepted_name,
         multiple_matches) |>
  arrange(match_type)

message(sprintf("Genuine residual (RAINBIO names with no Kew-accepted equivalent): %d",
                nrow(residual)))
message("  Categorise by match_type column:")
print(table(residual$match_type, useNA = "ifany"))

write.csv(residual, file.path(outdir, "step3_genuine_residual.csv"), row.names = FALSE)


# =====================================================================
# SUMMARY - the funnel from 3000 -> real difference
# =====================================================================
message("\n========== RECONCILIATION FUNNEL ==========")
message(sprintf("RAINBIO unique names              : %d", length(rb_names)))
message(sprintf("  exact match to Kew              : %d", length(shared_raw)))
message(sprintf("  RAINBIO-only (the discrepancy)  : %d", length(rb_only_raw)))
message(sprintf("    recovered by binomial strip   : %d", length(recovered_by_binomial)))
message(sprintf("    resolved to Kew via WCVP synonym: %d", n_resolved_into_kew))
message(sprintf("    GENUINE residual              : %d", nrow(residual)))
message("===========================================")
message(sprintf("\nFiles written to: %s", outdir))
message("  step1_rainbio_only_raw.csv, step2_wcvp_matched.csv, step3_genuine_residual.csv")
