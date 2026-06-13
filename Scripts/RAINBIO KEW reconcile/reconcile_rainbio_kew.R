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
#       repos = c("https://matildabrown.github.io/drat", getOption("repos")))
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

# --- DIAGNOSTIC: print the actual column names this rWCVP version returns
message("\n[diagnostic] wcvp_match_names returned columns:")
print(names(matched))

# Resolve each match to its ACCEPTED name in a VERSION-ROBUST way:
# instead of guessing the accepted-name column, join the matched
# accepted_id to the full WCVP names table and read the accepted name there.
wcvp_tbl <- rWCVPdata::wcvp_names   # full WCVP checklist

# the matched accepted-id column is 'wcvp_accepted_id' in current versions;
# fall back gracefully if named differently
acc_id_col <- intersect(c("wcvp_accepted_id", "accepted_plant_name_id",
                          "wcvp_accepted_plant_name_id"), names(matched))[1]
status_col <- intersect(c("wcvp_status", "taxon_status"), names(matched))[1]
mtype_col  <- intersect(c("match_type", "wcvp_match_type"), names(matched))[1]
message(sprintf("[diagnostic] using accepted-id col = '%s', status col = '%s'",
                acc_id_col, status_col))

# lookup table: plant_name_id -> taxon_name (the accepted name)
id_to_name <- wcvp_tbl |>
  select(plant_name_id, taxon_name) |>
  rename(accepted_id = plant_name_id, accepted_name = taxon_name)

resolved <- matched |>
  mutate(accepted_id = .data[[acc_id_col]]) |>
  left_join(id_to_name, by = "accepted_id") |>
  mutate(
    match_type    = .data[[mtype_col]],
    wcvp_status   = .data[[status_col]],
    accepted_name = coalesce(accepted_name, .data[["scientificname"]])
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
  select(any_of(c("scientificname", "match_type", "wcvp_status",
                  "accepted_name", "wcvp_name", "multiple_matches"))) |>
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


# =====================================================================
# STEP 4 - build a REVIEW worksheet for the supervisor
# Set = everything NOT an exact match that resolved into Kew.
# One row per RAINBIO name, with evidence columns + a review_category
# so it can be sorted and triaged. No verdict column (evidence only).
# =====================================================================
message("\n--- STEP 4: building supervisor review CSV ---")

# record counts per RAINBIO name (how well-collected is it?)
rb_counts <- rb |>
  mutate(species = clean_name(species)) |>
  count(species, name = "rainbio_records")

# genus-in-Kew flag (reuse the Kew genus set from Step 1)
genus_in_kew <- function(x) word(x, 1) %in% kw_genus

# Start from the resolved table (Step 2 output for the still-unmatched
# names) and ALSO fold in the binomial-recovered and exact-shared names
# so we can classify the full RAINBIO list, then filter.
#
# Build a master per-name frame:
master <- tibble(species = rb_names) |>
  mutate(
    exact_shared        = species %in% shared_raw,
    recovered_binomial  = species %in% recovered_by_binomial
  )

# bring in WCVP match info for the names that went through Step 2
res_small <- resolved |>
  transmute(
    species        = scientificname,
    wcvp_match     = match_type,
    wcvp_status    = wcvp_status,
    accepted_name  = accepted_name,
    accepted_in_kew = accepted_in_kew
  )

master <- master |>
  left_join(res_small, by = "species") |>
  left_join(rb_counts, by = "species") |>
  mutate(
    genus           = word(species, 1),
    genus_present_in_kew = genus_in_kew(species),
    rainbio_records = ifelse(is.na(rainbio_records), 0L, rainbio_records)
  )

# ---- classify every name into a review_category ----
master <- master |>
  mutate(review_category = case_when(
    exact_shared ~ "exact_match_in_kew",
    recovered_binomial ~ "matched_after_binomial_strip",
    accepted_in_kew & wcvp_match == "Exact (without author)" ~ "exact_resolved_into_kew",
    accepted_in_kew & str_detect(coalesce(wcvp_match, ""), regex("fuzzy", ignore_case = TRUE)) ~ "fuzzy_resolved_into_kew_CHECK",
    accepted_in_kew ~ "resolved_into_kew_other",
    !is.na(wcvp_match) & str_detect(coalesce(wcvp_match, ""), regex("fuzzy", ignore_case = TRUE)) ~ "fuzzy_match_NOT_in_kew_CHECK",
    !is.na(accepted_name) & !accepted_in_kew ~ "matched_wcvp_but_absent_from_kew",
    is.na(wcvp_match) | wcvp_match == "No match" ~ "unmatched_no_wcvp_record",
    TRUE ~ "other"
  ))

# ---- the review set: everything NOT exact-and-in-kew ----
review <- master |>
  filter(!(review_category %in% c("exact_match_in_kew",
                                  "exact_resolved_into_kew"))) |>
  arrange(review_category, desc(rainbio_records)) |>
  select(species, review_category, wcvp_match, wcvp_status,
         accepted_name, accepted_in_kew,
         genus, genus_present_in_kew, rainbio_records)

write.csv(review, file.path(outdir, "step4_review_for_supervisor.csv"),
          row.names = FALSE)

message(sprintf("Review set: %d names (of %d RAINBIO names)",
                nrow(review), length(rb_names)))
message("Breakdown by category:")
print(as.data.frame(table(review$review_category)))
message(sprintf("\nWritten: %s", file.path(outdir, "step4_review_for_supervisor.csv")))

# =====================================================================
# Build SUPERVISOR-FRIENDLY outputs from the RAINBIO/Kew reconciliation.
#
# Run this AFTER the main reconciliation script (it uses `master`,
# `rb`, `rb_names`, `shared_raw`, `recovered_by_binomial`, `outdir`,
# and `clean_name` from that script's environment).
#
# Produces:
#   1. reconciliation_summary.csv   - the one-page funnel (counts)
#   2. review_distribution_questions.csv - the 854 "in WCVP, not in Kew"
#                                          (the interesting ones)
#   3. review_matching_errors.csv   - the 52 fuzzy-not-in-Kew (discard pile)
#   4. review_ALL.csv               - both review sets combined, tidy
#
# Multiple matches are COLLAPSED to one row per species (best candidate),
# with a flag noting alternatives existed. Plain-English column names.
# =====================================================================

library(dplyr)
library(stringr)

# ---- helper: epithet (second word) for the "did epithet survive?" flag
epithet <- function(x) word(x, 2)

# ---- collapse multiple matches to ONE row per species ----------------
# Priority: prefer Accepted status, then an epithet-preserving match,
# then highest plausibility. Keep a flag if alternatives existed.
status_rank <- c("Accepted" = 1, "Synonym" = 2, "Unplaced" = 3,
                 "Illegitimate" = 4, "Invalid" = 5, "Misapplied" = 6)

collapse_best <- function(df) {
  df |>
    mutate(
      .status_rank = coalesce(status_rank[wcvp_status], 9L),
      .epithet_kept = epithet(species) == epithet(accepted_name)
    ) |>
    arrange(species, desc(.epithet_kept), .status_rank) |>
    group_by(species) |>
    mutate(n_candidates = n()) |>
    slice(1) |>                     # keep best candidate per species
    ungroup() |>
    select(-.status_rank, -.epithet_kept)
}

# ---- pull the two residual categories from master --------------------
not_in_kew <- master |>
  filter(review_category == "matched_wcvp_but_absent_from_kew") |>
  collapse_best()

fuzzy_bad <- master |>
  filter(review_category == "fuzzy_match_NOT_in_kew_CHECK") |>
  collapse_best()

# ---- tidy, plain-English columns -------------------------------------
tidy_cols <- function(df, note) {
  df |>
    mutate(
      epithet_preserved = epithet(species) == epithet(accepted_name),
      n_alternatives = pmax(n_candidates - 1, 0)
    ) |>
    transmute(
      rainbio_name      = species,
      genus,
      rainbio_records,
      wcvp_suggests     = accepted_name,
      wcvp_status,
      match_quality     = wcvp_match,
      epithet_preserved,           # TRUE = likely real; FALSE = suspicious
      n_alternatives,              # >0 means WCVP returned several guesses
      note = note
    ) |>
    arrange(desc(rainbio_records))
}

dist_q <- tidy_cols(
  not_in_kew,
  "Accepted in WCVP but not in our Kew TZ list - is it a real Tanzanian species?"
)
err <- tidy_cols(
  fuzzy_bad,
  "Fuzzy match that did not resolve into Kew - likely a matching error to discard"
)

write.csv(dist_q, file.path(outdir, "review_distribution_questions.csv"), row.names = FALSE)
write.csv(err,    file.path(outdir, "review_matching_errors.csv"), row.names = FALSE)

# combined, with a clear group label
review_all <- bind_rows(
  dist_q |> mutate(group = "distribution_question"),
  err    |> mutate(group = "likely_matching_error")
) |>
  relocate(group)
write.csv(review_all, file.path(outdir, "review_ALL.csv"), row.names = FALSE)

# ---- one-page summary funnel -----------------------------------------
# distinct-name counts so it sums to the RAINBIO total
cat_counts <- master |>
  distinct(species, .keep_all = TRUE) |>
  count(review_category)

get_n <- function(cat) {
  v <- cat_counts$n[cat_counts$review_category == cat]
  if (length(v)) v else 0L
}

summary_tbl <- tibble::tibble(
  stage = c(
    "TOTAL RAINBIO species",
    "Exact match to Kew",
    "Matched after binomial (rank) strip",
    "Synonym/exact resolved into Kew (WCVP)",
    "Fuzzy matched into Kew (WCVP)",
    "Matched WCVP but NOT in Kew (review)",
    "Fuzzy match NOT into Kew (review)"
  ),
  species = c(
    length(rb_names),
    get_n("exact_match_in_kew"),
    get_n("matched_after_binomial_strip"),
    get_n("exact_resolved_into_kew"),
    get_n("fuzzy_resolved_into_kew_CHECK"),
    nrow(dist_q),     # collapsed distinct count
    nrow(err)         # collapsed distinct count
  )
)
reconciled <- sum(summary_tbl$species[2:5])
summary_tbl <- summary_tbl |>
  mutate(pct_of_total = round(100 * species / length(rb_names), 1))

write.csv(summary_tbl, file.path(outdir, "reconciliation_summary.csv"), row.names = FALSE)

# ---- console report --------------------------------------------------
message("\n========== RECONCILIATION SUMMARY ==========")
print(as.data.frame(summary_tbl), row.names = FALSE)
message(sprintf("\nReconciled with Kew (any route): %d of %d (%.1f%%)",
                reconciled, length(rb_names),
                100 * reconciled / length(rb_names)))
message(sprintf("Genuine residual for review     : %d", nrow(dist_q) + nrow(err)))
message(sprintf("  - distribution questions: %d", nrow(dist_q)))
message(sprintf("  - likely matching errors: %d", nrow(err)))
message("\nFiles written to: ", outdir)
message("  reconciliation_summary.csv      (the funnel)")
message("  review_distribution_questions.csv  (the 'is it really in TZ?' set)")
message("  review_matching_errors.csv      (the discard pile)")
message("  review_ALL.csv                  (both, combined)")