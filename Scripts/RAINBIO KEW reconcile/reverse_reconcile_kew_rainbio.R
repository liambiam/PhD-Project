# =====================================================================
# REVERSE reconciliation: KEW vs RAINBIO Tanzanian plant checklists
#
# Which KEW species fail to reach RAINBIO?  i.e. RAINBIO's collection
# gap relative to the accepted Tanzanian flora.
#
# IMPORTANT FIX vs the earlier version:
#   The comparison is now ACCEPTED-vs-ACCEPTED. BOTH lists are resolved
#   to WCVP accepted names before comparison. The previous version
#   resolved only Kew and compared against RAINBIO's RAW (synonym-laden,
#   frozen-2015) names, which falsely pushed species into "absent" when
#   they were present in RAINBIO under an older synonym.
#
# Strategy:
#   0. Resolve RAINBIO names to WCVP accepted (once) -> rb_accepted set.
#   1. Mechanical diagnostic on raw names (cheap, informative).
#   2. WCVP-resolve the Kew-only names; check accepted name against the
#      ACCEPTED RAINBIO set (not raw RAINBIO).
#   3. Genuine residual = the collection gap.
#   4+ review master, categories, supervisor outputs, breakdowns.
#
# A match-rate sanity check is printed so a one-sided failure is visible.
# =====================================================================

library(rWCVP)
library(readxl)
library(dplyr)
library(stringr)
library(tidyr)

# ---- CONFIG ---------------------------------------------------------
rainbio_csv <- "C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat_labels.csv"
kew_xlsx    <- "C:/Users/liams/Documents/PhD-Project Data/Kew/List of species from Tanzania.xlsx"
outdir      <- "C:/Users/liams/Documents/PhD-Project Data/Reconciliation_Reverse"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
TZ_CODE <- "TAN"
# --------------------------------------------------------------------

clean_name  <- function(x) {
  x <- str_squish(x)
  x <- str_replace_all(x, "\\b(cf|aff)\\.?\\s+", "")
  x <- str_replace_all(x, "\u00d7", "x")
  x
}
to_binomial <- function(x) word(x, 1, 2)

wcvp_tbl <- rWCVPdata::wcvp_names
id_to_name <- wcvp_tbl |>
  select(plant_name_id, taxon_name) |>
  rename(accepted_id = plant_name_id, accepted_name = taxon_name)

# ---- reusable resolver: names -> WCVP accepted (with diagnostics) ----
resolve_block <- function(names_vec, label) {
  message(sprintf("  resolving %s (%d names) to WCVP accepted...", label, length(names_vec)))
  m <- wcvp_match_names(data.frame(scientificname = names_vec),
                        name_col = "scientificname",
                        fuzzy = TRUE, progress_bar = TRUE)
  acc_id_col <- intersect(c("wcvp_accepted_id", "accepted_plant_name_id",
                            "wcvp_accepted_plant_name_id"), names(m))[1]
  status_col <- intersect(c("wcvp_status", "taxon_status"), names(m))[1]
  mtype_col  <- intersect(c("match_type", "wcvp_match_type"), names(m))[1]
  out <- m |>
    mutate(accepted_id = .data[[acc_id_col]]) |>
    left_join(id_to_name, by = "accepted_id") |>
    mutate(
      match_type    = .data[[mtype_col]],
      wcvp_status   = .data[[status_col]],
      accepted_name = coalesce(accepted_name, .data[["scientificname"]])
    ) |>
    select(input_name = scientificname, match_type, wcvp_status, accepted_name)
  # SANITY CHECK: what fraction actually matched WCVP?
  matched_rate <- mean(!is.na(out$accepted_name) &
                       out$accepted_name != out$input_name |
                       out$match_type %in% c("Exact (with author)",
                                             "Exact (without author)") )
  message(sprintf("    [sanity] %s WCVP match rate ~ %.1f%% (should be high, ~90%%+)",
                  label, 100 * mean(!is.na(out$match_type) & out$match_type != "No match")))
  out
}

# ---- load -----------------------------------------------------------
message("Loading sources...")
rb <- read.csv(rainbio_csv, stringsAsFactors = FALSE)
kw <- read_excel(kew_xlsx)

rb_names <- rb$species   |> clean_name() |> unique()
rb_names <- rb_names[!is.na(rb_names) & rb_names != ""]
kw_names <- kw$taxon_name |> clean_name() |> unique()
kw_names <- kw_names[!is.na(kw_names) & kw_names != ""]
message(sprintf("  Kew: %d names | RAINBIO: %d names", length(kw_names), length(rb_names)))


# =====================================================================
# STEP 0 - resolve RAINBIO to accepted names (the fix)
# =====================================================================
message("\n--- STEP 0: resolve RAINBIO to WCVP accepted (both-sides fix) ---")
rb_res <- resolve_block(rb_names, "RAINBIO")
rb_accepted <- unique(na.omit(rb_res$accepted_name))
rb_accepted_binom <- unique(to_binomial(rb_accepted))
message(sprintf("  RAINBIO -> %d accepted species", length(rb_accepted)))


# =====================================================================
# STEP 1 - mechanical diagnostic (raw names, cheap)
# =====================================================================
message("\n--- STEP 1: mechanical diagnostic (raw) ---")
shared_raw  <- intersect(kw_names, rb_names)
kw_only_raw <- setdiff(kw_names, rb_names)
message(sprintf("Exact-string shared      : %d", length(shared_raw)))
message(sprintf("Kew-only (raw)           : %d", length(kw_only_raw)))

kw_only_binom <- to_binomial(kw_only_raw)
rb_binom      <- unique(to_binomial(rb_names))
recovered_by_binomial <- kw_only_raw[kw_only_binom %in% rb_binom]
message(sprintf("  recovered by binomial strip: %d", length(recovered_by_binomial)))

still_unmatched <- setdiff(kw_only_raw, recovered_by_binomial)
message(sprintf("  still unmatched (-> WCVP)  : %d", length(still_unmatched)))


# =====================================================================
# STEP 2 - WCVP-resolve Kew-only names; compare to ACCEPTED RAINBIO set
# =====================================================================
message("\n--- STEP 2: WCVP matching of Kew-only names ---")
resolved <- resolve_block(still_unmatched, "Kew-only")

# THE FIX: check accepted Kew name against ACCEPTED RAINBIO (not raw)
resolved <- resolved |>
  rename(species = input_name) |>
  mutate(accepted_in_rainbio =
           accepted_name %in% rb_accepted |
           to_binomial(accepted_name) %in% rb_accepted_binom)

n_resolved_into_rb <- sum(resolved$accepted_in_rainbio, na.rm = TRUE)
message(sprintf("WCVP Kew names that resolve into ACCEPTED RAINBIO: %d", n_resolved_into_rb))
if (n_resolved_into_rb < 50)
  message("  [WARNING] very low - check the sanity match-rate above; ",
          "if RAINBIO match rate was low, resolution failed upstream.")

write.csv(resolved, file.path(outdir, "step2_wcvp_matched.csv"), row.names = FALSE)


# =====================================================================
# STEP 3 - genuine residual (collection gap)
# =====================================================================
message("\n--- STEP 3: genuine residual ---")
residual <- resolved |>
  filter(!accepted_in_rainbio | is.na(accepted_name)) |>
  arrange(match_type)
message(sprintf("Genuine residual (no RAINBIO equivalent): %d", nrow(residual)))
print(table(residual$match_type, useNA = "ifany"))
write.csv(residual, file.path(outdir, "step3_genuine_residual.csv"), row.names = FALSE)


# =====================================================================
# STEP 4 - review master + categories
# =====================================================================
message("\n--- STEP 4: building review master ---")

kw_meta <- kw |>
  mutate(taxon_name = clean_name(taxon_name)) |>
  select(any_of(c("taxon_name", "family", "lifeform_description",
                  "climate_description"))) |>
  rename(species = taxon_name) |>
  distinct(species, .keep_all = TRUE)

rb_genus <- unique(word(rb_names, 1))
genus_in_rainbio <- function(x) word(x, 1) %in% rb_genus

master <- tibble(species = kw_names) |>
  mutate(
    exact_shared       = species %in% shared_raw,
    recovered_binomial = species %in% recovered_by_binomial
  )

res_small <- resolved |>
  transmute(species, wcvp_match = match_type, wcvp_status,
            accepted_name, accepted_in_rainbio)

master <- master |>
  left_join(res_small, by = "species") |>
  left_join(kw_meta, by = "species") |>
  mutate(
    genus = word(species, 1),
    genus_present_in_rainbio = genus_in_rainbio(species)
  )

master <- master |>
  mutate(review_category = case_when(
    exact_shared ~ "exact_match_in_rainbio",
    recovered_binomial ~ "matched_after_binomial_strip",
    accepted_in_rainbio & wcvp_match == "Exact (without author)" ~ "exact_resolved_into_rainbio",
    accepted_in_rainbio & str_detect(coalesce(wcvp_match, ""), regex("fuzzy", ignore_case = TRUE)) ~ "fuzzy_resolved_into_rainbio_CHECK",
    accepted_in_rainbio ~ "resolved_into_rainbio_other",
    !is.na(wcvp_match) & str_detect(coalesce(wcvp_match, ""), regex("fuzzy", ignore_case = TRUE)) ~ "fuzzy_match_NOT_in_rainbio_CHECK",
    !is.na(accepted_name) & !accepted_in_rainbio ~ "absent_from_rainbio",
    is.na(wcvp_match) | wcvp_match == "No match" ~ "unmatched_no_wcvp_record",
    TRUE ~ "other"
  ))

review <- master |>
  filter(!(review_category %in% c("exact_match_in_rainbio",
                                  "exact_resolved_into_rainbio"))) |>
  arrange(review_category, species) |>
  select(species, review_category, wcvp_match, wcvp_status,
         accepted_name, accepted_in_rainbio,
         genus, genus_present_in_rainbio,
         any_of(c("family", "lifeform_description")))
write.csv(review, file.path(outdir, "step4_review.csv"), row.names = FALSE)
message(sprintf("Review set: %d names", nrow(review)))
print(as.data.frame(table(review$review_category)))


# =====================================================================
# SUPERVISOR-FRIENDLY outputs
# =====================================================================
epithet <- function(x) word(x, 2)
status_rank <- c("Accepted"=1,"Synonym"=2,"Unplaced"=3,
                 "Illegitimate"=4,"Invalid"=5,"Misapplied"=6)

collapse_best <- function(df) {
  df |>
    mutate(.status_rank = coalesce(status_rank[wcvp_status], 9L),
           .epithet_kept = epithet(species) == epithet(accepted_name)) |>
    arrange(species, desc(.epithet_kept), .status_rank) |>
    group_by(species) |>
    mutate(n_candidates = n()) |>
    slice(1) |>
    ungroup() |>
    select(-.status_rank, -.epithet_kept)
}

gap <- master |> filter(review_category == "absent_from_rainbio") |> collapse_best()
fuzzy_bad <- master |> filter(review_category == "fuzzy_match_NOT_in_rainbio_CHECK") |> collapse_best()

tidy_cols <- function(df, note) {
  df |>
    mutate(epithet_preserved = epithet(species) == epithet(accepted_name),
           n_alternatives = pmax(n_candidates - 1, 0)) |>
    transmute(
      kew_name      = species, genus,
      family        = if ("family" %in% names(df)) family else NA_character_,
      growth_form   = if ("lifeform_description" %in% names(df)) lifeform_description else NA_character_,
      wcvp_accepted = accepted_name, wcvp_status,
      match_quality = wcvp_match, epithet_preserved, n_alternatives, note = note
    ) |>
    arrange(family, kew_name)
}

gap_tidy <- tidy_cols(gap,
  "Accepted Kew species with NO RAINBIO equivalent (both sides resolved to WCVP) - a genuine collection gap.")
err_tidy <- tidy_cols(fuzzy_bad,
  "Fuzzy match that did not reach RAINBIO - possible name/matching artefact.")

write.csv(gap_tidy, file.path(outdir, "review_collection_gap.csv"), row.names = FALSE)
write.csv(err_tidy, file.path(outdir, "review_matching_errors.csv"), row.names = FALSE)

by_family <- gap_tidy |> count(family, name = "n_missing") |> arrange(desc(n_missing))
write.csv(by_family, file.path(outdir, "gap_by_family.csv"), row.names = FALSE)

by_growth <- NULL
if (any(!is.na(gap_tidy$growth_form))) {
  by_growth <- gap_tidy |>
    mutate(growth_form = ifelse(is.na(growth_form), "(not given)", growth_form)) |>
    count(growth_form, name = "n_missing") |> arrange(desc(n_missing))
  write.csv(by_growth, file.path(outdir, "gap_by_growthform.csv"), row.names = FALSE)
}

cat_counts <- master |> distinct(species, .keep_all = TRUE) |> count(review_category)
get_n <- function(cat){ v <- cat_counts$n[cat_counts$review_category==cat]; if(length(v)) v else 0L }

summary_tbl <- tibble::tibble(
  stage = c("TOTAL Kew species","Exact match to RAINBIO",
            "Matched after binomial (rank) strip",
            "Synonym/exact resolved into RAINBIO (WCVP)",
            "Fuzzy matched into RAINBIO (WCVP)",
            "ABSENT from RAINBIO (collection gap)",
            "Fuzzy match NOT into RAINBIO (review)"),
  species = c(length(kw_names),
              get_n("exact_match_in_rainbio"),
              get_n("matched_after_binomial_strip"),
              get_n("exact_resolved_into_rainbio"),
              get_n("fuzzy_resolved_into_rainbio_CHECK"),
              nrow(gap_tidy), nrow(err_tidy))
) |> mutate(pct_of_total = round(100 * species / length(kw_names), 1))
write.csv(summary_tbl, file.path(outdir, "reverse_summary.csv"), row.names = FALSE)

reconciled <- sum(summary_tbl$species[2:5])
message("\n========== REVERSE SUMMARY (fixed) ==========")
print(as.data.frame(summary_tbl), row.names = FALSE)
message(sprintf("\nReached RAINBIO (any route): %d of %d (%.1f%%)",
                reconciled, length(kw_names), 100*reconciled/length(kw_names)))
message(sprintf("Collection gap: %d", nrow(gap_tidy)))
message("\nTop families in the gap:"); print(head(by_family, 15))
if (!is.null(by_growth)) { message("\nGap by growth form:"); print(head(by_growth, 15)) }

for (o in c("gap_tidy","err_tidy","summary_tbl","by_family","by_growth"))
  assign(o, get(o), envir = .GlobalEnv)