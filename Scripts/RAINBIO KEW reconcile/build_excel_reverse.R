# =====================================================================
# Build a supervisor-friendly EXCEL workbook from the REVERSE
# (Kew -> RAINBIO) reconciliation. Mirror of the forward Excel builder.
#
# resolution_confidence flag (same logic as forward):
#   HIGH  : Accepted, or Synonym with epithet preserved
#   CHECK : Synonym with epithet changed
#   LOW   : Invalid / Illegitimate / Unplaced / Misapplied
#
# Tabs:
#   1. Summary           - the reverse funnel
#   2. Collection gap    - Kew species absent from RAINBIO (the key set)
#   3. By family         - taxonomic shape of the gap
#   4. By growth form    - gap by Kew lifeform (if available)
#   5. Matching errors   - fuzzy-not-in-RAINBIO (QC pile)
#   6. Status guide      - plain-English WCVP status explanations
#
# Run AFTER reverse_reconcile.R (uses its objects), or reads its CSVs.
# =====================================================================

# install.packages("openxlsx")
library(openxlsx)
library(dplyr)
library(stringr)

outdir <- "C:/Users/liams/Documents/PhD-Project Data/Reconciliation_Reverse"

read_or_use <- function(obj_name, csv_name) {
  if (exists(obj_name) && !is.null(get(obj_name))) get(obj_name)
  else if (file.exists(file.path(outdir, csv_name)))
    read.csv(file.path(outdir, csv_name), stringsAsFactors = FALSE)
  else NULL
}
gap_tidy    <- read_or_use("gap_tidy",    "review_collection_gap.csv")
err_tidy    <- read_or_use("err_tidy",    "review_matching_errors.csv")
summary_tbl <- read_or_use("summary_tbl", "reverse_summary.csv")
by_family   <- read_or_use("by_family",   "gap_by_family.csv")
by_growth   <- read_or_use("by_growth",   "gap_by_growthform.csv")

# ---- add resolution_confidence -------------------------------------
add_confidence <- function(df) {
  if (is.null(df) || !"wcvp_status" %in% names(df)) return(df)
  df |>
    mutate(resolution_confidence = case_when(
      wcvp_status == "Accepted" ~ "HIGH",
      wcvp_status == "Synonym" & epithet_preserved ~ "HIGH",
      wcvp_status == "Synonym" & !epithet_preserved ~ "CHECK",
      wcvp_status %in% c("Invalid", "Illegitimate", "Unplaced", "Misapplied")
        ~ "LOW (problematic name)",
      TRUE ~ "CHECK"
    )) |>
    relocate(resolution_confidence, .after = wcvp_status)
}
gap_tidy <- add_confidence(gap_tidy)
err_tidy <- add_confidence(err_tidy)

# ---- status guide ---------------------------------------------------
status_guide <- tibble::tibble(
  wcvp_status = c("Accepted", "Synonym", "Invalid", "Illegitimate",
                  "Unplaced", "Misapplied"),
  meaning = c(
    "Currently accepted name. Trustworthy.",
    "Valid name but superseded; points to an accepted name. Trustworthy if epithet preserved.",
    "Not validly published (e.g. no proper description). WCVP's suggested link is unreliable - needs manual ID.",
    "Validly published but breaks a naming rule (often a later homonym or superfluous name). Suggested replacement usually OK but worth checking.",
    "Recognised name but not yet placed to an accepted taxon. Unresolved.",
    "Name has been wrongly applied to this plant in the literature. Suggested link unreliable."
  )
)

# ---- build workbook -------------------------------------------------
wb <- createWorkbook()
hdr <- createStyle(textDecoration = "bold", fgFill = "#1F4E79",
                   fontColour = "white", halign = "left", border = "bottom")

add_tab <- function(wb, name, df, note = NULL, widths = NULL) {
  if (is.null(df)) return(invisible())
  addWorksheet(wb, name)
  startRow <- 1
  if (!is.null(note)) {
    writeData(wb, name, note, startRow = 1)
    mergeCells(wb, name, cols = 1:max(2, ncol(df)), rows = 1)
    startRow <- 3
  }
  writeData(wb, name, df, startRow = startRow, headerStyle = hdr)
  freezePane(wb, name, firstActiveRow = startRow + 1)
  if (is.null(widths)) setColWidths(wb, name, cols = 1:ncol(df), widths = "auto")
  else setColWidths(wb, name, cols = 1:length(widths), widths = widths)
}

add_tab(wb, "Summary", summary_tbl,
        note = "Kew -> RAINBIO reverse reconciliation - funnel (distinct species counts)")

add_tab(wb, "Collection gap", gap_tidy,
        note = paste("Accepted Kew Tanzanian species with NO RAINBIO equivalent",
                     "(both sides resolved to WCVP accepted names, so synonyms are not counted as gaps).",
                     "These are species RAINBIO is missing - the collection gap. Sorted by family."))

add_tab(wb, "By family", by_family,
        note = "Number of missing species per family - the taxonomic shape of the gap.")

if (!is.null(by_growth))
  add_tab(wb, "By growth form", by_growth,
          note = "Number of missing species per Kew lifeform description.")

add_tab(wb, "Matching errors", err_tidy,
        note = "Fuzzy matches that did not reach RAINBIO - possible name/matching artefacts.")

add_tab(wb, "Status guide", status_guide, widths = c(16, 90))

# ---- conditional colour on confidence -------------------------------
for (sheet in c("Collection gap", "Matching errors")) {
  df <- if (sheet == "Collection gap") gap_tidy else err_tidy
  if (!is.null(df) && "resolution_confidence" %in% names(df)) {
    col  <- which(names(df) == "resolution_confidence")
    rows <- 4:(3 + nrow(df))
    high  <- createStyle(fgFill = "#C6EFCE")
    check <- createStyle(fgFill = "#FFEB9C")
    low   <- createStyle(fgFill = "#FFC7CE")
    vals <- df$resolution_confidence
    for (i in seq_along(vals)) {
      st <- if (str_detect(vals[i], "HIGH")) high
            else if (str_detect(vals[i], "LOW")) low else check
      addStyle(wb, sheet, st, rows = rows[i], cols = col, gridExpand = FALSE)
    }
  }
}

out_xlsx <- file.path(outdir, "Kew_to_RAINBIO_gap_for_Neil.xlsx")
saveWorkbook(wb, out_xlsx, overwrite = TRUE)
message("Excel workbook written: ", out_xlsx)
message("  Tabs: Summary | Collection gap | By family | By growth form | Matching errors | Status guide")