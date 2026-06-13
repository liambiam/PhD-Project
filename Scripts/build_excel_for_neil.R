# =====================================================================
# Build a single, supervisor-friendly EXCEL workbook from the
# RAINBIO/Kew reconciliation review outputs.
#
# Adds a `resolution_confidence` flag based on WCVP nomenclatural status:
#   HIGH  : Accepted, or Synonym with the epithet preserved (real reclass)
#   CHECK : Synonym with epithet changed (possible wrong species)
#   LOW   : Invalid / Illegitimate / Unplaced / Misapplied  (name itself
#           is nomenclaturally problematic - WCVP's suggestion unreliable)
#
# Tabs:
#   1. Summary               - the reconciliation funnel
#   2. Distribution questions - 854 'in WCVP not in Kew' (the interesting set)
#   3. Matching errors        - 52 fuzzy-not-in-Kew (discard pile)
#   4. Status guide           - plain-English explanation of WCVP statuses
#
# Run AFTER supervisor_outputs.R (uses dist_q, err, summary_tbl, outdir).
# If those aren't in memory, it reads the CSVs they wrote.
# =====================================================================

# install.packages("openxlsx")   # if needed
library(openxlsx)
library(dplyr)
library(stringr)

outdir      <- "C:/Users/liams/Documents/PhD-Project Data/Reconciliation"

# ---- load review tables (from memory, or from the CSVs) -------------
read_or_use <- function(obj_name, csv_name) {
  if (exists(obj_name)) get(obj_name)
  else read.csv(file.path(outdir, csv_name), stringsAsFactors = FALSE)
}
dist_q      <- read_or_use("dist_q",      "review_distribution_questions.csv")
err         <- read_or_use("err",         "review_matching_errors.csv")
summary_tbl <- read_or_use("summary_tbl", "reconciliation_summary.csv")

# ---- add resolution_confidence based on wcvp_status -----------------
add_confidence <- function(df) {
  if (!"wcvp_status" %in% names(df)) return(df)
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
dist_q <- add_confidence(dist_q)
err    <- add_confidence(err)

# ---- status guide tab ----------------------------------------------
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

# ---- build the workbook --------------------------------------------
wb <- createWorkbook()

hdr <- createStyle(textDecoration = "bold", fgFill = "#1F4E79",
                   fontColour = "white", halign = "left", border = "bottom")
wrap <- createStyle(wrapText = TRUE, valign = "top")

add_tab <- function(wb, name, df, note = NULL, widths = NULL) {
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
        note = "RAINBIO vs Kew reconciliation - funnel (distinct species counts)")

add_tab(wb, "Distribution questions", dist_q,
        note = paste("Accepted/valid species in RAINBIO + WCVP but NOT in our Kew Tanzania list.",
                     "Are these genuine Tanzanian species the Kew list is missing? Sorted by no. of RAINBIO records."))

add_tab(wb, "Matching errors", err,
        note = "Fuzzy matches that did not resolve into Kew - mostly matching errors to discard (e.g. shared epithet, unrelated genus).")

add_tab(wb, "Status guide", status_guide,
        widths = c(16, 90))

# conditional colour on confidence (green/amber/red) where present
for (sheet in c("Distribution questions", "Matching errors")) {
  df <- if (sheet == "Distribution questions") dist_q else err
  if ("resolution_confidence" %in% names(df)) {
    col <- which(names(df) == "resolution_confidence")
    rows <- (4):(3 + nrow(df))   # data starts row 4 (note+blank+header)
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

out_xlsx <- file.path(outdir, "RAINBIO_Kew_review_for_Neil.xlsx")
saveWorkbook(wb, out_xlsx, overwrite = TRUE)
message("Excel workbook written: ", out_xlsx)
message("  Tabs: Summary | Distribution questions | Matching errors | Status guide")

count(wcvp_status)
