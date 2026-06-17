# =====================================================================
# Build an Excel workbook from the REVERSE reconciliation
# (Kew species absent from RAINBIO).
#
# Tabs:
#   1. Summary             - how many Kew species missing from RAINBIO
#   2. Missing species     - the full list, with family + growth form
#   3. By family           - counts of missing species per family
#   4. By growth form      - counts per Kew lifeform (if available)
#
# Run AFTER reverse_reconcile_kew_rainbio.R (uses its objects), or it
# reads the CSVs that script wrote.
# =====================================================================

# install.packages("openxlsx")
library(openxlsx)
library(dplyr)

read_or_use <- function(obj_name, csv_name) {
  if (exists(obj_name) && !is.null(get(obj_name))) get(obj_name)
  else if (file.exists(file.path(outdir, csv_name)))
    read.csv(file.path(outdir, csv_name), stringsAsFactors = FALSE)
  else NULL
}

kew_only_df <- read_or_use("kew_only_df", "kew_species_absent_from_rainbio.csv")
by_family   <- read_or_use("by_family",   "kew_absent_by_family.csv")
by_growth   <- read_or_use("by_growth",   "kew_absent_by_growthform.csv")
rev_summary <- if (exists("rev_summary")) rev_summary else NULL

# ---- workbook -------------------------------------------------------
wb <- createWorkbook()
hdr <- createStyle(textDecoration = "bold", fgFill = "#1F4E79",
                   fontColour = "white", halign = "left", border = "bottom")

add_tab <- function(wb, name, df, note = NULL) {
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
  setColWidths(wb, name, cols = 1:ncol(df), widths = "auto")
}

if (!is.null(rev_summary))
  add_tab(wb, "Summary", rev_summary,
          note = "Kew (accepted) Tanzanian species absent from RAINBIO - completeness gap")

add_tab(wb, "Missing species", kew_only_df,
        note = paste("Accepted Tanzanian species in the Kew checklist with NO equivalent in RAINBIO",
                     "(both lists resolved to WCVP accepted names first, so synonyms are not counted as absences).",
                     "These are the species RAINBIO is missing relative to the accepted flora."))

add_tab(wb, "By family", by_family,
        note = "Number of missing species per family - shows the taxonomic shape of the gap.")

if (!is.null(by_growth))
  add_tab(wb, "By growth form", by_growth,
          note = "Number of missing species per Kew lifeform description.")

out_xlsx <- file.path(outdir, "Kew_species_missing_from_RAINBIO.xlsx")
saveWorkbook(wb, out_xlsx, overwrite = TRUE)
message("Excel workbook written: ", out_xlsx)
message("  Tabs: Summary | Missing species | By family | By growth form")
