library(readr)
library(dplyr)

path <- "C:/Users/liams/Documents/PhD-Project Data/TRY/SLA, PH, SM/50190.txt"

# --- column names only (like nrows=0) ---
cols <- names(read_tsv(path, locale = locale(encoding = "latin1"),
                       n_max = 0, show_col_types = FALSE))
print(cols)

# --- read first 200,000 rows (like nrows=200_000) ---
s <- read_tsv(path,
              locale = locale(encoding = "latin1"),
              n_max  = 1000000,
              show_col_types = FALSE)

# --- TraitName value counts, including NAs (like value_counts(dropna=False)) ---
s |>
  count(TraitName) |>
  arrange(desc(n)) |>
  print(n = Inf)

# --- distinct species per trait, trait rows only (TraitID not NA) ---
s |>
  filter(!is.na(TraitID)) |>
  group_by(TraitName) |>
  summarise(n_species = n_distinct(AccSpeciesName)) |>
  print(n = Inf)

# --- first 10 rows ---
head(s, 10)