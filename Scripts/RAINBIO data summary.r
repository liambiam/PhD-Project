# ============================================================
# LIBRARIES
# ============================================================

library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(stringr)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE <- "C:/Users/liams/Documents/PhD-Project/Data/RAINBIO/rainbio_published/published_database/RAINBIO.csv"
COUNTRY_FOCUS <- "Tanzania"
OUTPUT_PREFIX <- "rainbio"

# ============================================================
# 1. LOAD DATA
# ============================================================

cat("============================================================\n")
cat("RAINBIO DATASET ANALYSIS\n")
cat("============================================================\n")

df <- read_csv(INPUT_FILE, show_col_types = FALSE)

cat("\n[1] RAW DATA LOADED\n")
cat(sprintf("    Total records      : %s\n", format(nrow(df), big.mark=",")))
cat(sprintf("    Total columns      : %s\n", ncol(df)))
cat("    Columns            :", paste(names(df), collapse=", "), "\n")

# ============================================================
# 2. OVERALL DATASET SUMMARY
# ============================================================

cat("\n[2] OVERALL DATASET SUMMARY\n")

cat(sprintf("    Unique species     : %s\n", format(n_distinct(df$species), big.mark=",")))
cat(sprintf("    Unique genera      : %s\n", format(n_distinct(df$genus), big.mark=",")))
cat(sprintf("    Unique families    : %s\n", format(n_distinct(df$family), big.mark=",")))
cat(sprintf("    Unique orders      : %s\n", format(n_distinct(df$order), big.mark=",")))
cat(sprintf("    Countries covered  : %s\n", format(n_distinct(df$country), big.mark=",")))

# ============================================================
# 3. DATA QUALITY
# ============================================================

cat("\n[3] DATA QUALITY\n")

missing_df <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to="column", values_to="missing_count") %>%
  mutate(missing_pct = round(missing_count / nrow(df) * 100, 2)) %>%
  filter(missing_count > 0) %>%
  arrange(desc(missing_pct))

print(missing_df)

# Coordinates
no_coords <- df %>% filter(is.na(decimalLatitude) | is.na(decimalLongitude))
cat(sprintf("\n    Records missing coordinates : %s (%.1f%%)\n",
            nrow(no_coords), nrow(no_coords)/nrow(df)*100))

# Duplicates
dupes <- df %>%
  group_by(species, decimalLatitude, decimalLongitude) %>%
  filter(n() > 1)

cat(sprintf("    Duplicate species+coord records : %s\n", nrow(dupes)))

# Taxon rank
if ("taxon_rank" %in% names(df)) {
  cat("\n    Taxon rank breakdown:\n")
  print(table(df$taxon_rank))
}

cat("\n    Basis of record:\n")
print(table(df$basisofrecord))

cat("\n    Coordinate accuracy:\n")
print(table(df$calc_accuracy))

# ============================================================
# 4. TANZANIA FILTER
# ============================================================

tz <- df %>%
  filter(str_detect(country, regex(COUNTRY_FOCUS, ignore_case = TRUE)))

cat("\n[4] TANZANIA-SPECIFIC SUMMARY\n")

cat(sprintf("    Total records      : %s\n", format(nrow(tz), big.mark=",")))
cat(sprintf("    Unique species     : %s\n", n_distinct(tz$species)))
cat(sprintf("    Unique genera      : %s\n", n_distinct(tz$genus)))
cat(sprintf("    Unique families    : %s\n", n_distinct(tz$family)))
cat(sprintf("    basis of record    : %s\n", n_distinct(tz$basisOfRecord)))

tz %>%
  count(basisOfRecord) %>%
  ggplot(aes(x = reorder(basisOfRecord, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Records by Basis of Record (Tanzania)",
       x = NULL, y = "Count") +
  theme_minimal()

coords_count <- sum(!is.na(tz$decimalLatitude))
cat(sprintf("    Records with coords: %s (%.1f%%)\n",
            coords_count, coords_count/nrow(tz)*100))

tz %>%
  count(institutionCode, sort = TRUE) %>%
  mutate(pct = round(n / sum(n) * 100, 1))








# Records per species
records_per_species <- tz %>%
  group_by(species) %>%
  summarise(n = n()) %>%
  arrange(desc(n))

cat("\n    Records per species:\n")
cat(sprintf("      Mean   : %.1f\n", mean(records_per_species$n)))
cat(sprintf("      Median : %.1f\n", median(records_per_species$n)))
cat(sprintf("      Min    : %s\n", min(records_per_species$n)))
cat(sprintf("      Max    : %s\n", max(records_per_species$n)))

singletons <- sum(records_per_species$n == 1)
cat(sprintf("      Species with 1 record: %s (%.1f%%)\n",
            singletons, singletons/nrow(records_per_species)*100))

# ============================================================
# 5. TAXONOMY
# ============================================================

cat("\n[5] TAXONOMIC BREAKDOWN\n")

fam_species <- tz %>%
  group_by(family) %>%
  summarise(n_species = n_distinct(species)) %>%
  arrange(desc(n_species)) %>%
  slice_head(n = 20)

print(fam_species)

# ============================================================
# 6. HABIT
# ============================================================

cat("\n[6] HABIT\n")

habit_species <- tz %>%
  group_by(a_habit) %>%
  summarise(n_species = n_distinct(species)) %>%
  arrange(desc(n_species))

print(habit_species)

# ============================================================
# 7. GEOGRAPHY
# ============================================================

cat("\n[7] GEOGRAPHY\n")

tz_coords <- tz %>%
  filter(!is.na(decimalLatitude), !is.na(decimalLongitude))

cat(sprintf("    Latitude range : %.3f to %.3f\n",
            min(tz_coords$decimalLatitude), max(tz_coords$decimalLatitude)))
cat(sprintf("    Longitude range: %.3f to %.3f\n",
            min(tz_coords$decimalLongitude), max(tz_coords$decimalLongitude)))

# ============================================================
# 8. EXPORT SPECIES LIST
# ============================================================

mode_safe <- function(x) {
  ux <- na.omit(unique(x))
  if (length(ux) == 0) return(NA)
  ux[which.max(tabulate(match(x, ux)))]
}

tz_species <- tz %>%
  group_by(species) %>%
  summarise(
    family = first(family),
    genus = first(genus),
    order = first(order),
    primary_habit = mode_safe(a_habit),
    secondary_habit = mode_safe(a_habit_secondary),
    n_records = n(),
    n_coords = sum(!is.na(decimalLatitude)),
    mean_lat = mean(decimalLatitude, na.rm = TRUE),
    mean_lon = mean(decimalLongitude, na.rm = TRUE)
  ) %>%
  arrange(desc(n_records))

write_csv(tz_species, paste0(OUTPUT_PREFIX, "_tanzania_species_list.csv"))

cat("\n[8] SPECIES LIST EXPORTED\n")

# ============================================================
# 9. FIGURES (ggplot)
# ============================================================

# Top families
ggplot(fam_species %>% slice_head(n=15),
       aes(x = reorder(family, n_species), y = n_species)) +
  geom_col() +
  coord_flip() +
  ggtitle("Top Families by Species Count")

# Map scatter
ggplot(tz_coords, aes(x = decimalLongitude, y = decimalLatitude)) +
  geom_point(alpha = 0.1, size = 0.5) +
  ggtitle("Record Locations (Tanzania)")

cat("\n============================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("============================================================\n")

tz_points <- tz %>%
  filter(!is.na(decimalLatitude), !is.na(decimalLongitude)) %>%
  select(species, decimalLatitude, decimalLongitude, family, a_habit)

write_csv(tz_points, "tanzania_points.csv")