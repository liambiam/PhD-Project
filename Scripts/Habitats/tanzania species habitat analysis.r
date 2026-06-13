# =========================
# HABITAT FREQUENCY ANALYSIS
# =========================

# Packages
library(tidyverse)
library(ggplot2)
library(dplyr)

# -------------------------
# 1. Load data
# -------------------------
df <- read.csv("C://Users/liams/Documents/PhD-Project/Data/tanzania/habitat frequency data.csv", stringsAsFactors = FALSE)

# Inspect
glimpse(df)
head(df)

# -------------------------
# 2. Clean column names (important if messy from export)
# -------------------------
df <- df %>%
  rename(
    habitat_code = habitat_1,
    count = count,
    unique_species = unique,
    empty = empty,
    filled = filled,
    min_species = min,
    max_species = max,
    min_length = min_length,
    max_length = max_length,
    mean_length = mean_length,
    habitat_name = habitat.codes_habitat
  )

# -------------------------
# 3. Basic summaries
# -------------------------

# Total occurrences per habitat
habitat_summary <- df %>%
  group_by(habitat_code, habitat_name) %>%
  summarise(
    total_count = sum(count, na.rm = TRUE),
    total_unique_species = sum(unique_species, na.rm = TRUE),
    n_records = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_count))

print(n=20,habitat_summary)

write.csv(habitat_summary, "habitat_summary.csv", row.names = FALSE)

top10 <- habitat_summary %>%
  slice_max(total_unique_species, n = 10) %>%
  mutate(habitat_name = reorder(habitat_name, total_unique_species))

ggplot(top10, aes(x = total_unique_species, y = habitat_name)) +
  geom_col() +
  labs(
    title = "Top 10 habitats by species richness",
    x = "Number of unique species",
    y = "Habitat"
  ) +
  theme_minimal()
# -------------------------
# 4. Species richness vs abundance relationship
# -------------------------

ggplot(df, aes(x = count, y = unique_species)) +
  geom_point(alpha = 0.6) +
  scale_x_log10() +
  labs(
    title = "Species richness vs occurrences per habitat",
    x = "Total occurrences (log scale)",
    y = "Unique species"
  ) +
  theme_minimal()

# -------------------------
# 5. Habitat ranking by richness
# -------------------------

richness_plot <- habitat_summary %>%
  arrange(total_unique_species) %>%
  mutate(habitat_name = factor(habitat_name, levels = habitat_name))

ggplot(richness_plot, aes(x = total_unique_species, y = habitat_name)) +
  geom_col() +
  labs(
    title = "Species richness per habitat",
    x = "Unique species",
    y = "Habitat"
  ) +
  theme_minimal()

# -------------------------
# 6. Dominance structure (mean vs max species length proxy)
# -------------------------

ggplot(df, aes(x = mean_length, y = max_length)) +
  geom_point(alpha = 0.6) +
  labs(
    title = "Species list structure within habitats",
    x = "Mean species list length",
    y = "Max species list length"
  ) +
  theme_minimal()

# -------------------------
# 7. Identify most and least diverse habitats
# -------------------------

top_habitats <- habitat_summary %>%
  slice_max(total_unique_species, n = 10)

bottom_habitats <- habitat_summary %>%
  slice_min(total_unique_species, n = 10)

print(top_habitats)
print(bottom_habitats)

# -------------------------
# 8. Save outputs
# -------------------------

write.csv(habitat_summary, "habitat_summary_output.csv", row.names = FALSE)