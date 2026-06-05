library(dplyr)

df <- read.csv("C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat.csv")
lookup <- read.csv("C:/Users/liams/Documents/PhD-Project Data/IUCN Habitat/IUCN Habitat codes and names.csv",
  stringsAsFactors = FALSE
)

df$habitat_iucn <- as.integer(df$habitat_iucn)
lookup$habitat_iucn <- as.integer(lookup$habitat_iucn)

df <- left_join(
  df,
  lookup,
  by = "habitat_iucn"
)

write.csv(
  df,
  "tanzania_points_with_habitat_labels.csv",
  row.names = FALSE
)

df <- df %>%
  left_join(lookup, by = "habitat_iucn")

write.csv(
  df,
  "C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat_labels.csv",
  row.names = FALSE
)