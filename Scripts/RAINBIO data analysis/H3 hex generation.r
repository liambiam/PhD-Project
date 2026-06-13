
library(h3jsr)
library(dplyr)

df <- read.csv("C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat.csv")

df$h3 <- latlng_to_cell(
  lat = df$decimalLatitude,
  lng = df$decimalLongitude,
  res = 4
)