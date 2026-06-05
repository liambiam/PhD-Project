library(terra)

# Load occurrence data
points <- read.csv("C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points.csv")

# Load habitat raster
habitat <- rast("C:/Users/liams/Documents/PhD-Project Data/IUCN Habitat/iucn_habitatclassification_composite_lvl2_ver004/iucn_habitatclassification_composite_lvl2_ver004.tif")

# Create spatial points
pts <- vect(
  points[, c("decimalLongitude", "decimalLatitude")],
  geom = c("decimalLongitude", "decimalLatitude"),
  crs = "EPSG:4326"
)

# Extract habitat values
habitat_values <- extract(habitat, pts)

# Add habitat to dataframe
points$habitat_iucn <- habitat_values[,2]

# Save result
write.csv(
  points,
  "C:/Users/liams/Documents/PhD-Project Data/tanzania/tanzania_points_with_habitat.csv",
  row.names = FALSE
)