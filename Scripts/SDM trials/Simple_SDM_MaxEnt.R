library(terra)        # raster data
library(sf)           # spatial data
library(dismo)        # SDM tools including MaxEnt
library(rgbif)        # GBIF data access
library(geodata)      # climate data

# Choose a species, e.g., Narcissus pseudonarcissus (daffodil)
species_name <- "Narcissus pseudonarcissus"

# Get species key from GBIF
species_key <- name_suggest(q = species_name, rank = "species")$data$key[1]

# Download occurrences from GBIF (limit to 500 for simplicity, country = GB)
occ_data <- occ_search(taxonKey = species_key, country = "GB", limit = 500)

# Extract coordinates
occ <- data.frame(
  x = occ_data$data$decimalLongitude,
  y = occ_data$data$decimalLatitude
)

# Clean data: remove NAs and duplicates
occ <- na.omit(occ)
occ <- unique(occ)

# Convert to spatial points
occ_sf <- st_as_sf(occ, coords = c("x", "y"), crs = 4326)

# Download temperature data (bio1: annual mean temperature) at 10 arc-min resolution
temp <- worldclim_global(var = "bio", res = 10, path = tempdir())[[1]]  # bio1 is the first layer

# Convert to RasterLayer for compatibility with dismo::maxent
temp_raster <- raster(temp)

# Generate background points (pseudo-absences)
bg <- spatSample(temp, size = 1000, method = "random", xy = TRUE)
bg_df <- as.data.frame(bg)[, c("x", "y")]

# Extract temperature values at occurrence and background points
occ_vals <- extract(temp, occ)
bg_vals <- extract(temp, bg_df)

# Prepare data for MaxEnt: presence and background
pres <- occ_vals[, "wc2.1_10m_bio_1"]  # temperature values for presences
abs <- bg_vals[, "wc2.1_10m_bio_1"]    # temperature values for backgrounds

# Fit MaxEnt model
# MaxEnt requires presence points and predictor raster
me_model <- maxent(x = temp_raster, p = occ, silent = TRUE)

# Predict habitat suitability
pred <- predict(me_model, temp_raster)

# Plot the prediction
plot(pred, main = paste("Habitat suitability for", species_name, "using MaxEnt"))

# Optional: Add occurrence points to the plot
points(occ$x, occ$y, col = "red", pch = 20, cex = 0.5)