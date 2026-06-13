library(terra)        # raster data (climate layers)
library(sf)           # spatial points. polygons, and vectors
library(dismo)        # SDM tools
library(randomForest) # ML model to classify presence/absence
library(ggplot2)      # plot library
library(geodata)      # downloads global climate data
library(rgbif)        # GBIF data access

# Download climate variables at 10 arc-min resolution, saves in temp path
clim <- worldclim_global(var = "bio", res = 10, path = tempdir())

# Look at variable names
names(clim)           

# Download real species occurrences from GBIF
# Choose a species, e.g., Vulpes vulpes (red fox)
species_key <- name_suggest(q = "Vulpes vulpes", rank = "species")$data$key[1]

# Search for occurrences in the UK
occ_data <- occ_search(taxonKey = species_key, country = "GB", limit = 1000)

# Extract coordinates
occ <- data.frame(
  lon = occ_data$data$decimalLongitude,
  lat = occ_data$data$decimalLatitude
)

# Remove NA coordinates
occ <- na.omit(occ)

# Remove duplicates
occ <- unique(occ)

# Convert occ df to sf (geometry and coordinate reference system)
occ_sf <- st_as_sf(occ, coords = c("lon", "lat"), crs = 4326)

# Sample random locations in clim[1], with 1000 background points, returns coordinates
bg <- spatSample(clim[[1]], size = 1000, method = "random", xy = TRUE)

# Ensure df is 2-columns (lon/lat)
bg_df <- as.data.frame(bg)[, c("x", "y")]

# Pull raster values at each point climate at occurrence and background points
occ_vals <- extract(clim, occ)
bg_vals  <- extract(clim, bg_df)

#Add pa column (pseudo-absence)
occ_vals$pa <- 1  # presence
bg_vals$pa  <- 0  # background

# Combine presence and background into one df using rbind
data_all <- rbind(occ_vals, bg_vals)

# Remove missing data
data_all <- na.omit(data_all)


# Select only the predictor columns
predictors <- names(clim)          # e.g. "bio1", "bio2", ...

# Subset of columns containing only predictors + response
data_all_sub <- data_all[, c(predictors, "pa")]

# Fit classification model predicting presence (1) vs background (0) from climate
model <- randomForest(as.factor(pa) ~ ., data = data_all_sub, ntree = 200)


# Simple prediction of habitat suitability
# Applies the trained model to the raster
pred <- terra::predict(clim, model, type = "response")
# Visualise
plot(pred, main = "Habitat suitability")


# Better prediction of habitat suitability
# terra cannot directly get probs from randomForest, but we can do:
pred_raster <- clim[[1]]  # create empty raster template
values(pred_raster) <- predict(model, as.data.frame(clim, xy=FALSE), type="prob")[,2]
plot(pred_raster, main = "Habitat suitability probability")
