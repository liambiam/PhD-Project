# =============================================================================
# STRATIFIED RANDOM SAMPLING PIPELINE — RAINBIO DATASET
# =============================================================================
# Phases:
#   1. Clean raw data
#   2. Convert to grid
#   3. Attach environmental data (WorldClim)
#   4. Attach habitat data (IUCN level 2)
#   5. Classify rarity (unsure about)
#   6. Build strata
#   7. Allocate + sample
#   8. Validate
# =============================================================================
 
# --- PACKAGES ----------------------------------------------------------------
install.packages(c("tidyverse", "sf", "terra", "geodata", "ggplot2",
                    "factoextra", "cluster", "vegan", "patchwork"))
 
library(tidyverse)
library(sf)
library(terra)
library(geodata)     # WorldClim download
library(ggplot2)
library(factoextra)  # PCA visualisation
library(cluster)     # k-means
library(patchwork)   # multi-panel plots


# =============================================================================
# PHASE 1 — CLEAN THE RAW DATA
# =============================================================================
 
# --- 1.1 Load RAINBIO --------------------------------------------------------

raw <- read_csv("C:/Users/liams/Documents/PhD-Project/Data/RAINBIO/rainbio_published/published_database/RAINBIO.csv")   
# Inspect
glimpse(raw)

# Standardise column names to lowercase
raw <- raw %>%
  rename_with(tolower) %>%
  rename(
    species   = any_of(c("species", "taxon", "taxon_name")),
    family    = any_of(c("family", "fam")),
    longitude = any_of(c("longitude", "lon", "decimallongitude", "long")),
    latitude  = any_of(c("latitude", "lat", "decimallatitude")),
    basis     = any_of(c("basis_of_record", "basisofrecord", "basis"))
  )
head(raw)
# --- 1.2 Filter to Tanzania --------------------------------------------------


cat("Records before cleaning:", nrow(raw), "\n")
cat("Species before cleaning:", n_distinct(raw$species), "\n")

clean <- raw %>%
  # Remove missing coordinates
  filter(!is.na(longitude), !is.na(latitude)) %>%
  
  # Filter to Tanzania only — check exact value in your data first
  filter(country == "Tanzania") %>%          # try also "TANZANIA" or "Tanzania, United Republic of"
  
  # Remove unreliable basis of record
  filter(!basis %in% c("FOSSIL_SPECIMEN", "LIVING_SPECIMEN",
                        "MACHINE_OBSERVATION", NA)) %>%
  # Remove missing species or family
  filter(!is.na(species), !is.na(family)) %>%
  # Trim whitespace
  mutate(across(c(species, family1), str_trim))

cat("Records after cleaning:", nrow(clean), "\n")
cat("Species after cleaning:", n_distinct(clean$species), "\n")

# --- 1.3 Remove exact spatial duplicates -------------------------------------
clean <- clean %>%
  distinct(species, longitude, latitude, .keep_all = TRUE)
 
cat("Records after deduplication:", nrow(clean), "\n")
cat("Species after deduplication:", n_distinct(clean$species), "\n")


# =============================================================================
# PHASE 2 — CONVERT TO GRID
# =============================================================================
 
resolution <- 0.25  # degrees — change to 0.1 for finer grain
 
# --- 2.1 Assign grid cell IDs ------------------------------------------------
clean <- clean %>%
  mutate(
    cell_lon = floor(longitude / resolution) * resolution,
    cell_lat = floor(latitude  / resolution) * resolution,
    cell_id  = paste(cell_lon, cell_lat, sep = "_")
  )
 
# --- 2.2 Thin to unique species per cell -------------------------------------
thinned <- clean %>%
  distinct(cell_id, species, .keep_all = TRUE)
 
cat("Unique species × cell combinations:", nrow(thinned), "\n")
 
# --- 2.3 Build cell summary table --------------------------------------------
cell_summary <- thinned %>%
  group_by(cell_id, cell_lon, cell_lat) %>%
  summarise(
    species_richness = n_distinct(species),
    n_families       = n_distinct(family1),
    n_records        = n(),            # raw record count (effort proxy)
    .groups = "drop"
  )
 
cat("Total grid cells:", nrow(cell_summary), "\n")
 

# =============================================================================
# PHASE 3 — ATTACH ENVIRONMENTAL DATA
# =============================================================================
 
# --- 3.1 Download WorldClim bioclimatic variables ----------------------------
# Downloads to a local cache — only runs once
bio_stack <- geodata::worldclim_global(
  var  = "bio",
  res  = 0.5,           # 0.5 arcmin resolution
  path = "data/worldclim"
)


# Keep minimum viable set: Bio1, Bio4, Bio12, Bio15
bio_sub <- bio_stack[[c("wc2.1_30s_bio_1",  "wc2.1_30s_bio_4",
                         "wc2.1_30s_bio_12", "wc2.1_30s_bio_15")]]
names(bio_sub) <- c("bio1_temp_mean", "bio4_temp_seasonality",
                    "bio12_precip_mean", "bio15_precip_seasonality")
 
# --- 3.2 Extract climate values at cell centroids ----------------------------
coords_mat <- as.matrix(cell_summary[, c("cell_lon", "cell_lat")])
 
env_vals <- terra::extract(bio_sub, coords_mat) %>%
  as_tibble() 
 
cell_summary <- bind_cols(cell_summary, env_vals)

# Remove cells with no environmental data (ocean / edge cells)
cell_summary <- cell_summary %>%
  filter(if_all(starts_with("bio"), ~ !is.na(.)))

# --- 3.3 PCA on climate variables --------------------------------------------
env_matrix <- cell_summary %>%
  select(starts_with("bio")) %>%
  scale()   # normalise before PCA
 
pca_result <- prcomp(env_matrix, center = FALSE, scale. = FALSE)

# How much variance do PC1 and PC2 explain?
summary(pca_result)$importance[, 1:3]

# Add PC scores to cell summary
cell_summary <- cell_summary %>%
  mutate(
    PC1 = pca_result$x[, 1],
    PC2 = pca_result$x[, 2]
  )

pca_result$rotation  # Check loadings to interpret PCs (e.g., PC1 = temperature gradient, PC2 = precipitation seasonality)

 
# --- 3.4 K-means clustering on PC1 + PC2 → environmental clusters -----------
set.seed(42)
k       <- 5   # adjust based on elbow plot below
km_fit  <- kmeans(cell_summary[, c("PC1", "PC2")], centers = k, nstart = 25)
 
cell_summary <- cell_summary %>%
  mutate(env_cluster = paste0("E", km_fit$cluster))
 
# Elbow plot to help choose k
wss <- map_dbl(2:9, ~ {
  km <- kmeans(cell_summary[, c("PC1", "PC2")], centers = .x, nstart = 25)
  km$tot.withinss
})
 
elbow_plot <- ggplot(tibble(k = 2:9, wss = wss), aes(k, wss)) +
  geom_line() + geom_point() +
  labs(title = "Elbow plot — choose k at the bend",
       x = "Number of clusters (k)", y = "Total within-cluster SS") +
  theme_minimal()
 
print(elbow_plot)

# --- 3.5 Visualise clusters in PC space --------------------------------------
cluster_plot <- ggplot(cell_summary, aes(PC1, PC2, color = env_cluster)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "Environmental clusters in PCA space") +
  theme_minimal() +
  theme(legend.title = element_blank())
print(cluster_plot)


# Load raster
habitat_raster <- rast("C:/Users/liams/Documents/PhD-Project/Data/IUCN Habitat/iucn_habitatclassification_composite_lvl2_ver004/iucn_habitatclassification_composite_lvl2_ver004.tif")

# Extract values
coords_mat <- as.matrix(cell_summary[, c("cell_lon", "cell_lat")])

hab_vals <- terra::extract(habitat_raster, coords_mat) %>%
  as_tibble() %>%
  select(-any_of("ID"))

# Check distribution of habitat codes
table(cell_summary$habitat_code)
names(hab_vals)[1] <- "habitat_code"

cell_summary <- cell_summary %>%
  select(-matches("habitat_code")) %>%  # remove wrong one if exists
  bind_cols(hab_vals)

# Rename for clarity
cell_summary <- cell_summary %>%
  rename(habitat_code = 1)   # adjust if needed




# =============================================================================
# PHASE 5 — CLASSIFY RARITY (EFFORT-CORRECTED)
# =============================================================================
 
# --- 5.1 Flag well-sampled cells (effort filter) -----------------------------
effort_threshold <- quantile(cell_summary$n_records, 0.25)
 
cell_summary <- cell_summary %>%
  mutate(effort_flag = if_else(n_records >= effort_threshold,
                               "adequate", "sparse"))
 
cat("Well-sampled cells:", sum(cell_summary$effort_flag == "adequate"), "\n")
cat("Sparse cells:      ", sum(cell_summary$effort_flag == "sparse"), "\n")
 
# --- 5.2 Compute species range size (cells occupied) ------------------------
# Only within adequate-effort cells
adequate_cells <- cell_summary %>%
  filter(effort_flag == "adequate") %>%
  pull(cell_id)

range_size <- thinned %>%
  filter(cell_id %in% adequate_cells) %>%
  group_by(species) %>%
  summarise(n_cells_occupied = n_distinct(cell_id), .groups = "drop")
 
# --- 5.3 Assign rarity class (quartile-based) --------------------------------
q25 <- quantile(range_size$n_cells_occupied, 0.25)
q50 <- quantile(range_size$n_cells_occupied, 0.50)
q75 <- quantile(range_size$n_cells_occupied, 0.75)
 
range_size <- range_size %>%
  mutate(rarity_class = case_when(
    n_cells_occupied <= q25 ~ "very_rare",
    n_cells_occupied <= q50 ~ "rare",
    n_cells_occupied <= q75 ~ "common",
    TRUE                    ~ "very_common"
  ))
 
cat("\nRarity class distribution:\n")
print(table(range_size$rarity_class))
 
# --- 5.4 Assign dominant rarity class per cell -------------------------------
# A cell's rarity profile = the most frequent rarity class among its species
cell_rarity <- thinned %>%
  left_join(range_size %>% select(species, rarity_class), by = "species") %>%
  filter(!is.na(rarity_class)) %>%
  group_by(cell_id) %>%
  summarise(
    dominant_rarity = names(which.max(table(rarity_class))),
    prop_rare       = mean(rarity_class %in% c("very_rare", "rare")),
    .groups = "drop"
  )
 
cell_summary <- cell_summary %>%
  left_join(cell_rarity, by = "cell_id")
 
# --- 5.5 Two-axis rarity classification (effort-aware) -----------------------
cell_summary <- cell_summary %>%
  mutate(rarity_label = case_when(
    dominant_rarity %in% c("very_rare", "rare") & effort_flag == "adequate" ~ "likely_rare",
    dominant_rarity %in% c("very_rare", "rare") & effort_flag == "sparse"   ~ "uncertain_rare",
    TRUE                                                                      ~ "common"
  ))
 
table(cell_summary$rarity_label)
 
 