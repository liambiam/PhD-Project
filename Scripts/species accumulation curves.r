library(tidyverse)


#==================================================
# PATHS
#==================================================

base_dir <- "C:/Users/liams/Documents/PhD-Project Data/tanzania"

input_file <- file.path(
  base_dir,
  "tanzania_points_with_habitat.csv"
)

#==================================================
# LOAD DATA
#==================================================

df <- read.csv(input_file)

df <- df %>%
  mutate(
    species = as.character(species),
    habitat_iucn = as.character(habitat_iucn),
    a_habit = as.character(a_habit),
    h3 = as.character(h3_cell)   # assumes you already created this
  )

df <- df %>% filter(!is.na(h3))

#==================================================
# OUTPUT FOLDERS
#==================================================

dir.create(file.path(base_dir, "accum_total"), showWarnings = FALSE)
dir.create(file.path(base_dir, "accum_habitat"), showWarnings = FALSE)
dir.create(file.path(base_dir, "accum_growthform"), showWarnings = FALSE)

#==================================================
# PRE-COMPUTE HEX LISTS (VERY IMPORTANT FOR SPEED)
#==================================================

hex_species <- df %>%
  group_by(h3) %>%
  summarise(species = list(unique(species)), .groups = "drop")

hex_lookup <- setNames(hex_species$species, hex_species$h3)

all_hexes <- unique(df$h3)

#==================================================
# FAST ACCUMULATION FUNCTION (HEX-BASED)
#==================================================

accumulate_hexes <- function(hex_vector, lookup, n_perm = 100){

  n <- length(hex_vector)

  all_curves <- matrix(0, nrow = n_perm, ncol = n)

  for(p in 1:n_perm){

    sampled_hexes <- sample(hex_vector)

    seen_species <- new.env(hash = TRUE)
    count <- 0

    for(i in seq_len(n)){

      spp <- lookup[[ sampled_hexes[i] ]]

      # add species in this hex
      for(s in spp){
        if(!exists(s, envir = seen_species)){
          assign(s, TRUE, envir = seen_species)
          count <- count + 1
        }
      }

      all_curves[p, i] <- count
    }

    if(p %% 10 == 0)
      cat("Permutation", p, "\n")
  }

  mean_curve <- colMeans(all_curves)

  lower <- apply(all_curves, 2, quantile, 0.025)
  upper <- apply(all_curves, 2, quantile, 0.975)

  data.frame(
    step = 1:n,
    mean = mean_curve,
    lower = lower,
    upper = upper
  )
}

#==================================================
# 1. TOTAL CURVE
#==================================================

cat("Running TOTAL curve...\n")

total_curve <- accumulate_hexes(
  hex_vector = all_hexes,
  lookup = hex_lookup,
  n_perm = 100
)

write.csv(
  total_curve,
  file.path(base_dir, "accum_total", "total_curve.csv"),
  row.names = FALSE
)

#==================================================
# 2. HABITAT CURVES (NO HEX LABELING BIAS)
#==================================================

cat("Running HABITAT curves...\n")

habitats <- na.omit(unique(df$habitat_iucn))

for(h in habitats){

  sub <- df %>% filter(habitat_iucn == h)

  # species per hex but ONLY within habitat occurrences
  hex_species_hab <- sub %>%
    group_by(h3) %>%
    summarise(species = list(unique(species)), .groups = "drop")

  lookup_h <- setNames(hex_species_hab$species, hex_species_hab$h3)

  hexes_h <- unique(sub$h3)

  if(length(hexes_h) < 20){
    cat("Skipping", h, "\n")
    next
  }

  cat("Processing habitat:", h, "\n")

  curve <- accumulate_hexes(
    hex_vector = hexes_h,
    lookup = lookup_h,
    n_perm = 100
  )

  safe <- gsub("[^A-Za-z0-9]", "_", h)

  write.csv(
    curve,
    file.path(base_dir, "accum_habitat",
              paste0("habitat_", safe, ".csv")),
    row.names = FALSE
  )
}

#==================================================
# 3. GROWTH FORM CURVES
#==================================================

cat("Running GROWTH FORM curves...\n")

forms <- na.omit(unique(df$a_habit))

for(f in forms){

  sub <- df %>% filter(a_habit == f)

  hex_species_f <- sub %>%
    group_by(h3) %>%
    summarise(species = list(unique(species)), .groups = "drop")

  lookup_f <- setNames(hex_species_f$species, hex_species_f$h3)

  hexes_f <- unique(sub$h3)

  if(length(hexes_f) < 20){
    cat("Skipping", f, "\n")
    next
  }

  cat("Processing form:", f, "\n")

  curve <- accumulate_hexes(
    hex_vector = hexes_f,
    lookup = lookup_f,
    n_perm = 100
  )

  safe <- gsub("[^A-Za-z0-9]", "_", f)

  write.csv(
    curve,
    file.path(base_dir, "accum_growthform",
              paste0("growth_", safe, ".csv")),
    row.names = FALSE
  )
}

cat("DONE\n")