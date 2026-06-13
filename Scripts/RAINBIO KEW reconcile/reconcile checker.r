library(dplyr)

review <- read.csv("C:/Users/liams/Documents/PhD-Project Data/Reconciliation/step4_review_for_supervisor.csv")
nrow(review)                          # 2734 rows
length(unique(review$species))        # distinct names — will be lower
table(review$review_category)         # where they all sit
# names in RAINBIO but in NEITHER the exact-match set NOR the review file
review |>
  distinct(species, .keep_all = TRUE) |>
  count(review_category)