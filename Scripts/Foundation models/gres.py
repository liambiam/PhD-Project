import pandas as pd
rainbio = pd.read_csv("C:/Users/liams/Documents/PhD-Project/Data/RAINBIO/rainbio_tanzania_species_list.csv", low_memory=False, nrows=5)
print(rainbio.columns.tolist())
print(rainbio.head())