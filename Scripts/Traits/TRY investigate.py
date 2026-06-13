import pandas as pd

cols = pd.read_csv(r"C:\Users\liams\Documents\PhD-Project Data\TRY\SLA, PH, SM\50190.txt", sep="\t", encoding="latin-1", nrows=0).columns.tolist()
print(cols)

s = pd.read_csv(r"C:\Users\liams\Documents\PhD-Project Data\TRY\SLA, PH, SM\50190.txt", sep="\t", encoding="latin-1",
                nrows=200_000, low_memory=False)
print(s["TraitName"].value_counts(dropna=False))

t = s[s["TraitID"].notna()]
print(t.groupby("TraitName")["AccSpeciesName"].nunique())

s.head(10)