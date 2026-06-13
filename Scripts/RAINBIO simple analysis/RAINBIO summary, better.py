"""
============================================================
RAINBIO TANZANIA — DATASET SUMMARY ANALYSIS
============================================================

Purpose: Produce comprehensive characterisation of the RAINBIO
dataset as it applies to Tanzania, for supervisor meeting.

Addresses Neil's "understand your data inside out" requirement:
  - What is RAINBIO?
  - Where does it come from?
  - What cleaning was done?
  - Have non-natives been excluded? (YES — confirmed)
  - How complete is it (% of flora)?
  - Records per species distribution
  - Habit composition
  - Coverage gaps

REFERENCE:
  Dauby et al. (2016) RAINBIO: a mega-database of tropical
  African vascular plants distributions. PhytoKeys 74: 1-18.
  DOI: 10.3897/phytokeys.74.9723

KEY DOCUMENTED FACTS (from Dauby et al. 2016 + GBIF data-use):
  - Raw records: ~977,000 → cleaned: ~614,000 (~37% removed)
  - 1,635 non-native species removed using GBIF cross-check
  - 25,356 native species retained = ~89% of known African flora
  - 91% of species have habit information
  - Time span: 1782-2015
  - 13 source datasets combined
  - Geographic focus: south of Sahel, north of Southern Africa
  - License: CC BY-NC

INPUT:  rainbio_tanzania_species_list.csv
OUTPUT: figures + summary tables in rainbio_summary/
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\liams\Documents\PhD-Project\Data\RAINBIO\rainbio_tanzania_species_list.csv"
OUTPUT_DIR = r"C:\Users\liams\Documents\PhD-Project\Data\RAINBIO\rainbio_summary"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tanzania flora reference numbers (from literature — cite these in thesis)
TANZANIA_FLORA_TOTAL = 12700      # Approximate native vascular plant species
                                  # Source: Flora of Tropical East Africa estimates
RAINBIO_AFRICAN_FLORA_PCT = 89    # % of African flora in RAINBIO (Dauby et al. 2016)
RAINBIO_RAW_RECORDS = 977000      # Initial uncleaned records
RAINBIO_FINAL_RECORDS = 614000    # After cleaning
RAINBIO_NONNATIVE_REMOVED = 1635  # Non-native species excluded

print("=" * 70)
print("RAINBIO TANZANIA — DATASET SUMMARY ANALYSIS")
print("=" * 70)


# ============================================================
# STEP 1 — LOAD AND INSPECT
# ============================================================

print("\n[1] LOADING DATA")
print("-" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip()  # Clean any stray whitespace

print(f"    File           : {INPUT_FILE}")
print(f"    Total records  : {len(df):,}")
print(f"    Columns        : {len(df.columns)}")
print(f"    Column names   : {list(df.columns)}")

# Auto-detect lat/lon columns
lat_col = next((c for c in df.columns
                if c.lower() in ("decimallatitude", "decimallat", "latitude", "lat")), None)
lon_col = next((c for c in df.columns
                if c.lower() in ("decimallongitude", "decimallong", "decimal_long",
                                 "longitude", "lon", "lng")), None)
print(f"    Lat column     : {lat_col}")
print(f"    Lon column     : {lon_col}")


# ============================================================
# STEP 2 — DOCUMENTED PROVENANCE & CLEANING (FOR THE MEETING)
# ============================================================

provenance_table = pd.DataFrame([
    {"Aspect": "Source",                "Value": "Dauby et al. (2016) PhytoKeys 74:1-18"},
    {"Aspect": "DOI",                   "Value": "10.3897/phytokeys.74.9723"},
    {"Aspect": "License",               "Value": "CC BY-NC"},
    {"Aspect": "Source datasets",       "Value": "13 (incl. GBIF, herbaria, field surveys)"},
    {"Aspect": "Geographic scope",      "Value": "Tropical Africa (south of Sahel, north of S. Africa)"},
    {"Aspect": "Temporal scope",        "Value": "1782 – 2015"},
    {"Aspect": "Raw records",           "Value": f"~{RAINBIO_RAW_RECORDS:,}"},
    {"Aspect": "Records after cleaning","Value": f"~{RAINBIO_FINAL_RECORDS:,} (~37% removed)"},
    {"Aspect": "Final species count",   "Value": "25,356 native vascular plants"},
    {"Aspect": "Coverage of African flora", "Value": f"~{RAINBIO_AFRICAN_FLORA_PCT}% of known species"},
    {"Aspect": "Habit info available",  "Value": "91% of species"},
])

cleaning_table = pd.DataFrame([
    {"Cleaning step": "Georeferencing verification",
     "Description": "Automatic + manual checks of all coordinates against country/region polygons"},
    {"Cleaning step": "Taxonomic standardisation",
     "Description": "Names checked against African Plant Database & World Checklist of Selected Plant Families"},
    {"Cleaning step": "Duplicate detection & merging",
     "Description": "Records sharing collector, date, and locality merged"},
    {"Cleaning step": "Non-native species removal",
     "Description": f"{RAINBIO_NONNATIVE_REMOVED:,} non-native species excluded using GBIF distribution checks"},
    {"Cleaning step": "Cultivated specimen flagging",
     "Description": "Cultivated and introduced taxa identified and excluded from main dataset"},
    {"Cleaning step": "Coordinate accuracy coding",
     "Description": "Each record assigned an accuracy code (1=exact, 6=country centroid)"},
    {"Cleaning step": "Expert review",
     "Description": "30+ African flora experts manually reviewed taxonomic and geographic data"},
])

print("\n[2] DOCUMENTED PROVENANCE")
print("-" * 70)
print(provenance_table.to_string(index=False))
print("\n[3] DOCUMENTED CLEANING STEPS")
print("-" * 70)
print(cleaning_table.to_string(index=False))

# Save tables
provenance_table.to_csv(f"{OUTPUT_DIR}/01_provenance_table.csv", index=False)
cleaning_table.to_csv(f"{OUTPUT_DIR}/02_cleaning_table.csv", index=False)


# ============================================================
# STEP 3 — TANZANIA-SPECIFIC SUMMARY STATISTICS
# ============================================================

print("\n[4] TANZANIA SUBSET — KEY STATISTICS")
print("-" * 70)

# Detect species column
species_col = next((c for c in df.columns
                    if c.lower() in ("species", "scientificname", "taxon")), None)
family_col  = next((c for c in df.columns
                    if c.lower() in ("family", "fam")), None)
genus_col   = next((c for c in df.columns
                    if c.lower() in ("genus",)), None)
habit_col   = next((c for c in df.columns
                    if c.lower() in ("a_habit", "habit")), None)

n_records  = len(df)
n_species  = df[species_col].nunique() if species_col else None
n_genera   = df[genus_col].nunique() if genus_col else None
n_families = df[family_col].nunique() if family_col else None

records_per_sp = df.groupby(species_col).size() if species_col else None

# Coverage estimate
coverage_pct = (n_species / TANZANIA_FLORA_TOTAL) * 100 if n_species else None

stats_table = pd.DataFrame([
    {"Metric": "Records (Tanzania)",         "Value": f"{n_records:,}"},
    {"Metric": "Unique species",             "Value": f"{n_species:,}" if n_species else "N/A"},
    {"Metric": "Unique genera",              "Value": f"{n_genera:,}" if n_genera else "N/A"},
    {"Metric": "Unique families",            "Value": f"{n_families:,}" if n_families else "N/A"},
    {"Metric": "Estimated Tanzanian flora",  "Value": f"~{TANZANIA_FLORA_TOTAL:,} species"},
    {"Metric": "Estimated coverage",         "Value": f"~{coverage_pct:.1f}% of native flora" if coverage_pct else "N/A"},
])

if records_per_sp is not None:
    stats_table = pd.concat([stats_table, pd.DataFrame([
        {"Metric": "Records per species (median)", "Value": f"{records_per_sp.median():.0f}"},
        {"Metric": "Records per species (mean)",   "Value": f"{records_per_sp.mean():.1f}"},
        {"Metric": "Records per species (min)",    "Value": f"{records_per_sp.min()}"},
        {"Metric": "Records per species (max)",    "Value": f"{records_per_sp.max():,}"},
        {"Metric": "Species with only 1 record",   "Value": f"{(records_per_sp == 1).sum():,} ({(records_per_sp == 1).sum()/len(records_per_sp)*100:.1f}%)"},
        {"Metric": "Species with <5 records",      "Value": f"{(records_per_sp < 5).sum():,} ({(records_per_sp < 5).sum()/len(records_per_sp)*100:.1f}%)"},
    ])], ignore_index=True)

print(stats_table.to_string(index=False))
stats_table.to_csv(f"{OUTPUT_DIR}/03_tanzania_stats.csv", index=False)


# ============================================================
# STEP 4 — ANSWER NEIL'S QUESTIONS DIRECTLY (FOR THE MEETING)
# ============================================================

neils_questions = pd.DataFrame([
    {"Question": "How many species in Tanzania subset?",
     "Answer": f"{n_species:,} unique species" if n_species else "N/A",
     "Source": "RAINBIO Tanzania filter"},
    {"Question": "What proportion of Tanzanian flora?",
     "Answer": f"~{coverage_pct:.0f}% of estimated {TANZANIA_FLORA_TOTAL:,} native species" if coverage_pct else "N/A",
     "Source": "Cross-ref with FTEA estimate"},
    {"Question": "What % does RAINBIO claim for Africa?",
     "Answer": f"{RAINBIO_AFRICAN_FLORA_PCT}% of all known tropical African plant species",
     "Source": "Dauby et al. 2016"},
    {"Question": "How many specimens per species?",
     "Answer": f"Median: {records_per_sp.median():.0f}, Mean: {records_per_sp.mean():.1f}, Range: {records_per_sp.min()}–{records_per_sp.max():,}" if records_per_sp is not None else "N/A",
     "Source": "RAINBIO Tanzania subset"},
    {"Question": "Have non-natives been excluded?",
     "Answer": f"YES — {RAINBIO_NONNATIVE_REMOVED:,} non-native species removed using GBIF cross-check",
     "Source": "Dauby et al. 2016, sec. 'Identification of introduced and cultivated taxa'"},
    {"Question": "Are cultivated specimens excluded?",
     "Answer": "Cultivated taxa identified and excluded from main dataset",
     "Source": "Dauby et al. 2016"},
    {"Question": "Are anthropogenic-habitat species excluded?",
     "Answer": "Not explicitly — but cultivated specimens are. Habit field can help identify trees vs herbs vs lianas.",
     "Source": "RAINBIO documentation"},
    {"Question": "What habit information is available?",
     "Answer": f"a_habit & a_habitsecond columns; {df[habit_col].notna().sum() / len(df) * 100:.1f}% of records have primary habit" if habit_col else "Not in this subset",
     "Source": "RAINBIO Tanzania subset"},
])

print("\n[5] DIRECT ANSWERS TO SUPERVISOR QUESTIONS")
print("-" * 70)
for _, row in neils_questions.iterrows():
    print(f"\n  Q: {row['Question']}")
    print(f"  A: {row['Answer']}")
    print(f"     [Source: {row['Source']}]")

neils_questions.to_csv(f"{OUTPUT_DIR}/04_supervisor_questions.csv", index=False)


# ============================================================
# STEP 5 — RECORDS PER SPECIES DISTRIBUTION
# ============================================================

if records_per_sp is not None:
    print("\n[6] RECORDS PER SPECIES — DISTRIBUTION")
    print("-" * 70)

    bins = [0, 1, 5, 10, 25, 50, 100, 500, np.inf]
    labels = ["1 record", "2-4", "5-9", "10-24", "25-49", "50-99", "100-499", "500+"]
    binned = pd.cut(records_per_sp, bins=bins, labels=labels, right=False)
    bin_counts = binned.value_counts().sort_index()

    print(f"\n    Distribution of records per species:")
    for lab, n in bin_counts.items():
        bar = "█" * int(n / bin_counts.max() * 30)
        pct = n / len(records_per_sp) * 100
        print(f"    {lab:>10s}  {n:>5,d}  {pct:>5.1f}%  {bar}")


# ============================================================
# STEP 6 — TAXONOMIC BREAKDOWN
# ============================================================

print("\n[7] TAXONOMIC BREAKDOWN")
print("-" * 70)

if family_col and species_col:
    fam_species = df.groupby(family_col)[species_col].nunique().sort_values(ascending=False)
    fam_records = df.groupby(family_col).size().sort_values(ascending=False)

    top_fam_table = pd.DataFrame({
        "Family": fam_species.index[:15],
        "Species": fam_species.values[:15],
        "Records": [fam_records[f] for f in fam_species.index[:15]],
        "% of species": (fam_species.values[:15] / n_species * 100).round(1),
    })
    print(f"\n    Top 15 families by species count:")
    print(top_fam_table.to_string(index=False))
    top_fam_table.to_csv(f"{OUTPUT_DIR}/05_top_families.csv", index=False)


# ============================================================
# STEP 7 — HABIT BREAKDOWN
# ============================================================

if habit_col and species_col:
    print("\n[8] HABIT BREAKDOWN (GROWTH FORM)")
    print("-" * 70)

    habit_species = df.groupby(habit_col)[species_col].nunique().sort_values(ascending=False)
    habit_table = pd.DataFrame({
        "Habit": habit_species.index,
        "Species": habit_species.values,
        "% of species": (habit_species.values / n_species * 100).round(1),
    })
    print(f"\n{habit_table.to_string(index=False)}")
    habit_table.to_csv(f"{OUTPUT_DIR}/06_habit_breakdown.csv", index=False)


# ============================================================
# STEP 8 — SUMMARY DASHBOARD FIGURE
# ============================================================

print("\n[9] CREATING SUMMARY DASHBOARD")
print("-" * 70)

fig = plt.figure(figsize=(18, 12))
fig.suptitle("RAINBIO — Tanzania Dataset Summary", fontsize=16, fontweight="bold", y=0.995)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# --- 8a — Headline numbers ---
ax_head = fig.add_subplot(gs[0, 0])
ax_head.axis("off")
headlines = [
    ("Records", f"{n_records:,}"),
    ("Species", f"{n_species:,}" if n_species else "N/A"),
    ("Families", f"{n_families:,}" if n_families else "N/A"),
    ("% of flora", f"~{coverage_pct:.0f}%" if coverage_pct else "N/A"),
]
y_pos = 0.95
for label, value in headlines:
    ax_head.text(0.05, y_pos, label, fontsize=11, color="#666", transform=ax_head.transAxes)
    ax_head.text(0.05, y_pos - 0.08, value, fontsize=22, fontweight="bold",
                 color="#1F3864", transform=ax_head.transAxes)
    y_pos -= 0.24
ax_head.set_title("Headline numbers", fontsize=11, fontweight="bold", loc="left")

# --- 8b — Records per species histogram ---
ax_hist = fig.add_subplot(gs[0, 1:])
if records_per_sp is not None:
    capped = records_per_sp.clip(upper=100)
    ax_hist.hist(capped, bins=50, color="steelblue", edgecolor="white")
    ax_hist.axvline(records_per_sp.median(), color="red", linestyle="--",
                    label=f"Median: {records_per_sp.median():.0f}")
    ax_hist.set_xlabel("Records per species (capped at 100)")
    ax_hist.set_ylabel("Number of species")
    ax_hist.set_title("Records per species distribution", fontweight="bold")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

# --- 8c — Top families ---
ax_fam = fig.add_subplot(gs[1, :2])
if family_col and species_col:
    top = fam_species.head(15).iloc[::-1]
    ax_fam.barh(top.index, top.values, color="seagreen", edgecolor="white")
    ax_fam.set_xlabel("Number of species")
    ax_fam.set_title("Top 15 families by species richness", fontweight="bold")
    for i, v in enumerate(top.values):
        ax_fam.text(v + 5, i, str(v), va="center", fontsize=9)

# --- 8d — Habit pie ---
ax_habit = fig.add_subplot(gs[1, 2])
if habit_col and species_col:
    habit_data = habit_species[habit_species.index.notna()]
    colors = plt.cm.tab20(np.linspace(0, 1, len(habit_data)))
    wedges, texts, autotexts = ax_habit.pie(
        habit_data.values, labels=habit_data.index, autopct="%1.0f%%",
        startangle=90, colors=colors, textprops={"fontsize": 8})
    ax_habit.set_title("Growth forms (species)", fontweight="bold")

# --- 8e — Cleaning waterfall ---
ax_clean = fig.add_subplot(gs[2, :2])
stages = ["Raw\n(Africa)", "After cleaning\n(Africa)", "Tanzania subset"]
values = [RAINBIO_RAW_RECORDS, RAINBIO_FINAL_RECORDS, n_records]
bars = ax_clean.bar(stages, values, color=["#cccccc", "#888888", "#1F3864"], edgecolor="white")
ax_clean.set_ylabel("Number of records")
ax_clean.set_title("Data cleaning pipeline (continental → Tanzania)", fontweight="bold")
for bar, value in zip(bars, values):
    ax_clean.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                  f"{value:,}", ha="center", fontsize=10, fontweight="bold")
ax_clean.grid(alpha=0.3, axis="y")

# --- 8f — Key facts panel ---
ax_facts = fig.add_subplot(gs[2, 2])
ax_facts.axis("off")
facts = [
    "Source: Dauby et al. 2016",
    "PhytoKeys 74:1-18",
    "DOI: 10.3897/phytokeys.74.9723",
    "",
    f"Non-natives removed: {RAINBIO_NONNATIVE_REMOVED:,}",
    "Cultivated taxa: excluded",
    "13 source datasets combined",
    "Time span: 1782-2015",
    "License: CC BY-NC",
]
for i, fact in enumerate(facts):
    weight = "bold" if i == 0 else "normal"
    size = 10 if i == 0 else 9
    ax_facts.text(0.05, 0.95 - i*0.10, fact, fontsize=size, fontweight=weight,
                  transform=ax_facts.transAxes, color="#333")
ax_facts.set_title("Provenance & cleaning facts", fontsize=11, fontweight="bold", loc="left")

plt.savefig(f"{OUTPUT_DIR}/rainbio_tanzania_dashboard.png", dpi=150, bbox_inches="tight")
print(f"    Saved: {OUTPUT_DIR}/rainbio_tanzania_dashboard.png")
plt.close()


# ============================================================
# STEP 9 — SPECIES LIST EXPORT FOR DOWNSTREAM USE
# ============================================================

print("\n[10] EXPORTING DOWNSTREAM-READY SPECIES LIST")
print("-" * 70)

if species_col and family_col and habit_col:
    species_summary = (
        df.groupby(species_col)
        .agg(
            family=(family_col, "first"),
            primary_habit=(habit_col, lambda x: x.mode()[0] if not x.mode().empty else np.nan),
            n_records=(species_col, "size"),
        )
        .reset_index()
        .sort_values("n_records", ascending=False)
    )
    species_summary.to_csv(f"{OUTPUT_DIR}/07_tanzania_species_summary.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR}/07_tanzania_species_summary.csv")
    print(f"    Total species in summary: {len(species_summary):,}")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE — OUTPUTS SUMMARY")
print("=" * 70)
print(f"\n    All outputs saved to: {OUTPUT_DIR}\n")
print("    Files produced:")
print("      01_provenance_table.csv         — RAINBIO source & metadata")
print("      02_cleaning_table.csv           — Cleaning steps documented")
print("      03_tanzania_stats.csv           — Tanzania subset statistics")
print("      04_supervisor_questions.csv     — Direct answers for meeting")
print("      05_top_families.csv             — Top 15 families")
print("      06_habit_breakdown.csv          — Growth form composition")
print("      07_tanzania_species_summary.csv — Per-species summary table")
print("      rainbio_tanzania_dashboard.png  — Headline summary figure")
print("\n    For the meeting: bring the dashboard PNG and the supervisor")
print("    questions CSV. Both directly address Neil's questions from")
print("    the previous meeting.")
print("=" * 70)