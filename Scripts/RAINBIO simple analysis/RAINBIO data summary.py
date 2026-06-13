import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:/Users/liams/Downloads/rainbio_published/published_database/RAINBIO.csv"        # <-- change to your file path
COUNTRY_FOCUS = "Tanzania"        # <-- change if needed
OUTPUT_PREFIX = "rainbio"         # prefix for saved outputs

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("RAINBIO DATASET ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"\n[1] RAW DATA LOADED")
print(f"    Total records      : {len(df):,}")
print(f"    Total columns      : {len(df.columns)}")
print(f"    Columns            : {list(df.columns)}")

# ============================================================
# 2. OVERALL DATASET SUMMARY
# ============================================================

print(f"\n[2] OVERALL DATASET SUMMARY")
print(f"    Unique species     : {df['species'].nunique():,}")
print(f"    Unique genera      : {df['genus'].nunique():,}")
print(f"    Unique families    : {df['family'].nunique():,}")
print(f"    Unique orders      : {df['order'].nunique():,}")
print(f"    Countries covered  : {df['country'].nunique():,}")

# ============================================================
# 3. DATA QUALITY ASSESSMENT
# ============================================================

print(f"\n[3] DATA QUALITY")

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
quality_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
quality_df = quality_df[quality_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
print("\n    Missing values per column:")
print(quality_df.to_string())

# Coordinate quality
no_coords = df[df['decimalLat'].isnull() | df['decimalLong'].isnull()]
print(f"\n    Records missing coordinates     : {len(no_coords):,} ({len(no_coords)/len(df)*100:.1f}%)")

# Duplicate records
dupes = df.duplicated(subset=['species', 'decimalLat', 'decimalLong'], keep=False)
print(f"    Duplicate species+coord records : {dupes.sum():,}")

# Taxon rank breakdown
print(f"\n    Taxon rank breakdown:")
print(df['taxon_rank'].value_counts().to_string() if 'taxon_rank' in df.columns else "    (taxon_rank column not found)")

# Basis of record
print(f"\n    Basis of record breakdown:")
print(df['basisofrecord'].value_counts().to_string())

# Coordinate accuracy
print(f"\n    Coordinate accuracy (calc_accuracy) breakdown:")
print(df['calc_accuracy'].value_counts().to_string())

# ============================================================
# 4. TANZANIA-SPECIFIC ANALYSIS
# ============================================================

tz = df[df['country'].str.contains(COUNTRY_FOCUS, case=False, na=False)].copy()

print(f"\n[4] TANZANIA-SPECIFIC SUMMARY")
print(f"    Total records      : {len(tz):,}")
print(f"    Unique species     : {tz['species'].nunique():,}")
print(f"    Unique genera      : {tz['genus'].nunique():,}")
print(f"    Unique families    : {tz['family'].nunique():,}")
print(f"    Records with coords: {tz['decimalLat'].notna().sum():,} ({tz['decimalLat'].notna().sum()/len(tz)*100:.1f}%)")

# Records per species distribution
records_per_species = tz.groupby('species').size()
print(f"\n    Records per species:")
print(f"      Mean             : {records_per_species.mean():.1f}")
print(f"      Median           : {records_per_species.median():.1f}")
print(f"      Min              : {records_per_species.min()}")
print(f"      Max              : {records_per_species.max()}")
print(f"      Species with 1 record only : {(records_per_species == 1).sum():,} ({(records_per_species == 1).sum()/len(records_per_species)*100:.1f}%)")

# ============================================================
# 5. TAXONOMIC BREAKDOWN — TANZANIA
# ============================================================

print(f"\n[5] TAXONOMIC BREAKDOWN (Tanzania)")

print(f"\n    Top 20 families by species count:")
fam_species = tz.groupby('family')['species'].nunique().sort_values(ascending=False).head(20)
print(fam_species.to_string())

print(f"\n    Top 20 genera by species count:")
gen_species = tz.groupby('genus')['species'].nunique().sort_values(ascending=False).head(20)
print(gen_species.to_string())

print(f"\n    Top 10 orders by species count:")
ord_species = tz.groupby('order')['species'].nunique().sort_values(ascending=False).head(10)
print(ord_species.to_string())

# ============================================================
# 6. HABIT / GROWTH FORM BREAKDOWN — TANZANIA
# ============================================================

print(f"\n[6] GROWTH FORM / HABIT (Tanzania)")

print(f"\n    Primary habit breakdown (species):")
habit_species = tz.groupby('a_habit')['species'].nunique().sort_values(ascending=False)
print(habit_species.to_string())

print(f"\n    Secondary habit breakdown (species):")
habit2_species = tz.groupby('a_habitsecond')['species'].nunique().sort_values(ascending=False)
print(habit2_species.to_string())

# ============================================================
# 7. GEOGRAPHIC SPREAD — TANZANIA
# ============================================================

print(f"\n[7] GEOGRAPHIC SPREAD (Tanzania)")

tz_coords = tz.dropna(subset=['decimalLat', 'decimalLong'])
print(f"    Latitude  range    : {tz_coords['decimalLat'].min():.3f} to {tz_coords['decimalLat'].max():.3f}")
print(f"    Longitude range    : {tz_coords['decimalLong'].min():.3f} to {tz_coords['decimalLong'].max():.3f}")

# Grid cell coverage (1-degree cells as rough proxy for spatial spread)
tz_coords['lat_grid'] = tz_coords['decimalLat'].round(0)
tz_coords['lon_grid'] = tz_coords['decimalLong'].round(0)
grid_cells = tz_coords.groupby(['lat_grid', 'lon_grid']).size()
print(f"    Unique 1° grid cells occupied : {len(grid_cells):,}")
print(f"    Mean records per grid cell    : {grid_cells.mean():.1f}")

# ============================================================
# 8. INSTITUTION & COLLECTION BREAKDOWN
# ============================================================

print(f"\n[8] COLLECTION SOURCES (Tanzania)")
print(f"\n    Top 10 institutions by record count:")
print(tz['institutionCode'].value_counts().head(10).to_string())

print(f"\n    Kind of collection breakdown:")
print(tz['kind_col'].value_counts().to_string())

# ============================================================
# 9. SPECIES LIST FOR TANZANIA — EXPORT
# ============================================================

tz_species = (
    tz.groupby('species')
    .agg(
        family=('family', 'first'),
        genus=('genus', 'first'),
        order=('order', 'first'),
        primary_habit=('a_habit', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        secondary_habit=('a_habitsecond', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        n_records=('species', 'count'),
        n_coords=('decimalLat', lambda x: x.notna().sum()),
        mean_lat=('decimalLat', 'mean'),
        mean_lon=('decimalLong', 'mean'),
    )
    .reset_index()
    .sort_values('n_records', ascending=False)
)

species_output = f"{OUTPUT_PREFIX}_tanzania_species_list.csv"
tz_species.to_csv(species_output, index=False)
print(f"\n[9] TANZANIA SPECIES LIST EXPORTED")
print(f"    File               : {species_output}")
print(f"    Total species      : {len(tz_species):,}")

# ============================================================
# 10. FIGURES
# ============================================================

fig = plt.figure(figsize=(18, 14))
fig.suptitle(f'RAINBIO — Tanzania Analysis', fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# --- 10a. Top 15 families by species count ---
ax1 = fig.add_subplot(gs[0, :2])
fam_plot = fam_species.head(15).sort_values()
fam_plot.plot(kind='barh', ax=ax1, color='steelblue', edgecolor='white')
ax1.set_title('Top 15 Families by Species Count', fontweight='bold')
ax1.set_xlabel('Number of Species')
ax1.set_ylabel('')

# --- 10b. Primary habit breakdown ---
ax2 = fig.add_subplot(gs[0, 2])
habit_plot = habit_species[habit_species.index.notna()]
habit_plot.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 7})
ax2.set_title('Primary Habit (Species)', fontweight='bold')
ax2.set_ylabel('')

# --- 10c. Records per species distribution ---
ax3 = fig.add_subplot(gs[1, 0])
records_per_species_plot = records_per_species[records_per_species <= 50]
ax3.hist(records_per_species_plot, bins=30, color='darkorange', edgecolor='white')
ax3.set_title('Records per Species\n(capped at 50)', fontweight='bold')
ax3.set_xlabel('Number of Records')
ax3.set_ylabel('Number of Species')

# --- 10d. Species per family distribution ---
ax4 = fig.add_subplot(gs[1, 1])
spp_per_fam = tz.groupby('family')['species'].nunique()
ax4.hist(spp_per_fam, bins=30, color='mediumseagreen', edgecolor='white')
ax4.set_title('Species per Family\nDistribution', fontweight='bold')
ax4.set_xlabel('Number of Species')
ax4.set_ylabel('Number of Families')

# --- 10e. Coordinate scatter (rough map) ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(tz_coords['decimalLong'], tz_coords['decimalLat'],
            alpha=0.1, s=1, color='royalblue')
ax5.set_title('Record Locations\n(Tanzania)', fontweight='bold')
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')

# --- 10f. Missing data heatmap ---
ax6 = fig.add_subplot(gs[2, :])
key_cols = ['species', 'family', 'genus', 'order', 'decimalLat', 'decimalLong',
            'a_habit', 'a_habitsecond', 'calc_accuracy', 'basisofrecord', 'kind_col']
key_cols = [c for c in key_cols if c in tz.columns]
missing_tz = tz[key_cols].isnull().mean() * 100
missing_tz.plot(kind='bar', ax=ax6, color='tomato', edgecolor='white')
ax6.set_title('% Missing Values per Key Column (Tanzania records)', fontweight='bold')
ax6.set_xlabel('')
ax6.set_ylabel('% Missing')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=30, ha='right')
ax6.axhline(y=20, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='20% threshold')
ax6.legend()

figures_output = f"{OUTPUT_PREFIX}_figures.png"
plt.savefig(figures_output, dpi=150, bbox_inches='tight')
print(f"\n[10] FIGURES SAVED")
print(f"    File               : {figures_output}")

plt.show()

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)