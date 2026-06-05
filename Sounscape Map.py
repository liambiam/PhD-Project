"""
TREES · Château du Fraïssinet — Soundscape Map
================================================
Run locally where OSM tiles are accessible.

Requirements:
    pip install matplotlib reportlab requests Pillow contextily

Usage:
    python soundscape_map_local.py
    → outputs soundscape_map_final.pdf
"""

import math, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ── Try to import contextily for OSM tiles ─────────────────────────────────
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("contextily not found — map will render without background tiles.")
    print("Install with: pip install contextily")

# ── Data ──────────────────────────────────────────────────────────────────────
# Format: (group, soundtype, label, lat, lon)
# Edits applied:
#   - Big House Train moved north from 44.227008 → 44.2278
#   - One Gunshot moved west (Big House Gunshot!!! lon shifted from 3.998119 → 3.9975)
#   - Southernmost Firecrest & Treecreeper (Wavewalkers northern ones) removed
#   - Church bells removed
#   - Car driving by added at 44.2288, 3.996323
#   - Loud noises from the terrasse added at 44.2284, 3.9987

data = [
    # ── The Full-Spectrals ──────────────────────────────────────────────────
    ("The Full-Spectrals", "Geophony",    "River flowing",              44.228417, 3.99848),
    ("The Full-Spectrals", "Biophony",    "European Robin",             44.228449, 3.998545),
    ("The Full-Spectrals", "Biophony",    "Common Swift",               44.22834,  3.998357),
    ("The Full-Spectrals", "Geophony",    "Rustling leaves",            44.2283,   3.998334),
    ("The Full-Spectrals", "Anthrophony", "Terrace noise",              44.228237, 3.998031),
    ("The Full-Spectrals", "Biophony",    "Crickets",                   44.2285,   3.997183),
    ("The Full-Spectrals", "Biophony",    "Common Nightingale",         44.228752, 3.996973),
    ("The Full-Spectrals", "Anthrophony", "Explosion(?)",               44.228294, 3.996817),

    # ── Liam's Minions ─────────────────────────────────────────────────────
    ("Liam's Minions",     "Anthrophony", "Voices",                     44.227875, 3.997353),
    ("Liam's Minions",     "Anthrophony", "Water trough",               44.22808,  3.99697),
    ("Liam's Minions",     "Biophony",    "Blackbird",                  44.227993, 3.996989),
    ("Liam's Minions",     "Biophony",    "Sparrow",                    44.227995, 3.99698),
    ("Liam's Minions",     "Biophony",    "Robin",                      44.228007, 3.996967),
    ("Liam's Minions",     "Biophony",    "Blackcap",                   44.228013, 3.996801),
    ("Liam's Minions",     "Biophony",    "Swift",                      44.228045, 3.996503),
    ("Liam's Minions",     "Anthrophony", "Traffic noise",              44.228089, 3.996477),
    ("Liam's Minions",     "Biophony",    "Blackcap",                   44.228071, 3.996456),
    ("Liam's Minions",     "Biophony",    "Cricket",                    44.228092, 3.996436),
    ("Liam's Minions",     "Biophony",    "Nightingale",                44.228077, 3.996446),
    ("Liam's Minions",     "Biophony",    "Blackcap",                   44.228005, 3.996111),
    ("Liam's Minions",     "Biophony",    "Robin",                      44.228008, 3.996123),
    ("Liam's Minions",     "Anthrophony", "Train",                      44.228066, 3.995989),
    ("Liam's Minions",     "Biophony",    "Blue tit",                   44.227931, 3.996186),
    ("Liam's Minions",     "Biophony",    "Robin",                      44.227918, 3.996177),
    ("Liam's Minions",     "Biophony",    "Great tit",                  44.228305, 3.996762),
    ("Liam's Minions",     "Biophony",    "Great tit",                  44.228664, 3.996724),
    ("Liam's Minions",     "Geophony",    "River",                      44.228698, 3.996875),
    ("Liam's Minions",     "Biophony",    "Cricket",                    44.228698, 3.997073),
    ("Liam's Minions",     "Biophony",    "Dogs barking",               44.228643, 3.997152),
    ("Liam's Minions",     "Geophony",    "Waterfall",                  44.228351, 3.998512),
    ("Liam's Minions",     "Anthrophony", "Gunshot",                    44.228153, 3.998177),
    ("Liam's Minions",     "Biophony",    "Crickets",                   44.227961, 3.997512),
    ("Liam's Minions",     "Anthrophony", "Human voices",               44.227953, 3.997486),

    # ── Big House ──────────────────────────────────────────────────────────
    ("Big House",          "Biophony",    "Nightingale",                44.227938, 3.997379),
    ("Big House",          "Anthrophony", "Footsteps",                  44.228129, 3.997059),
    ("Big House",          "Biophony",    "Blackcap",                   44.227969, 3.996918),
    ("Big House",          "Biophony",    "Robin",                      44.228023, 3.996551),
    ("Big House",          "Biophony",    "Nightingale",                44.228119, 3.996312),
    ("Big House",          "Anthrophony", "Train",                      44.2278,   3.996236),  # moved north from 44.227008
    ("Big House",          "Geophony",    "Wind in the leaves",         44.227991, 3.996184),
    ("Big House",          "Biophony",    "Blue tit",                   44.228169, 3.995933),
    ("Big House",          "Biophony",    "Crickets",                   44.228295, 3.996676),
    ("Big House",          "Anthrophony", "Traffic",                    44.228689, 3.996486),
    ("Big House",          "Biophony",    "Robin",                      44.228697, 3.996492),
    ("Big House",          "Biophony",    "Pipistrelle bat",            44.228698, 3.996885),
    ("Big House",          "Biophony",    "Dogs barking",               44.228674, 3.997128),
    ("Big House",          "Anthrophony", "Owl mimic",                  44.228315, 3.998042),
    ("Big House",          "Geophony",    "River over rocks",           44.228325, 3.998387),
    ("Big House",          "Anthrophony", "Gunshot!!!",                 44.228073, 3.9975),
    ("Big House",          "Biophony",    "Firecrest",                  44.227801, 3.996201),  # southern one kept
    # moved west from 3.998119

    # ── The Wavewalkers ────────────────────────────────────────────────────
    ("The Wavewalkers",    "Biophony",    "Song Thrush",                44.228102, 3.996842),
    ("The Wavewalkers",    "Biophony",    "Robin",                      44.228055, 3.996654),
    ("The Wavewalkers",    "Biophony",    "Treecreeper",                44.227889, 3.996521),  # southern one kept
    ("The Wavewalkers",    "Biophony",    "Blackcap",                   44.227844, 3.996398),
    ("The Wavewalkers",    "Biophony",    "Nightingale",                44.227763, 3.996055),
    ("The Wavewalkers",    "Biophony",    "Great tit",                  44.22782,  3.995901),
    ("The Wavewalkers",    "Biophony",    "Blue tit",                   44.22791,  3.995812),
    ("The Wavewalkers",    "Biophony",    "Swift",                      44.228031, 3.995743),
    ("The Wavewalkers",    "Biophony",    "Song Thrush",                44.22819,  3.995821),
    ("The Wavewalkers",    "Biophony",    "Cricket",                    44.228312, 3.995998),
    ("The Wavewalkers",    "Biophony",    "Blackbird",                  44.228441, 3.996187),
    ("The Wavewalkers",    "Geophony",    "Wind in the leaves",         44.22814,  3.99673),
    ("The Wavewalkers",    "Geophony",    "River",                      44.22785,  3.99631),
    ("The Wavewalkers",    "Geophony",    "Waterfall",                  44.22839,  3.99821),
    ("The Wavewalkers",    "Geophony",    "Rustling leaves",            44.22825,  3.99789),
    ("The Wavewalkers",    "Anthrophony", "Traffic noise",              44.228033, 3.99639),
    ("The Wavewalkers",    "Anthrophony", "Train",                      44.228112, 3.996155),
    ("The Wavewalkers",    "Anthrophony", "Voices",                     44.22848,  3.99731),
    ("The Wavewalkers",    "Anthrophony", "Gunshot",                    44.2282,   3.99809),

    # ── Added / edited ─────────────────────────────────────────────────────
    ("All groups",         "Anthrophony", "Loud noises from the terrasse", 44.2284, 3.9987),
    ("All groups",         "Anthrophony", "Car driving by",             44.2288,   3.996323),
]

# ── Styling ───────────────────────────────────────────────────────────────────
TYPE_COLORS  = {"Biophony": "#2e8b57", "Geophony": "#4a90d9", "Anthrophony": "#d94f3a"}
TYPE_MARKERS = {"Biophony": "o",       "Geophony": "s",       "Anthrophony": "^"}

# ── Convert to metres offsets from centre ─────────────────────────────────────
lats = [r[3] for r in data]
lons = [r[4] for r in data]
centre_lat = (min(lats) + max(lats)) / 2
centre_lon = (min(lons) + max(lons)) / 2

def to_m(lat, lon):
    dx = (lon - centre_lon) * math.cos(math.radians(centre_lat)) * 111320
    dy = (lat - centre_lat) * 111320
    return dx, dy

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.69, 8.27))   # A4 landscape inches
fig.patch.set_facecolor('#f7f4ef')
ax.set_facecolor('#e8f0e8')

# ── OSM background via contextily ─────────────────────────────────────────────
if HAS_CONTEXTILY:
    try:
        import numpy as np
        # Build extent in Web Mercator (EPSG:3857) for contextily
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        pad_lat = 0.0003
        pad_lon = 0.0004
        west, east = min(lons) - pad_lon, max(lons) + pad_lon
        south, north = min(lats) - pad_lat, max(lats) + pad_lat

        x_min, y_min = transformer.transform(west, south)
        x_max, y_max = transformer.transform(east, north)

        # Convert data points to metres-from-centre for axes, but
        # we need to set the axes extent to match the tile extent.
        # Easiest: use a secondary approach — plot on a geo axes.
        # Actually contextily works by adding tiles to an existing axes
        # that already has Web Mercator extent set.

        # Set axes limits in Web Mercator
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=17)

        # Now plot points in Web Mercator
        for grp, stype, species, lat, lon in data:
            mx, my = transformer.transform(lon, lat)
            ax.scatter(mx, my, c=TYPE_COLORS[stype], marker=TYPE_MARKERS[stype],
                       s=55, zorder=5, edgecolors='white', linewidths=0.6, alpha=0.92)
            ax.annotate(species, xy=(mx, my), xytext=(4, 4),
                        textcoords='offset points', fontsize=6, color='#1a1a1a',
                        zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))

        ax.set_xlabel("Latitude", fontsize=8, color='#444')
        ax.set_ylabel("Longitude", fontsize=8, color='#444')

    except Exception as e:
        print(f"contextily failed ({e}), falling back to plain axes.")
        HAS_CONTEXTILY = False

if not HAS_CONTEXTILY:
    # ── Plain axes fallback (no background tiles) ─────────────────────────
    ax.grid(True, color='white', linewidth=0.5, alpha=0.7, zorder=0)

    for grp, stype, species, lat, lon in data:
        x, y = to_m(lat, lon)
        ax.scatter(x, y, c=TYPE_COLORS[stype], marker=TYPE_MARKERS[stype],
                   s=75, zorder=5, edgecolors='white', linewidths=0.6, alpha=0.92)
        ax.annotate(species, xy=(x, y), xytext=(4, 4),
                    textcoords='offset points', fontsize=6, color='#1a1a1a',
                    zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))

    ax.set_xlabel("Metres (E–W)", fontsize=8, color='#444')
    ax.set_ylabel("Metres (N–S)", fontsize=8, color='#444')

ax.tick_params(labelsize=7, colors='#555')
for spine in ax.spines.values():
    spine.set_edgecolor('#bbb')

# ── Info box ──────────────────────────────────────────────────────────────────
info_lines = [
    (f"  Total detections:          {len(data)}", False),
    (f"  Biophony:    {sum(1 for r in data if r[1]=='Biophony')} detections", False),
    ("        13 bird and 1 bat species detected", False),
    ("        Most recorded: Robin (6)", False),
    ("        Bird detections higher near forest", False),
    (f"  Geophony:   {sum(1 for r in data if r[1]=='Geophony')} detections", False),
        ("        Greater diversity towards river (wind and water)", False),
    (f"  Anthrophony: {sum(1 for r in data if r[1]=='Anthrophony')} detections", False),
        ("        All groups detected gunshot at 21:42", False),
        ("Area covered: ~6 hectares", False),
        ("Survey duration: ~90 minutes", False),]

xlim = ax.get_xlim(); ylim = ax.get_ylim()
box_x = xlim[1] - (xlim[1]-xlim[0]) * 0.01
box_y = ylim[1] - (ylim[1]-ylim[0]) * 0.01

line_h = (ylim[1]-ylim[0]) * 0.028
char_w = (xlim[1]-xlim[0]) * 0.007
box_w  = char_w * 30
box_h  = line_h * len(info_lines) + line_h * 0.6

ax.add_patch(plt.Rectangle(
    (box_x - box_w, box_y - box_h),
    box_w, box_h,
    transform=ax.transData, zorder=9,
    facecolor='white', edgecolor='#888', linewidth=0.8, alpha=0.88
))

for i, (line, bold) in enumerate(info_lines):
    ax.text(box_x - box_w + char_w*0.4,
            box_y - line_h*(i+0.7),
            line,
            fontsize=5.5,
            fontweight='bold' if bold else 'normal',
            color='#1a1a1a',
            va='top', zorder=10,
        )
# ── North arrow ───────────────────────────────────────────────────────────────
xlim = ax.get_xlim(); ylim = ax.get_ylim()
xn = xlim[1] - (xlim[1]-xlim[0])*0.04
yn = ylim[1] - (ylim[1]-ylim[0])*0.08
ax.annotate('', xy=(xn, yn), xytext=(xn, yn - (ylim[1]-ylim[0])*0.06),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.text(xn, yn + (ylim[1]-ylim[0])*0.01, 'N', ha='center', va='bottom',
        fontsize=9, fontweight='bold', color='#333')

# ── Scale bar (~50 m) ─────────────────────────────────────────────────────────
if HAS_CONTEXTILY:
    # 50m in Web Mercator units (roughly constant at this latitude)
    scale_m = 50
else:
    scale_m = 50
sb_x = xlim[0] + (xlim[1]-xlim[0])*0.03
sb_y = ylim[0] + (ylim[1]-ylim[0])*0.04
ax.plot([sb_x, sb_x+scale_m], [sb_y, sb_y], color='#333', lw=2)
ax.plot([sb_x, sb_x],           [sb_y-2, sb_y+2], color='#333', lw=1.5)
ax.plot([sb_x+scale_m, sb_x+scale_m], [sb_y-2, sb_y+2], color='#333', lw=1.5)
ax.text(sb_x+scale_m/2, sb_y+4, '50 m', ha='center', fontsize=7, color='#333')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=TYPE_COLORS["Biophony"],
           markersize=8, label='Biophony',    markeredgecolor='#aaa', markeredgewidth=0.5),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=TYPE_COLORS["Geophony"],
           markersize=8, label='Geophony',    markeredgecolor='#aaa', markeredgewidth=0.5),
    Line2D([0],[0], marker='^', color='w', markerfacecolor=TYPE_COLORS["Anthrophony"],
           markersize=8, label='Anthrophony', markeredgecolor='#aaa', markeredgewidth=0.5),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
          framealpha=0.9, edgecolor='#bbb', title='Sound type', title_fontsize=8)

# ── Save to buffer then PDF ───────────────────────────────────────────────────
buf = io.BytesIO()
plt.tight_layout(pad=1.2)
plt.savefig(buf, format='png', dpi=220, bbox_inches='tight',
            facecolor=fig.get_facecolor())
buf.seek(0)
plt.close()

PAGE_W, PAGE_H = landscape(A4)
out_path = "C:\\Users\\liams\\Documents\\PhD-Project\\soundscape_map_final.pdf"
c = rl_canvas.Canvas(out_path, pagesize=landscape(A4))

HEADER_H = 32 * mm
FOOTER_H = 16 * mm

c.setFillColorRGB(0.12, 0.25, 0.18)
c.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
c.setFillColorRGB(1, 1, 1)
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(PAGE_W/2, PAGE_H - 16*mm, "TREES DLA ·  Chateau du Fraissinet")
c.setFont("Helvetica", 11)
c.drawCentredString(PAGE_W/2, PAGE_H - 25*mm, "Evening Soundscape Map // 13 May 2026")

map_bottom = FOOTER_H + 6*mm
map_top    = PAGE_H - HEADER_H - 6*mm
c.drawImage(ImageReader(buf), 10*mm, map_bottom,
            width=PAGE_W - 20*mm, height=map_top - map_bottom,
            preserveAspectRatio=True, anchor='c')

c.save()
print(f"Saved: {out_path}")