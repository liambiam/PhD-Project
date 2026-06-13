"""
Three-way reconciliation of Tanzanian plant checklists: RAINBIO vs Kew vs GBIF.

The whole point: a RAW name comparison massively overstates differences because
of synonyms, author strings, and formatting. This script resolves every name to
a single backbone (the GBIF Backbone Taxonomy via pygbif) and compares on the
RESOLVED accepted name, then breaks the residual differences down by match type
so you can separate real biological differences from name-matching artefacts.

Outputs (written to OUTDIR):
  - resolved_names.csv        : every input name + its resolved accepted name + match flags
  - reconciliation_crosstab.csv : presence/absence of each accepted species in the 3 sources
  - match_type_summary.csv    : counts of exact / fuzzy / synonym / unmatched per source
  - venn_counts printed to console

Requires network access at runtime (pygbif queries the GBIF API). Name lookups
are CACHED to name_cache.json so reruns are fast and you only pay the lookup once.

CONFIG flagged inline.
"""

import os
import re
import json
import time
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG  -- adjust paths if needed
RAINBIO_CSV = r"C:\Users\liams\Documents\PhD-Project Data\tanzania\tanzania_points_with_habitat_labels.csv"
GBIF_CSV    = r"C:\Users\liams\Documents\PhD-Project Data\GBIF Tanzania\0030978-260519110011954.csv"
KEW_XLSX    = r"C:\Users\liams\Documents\PhD-Project Data\Kew\List of species from Tanzania.xlsx"
OUTDIR      = r"C:\Users\liams\Documents\PhD-Project Data\Reconciliation"
CACHE_PATH  = os.path.join(OUTDIR, "name_cache.json")

# Column holding the species binomial in each source
RAINBIO_NAME_COL = "species"       # e.g. "Acanthopale confertiflora"
GBIF_NAME_COL    = "species"       # GBIF 'species' = binomial (no authorship); cleaner than scientificName
KEW_NAME_COL     = "taxon_name"    # e.g. "Acanthopale confertiflora"

GBIF_SEP = "\t"   # GBIF 'simple' CSV download is TAB-separated; change to "," if comma
# ----------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------------------
# Name cleaning (light) before backbone lookup
# ----------------------------------------------------------------------
def clean_name(name):
    if pd.isna(name):
        return None
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)                 # collapse whitespace
    s = re.sub(r"\b(cf|aff)\.?\s+", "", s, flags=re.I)  # drop cf./aff. qualifiers
    s = s.replace("×", "x")                    # hybrid marker
    return s or None


# ----------------------------------------------------------------------
# GBIF backbone resolution (cached)
# ----------------------------------------------------------------------
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def resolve_names(names, cache):
    """Resolve a set of cleaned names against the GBIF backbone.
    Returns dict: input_name -> {accepted, rank, status, matchType, confidence}."""
    from pygbif import species as gb_species
    out = {}
    todo = [n for n in names if n and n not in cache]
    print(f"  {len(names)} unique names; {len(todo)} need lookup "
          f"({len(names) - len(todo)} cached)")
    for i, n in enumerate(todo, 1):
        try:
            r = gb_species.name_backbone(name=n, kingdom="Plantae", strict=False)
            cache[n] = {
                "accepted": r.get("species") or r.get("canonicalName") or r.get("scientificName"),
                "acceptedKey": r.get("speciesKey") or r.get("usageKey"),
                "rank": r.get("rank"),
                "status": r.get("status"),
                "matchType": r.get("matchType"),   # EXACT / FUZZY / HIGHERRANK / NONE
                "confidence": r.get("confidence"),
            }
        except Exception as e:
            cache[n] = {"accepted": None, "matchType": "ERROR", "status": str(e)[:80]}
        if i % 200 == 0:
            print(f"    {i}/{len(todo)} resolved...")
            save_cache(cache)
            time.sleep(0.2)  # be polite to the API
    save_cache(cache)
    for n in names:
        out[n] = cache.get(n, {"accepted": None, "matchType": "NONE"})
    return out


# ----------------------------------------------------------------------
# Load the three source species lists
# ----------------------------------------------------------------------
def load_sources():
    rb = pd.read_csv(RAINBIO_CSV)
    gb = pd.read_csv(GBIF_CSV, sep=GBIF_SEP, low_memory=False)
    kw = pd.read_excel(KEW_XLSX)

    rb_names = set(rb[RAINBIO_NAME_COL].dropna().map(clean_name)) - {None}
    gb_names = set(gb[GBIF_NAME_COL].dropna().map(clean_name)) - {None}
    kw_names = set(kw[KEW_NAME_COL].dropna().map(clean_name)) - {None}
    print(f"  RAINBIO: {len(rb_names)} unique names")
    print(f"  GBIF   : {len(gb_names)} unique names")
    print(f"  Kew    : {len(kw_names)} unique names")
    return rb_names, gb_names, kw_names


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Loading source lists...")
    rb_names, gb_names, kw_names = load_sources()
    all_names = rb_names | gb_names | kw_names
    print(f"  {len(all_names)} unique raw names across all three")

    print("Resolving to GBIF backbone (cached)...")
    cache = load_cache()
    resolved = resolve_names(all_names, cache)

    # raw-name -> accepted-name map
    def accepted_of(nameset):
        acc = {}
        for n in nameset:
            a = resolved.get(n, {}).get("accepted")
            if a:
                acc[a] = acc.get(a, 0) + 1
        return set(acc.keys())

    rb_acc = accepted_of(rb_names)
    gb_acc = accepted_of(gb_names)
    kw_acc = accepted_of(kw_names)

    # ---- per-name resolution table
    rows = []
    for src, nameset in [("RAINBIO", rb_names), ("GBIF", gb_names), ("Kew", kw_names)]:
        for n in nameset:
            info = resolved.get(n, {})
            rows.append({
                "source": src, "input_name": n,
                "accepted_name": info.get("accepted"),
                "match_type": info.get("matchType"),
                "status": info.get("status"),
                "confidence": info.get("confidence"),
            })
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUTDIR, "resolved_names.csv"), index=False)

    # ---- match-type summary
    summ = (res_df.groupby(["source", "match_type"]).size()
            .unstack(fill_value=0))
    summ.to_csv(os.path.join(OUTDIR, "match_type_summary.csv"))
    print("\n=== Match-type breakdown (per source) ===")
    print(summ)

    # ---- three-way crosstab on ACCEPTED names
    all_acc = rb_acc | gb_acc | kw_acc
    cross = pd.DataFrame({
        "accepted_name": sorted(all_acc),
    })
    cross["in_RAINBIO"] = cross["accepted_name"].isin(rb_acc)
    cross["in_GBIF"]    = cross["accepted_name"].isin(gb_acc)
    cross["in_Kew"]     = cross["accepted_name"].isin(kw_acc)
    cross.to_csv(os.path.join(OUTDIR, "reconciliation_crosstab.csv"), index=False)

    # ---- before/after comparison (raw vs resolved overlap)
    raw_all = len(rb_names | gb_names | kw_names)
    raw_shared3 = len(rb_names & gb_names & kw_names)
    acc_shared3 = len(rb_acc & gb_acc & kw_acc)

    print("\n=== Overlap: RAW names vs RESOLVED accepted names ===")
    print(f"  Unique RAW names total       : {raw_all}")
    print(f"  Shared by all 3 (RAW)        : {raw_shared3}")
    print(f"  Unique ACCEPTED species total: {len(all_acc)}")
    print(f"  Shared by all 3 (ACCEPTED)   : {acc_shared3}")

    print("\n=== Three-way membership (accepted names) ===")
    combo = (cross.groupby(["in_RAINBIO", "in_GBIF", "in_Kew"]).size()
             .reset_index(name="n_species")
             .sort_values("n_species", ascending=False))
    print(combo.to_string(index=False))

    print(f"\nWritten to: {OUTDIR}")
    print("  resolved_names.csv, reconciliation_crosstab.csv, match_type_summary.csv")


if __name__ == "__main__":
    main()
