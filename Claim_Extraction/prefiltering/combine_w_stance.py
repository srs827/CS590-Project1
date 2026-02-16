import pandas as pd
# we combine stance labels with the claim labels, get all the data rows with a claim


# ----------------------------
# CONFIG: set your file paths
# ----------------------------
# meta path
DATASET1_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/prefiltering/meta_results.csv"
# bluesky path
DATASET2_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/prefiltering/bluesky_results.csv"

# proclimate path
TABLE1_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/proclimate.csv"
# proenergy path
TABLE2_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/proenergy.csv"
# neutral path
TABLE3_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/neutral.csv"
# full stance bluesky path
TABLE4_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/full_stance_bluesky.csv"
# full stance meta path
TABLE5_PATH = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/full_stance_meta.csv"

OUT_PATH = "claims_with_stance.csv"

# ----------------------------
# Helpers
# ----------------------------
def read_csv_str(path: str) -> pd.DataFrame:
    # dtype=str keeps IDs consistent (some look numeric, some are bafy... strings)
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def normalize_lookup(df: pd.DataFrame, cid_col: str, stance_col: str) -> pd.DataFrame:
    out = df[[cid_col, stance_col]].copy()
    out.columns = ["cid", "stance"]
    out["cid"] = out["cid"].astype(str).str.strip()
    out["stance"] = out["stance"].astype(str).str.strip()
    # drop empty stances
    out.loc[out["stance"] == "", "stance"] = pd.NA
    out = out.dropna(subset=["stance"])
    # if duplicates, keep first occurrence 
    out = out.drop_duplicates(subset=["cid"], keep="first")
    return out

def attach_stance_priority(base: pd.DataFrame, lookups: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Sequentially left-join stance from each lookup; fill stance only if missing.
    Priority order is the order of lookups in the list (earlier = higher priority).
    """
    out = base.copy()
    out["stance"] = pd.NA

    for i, lk in enumerate(lookups, start=1):
        out = out.merge(lk, on="cid", how="left", suffixes=("", f"_lk{i}"))
        # fill only where missing
        out["stance"] = out["stance"].fillna(out[f"stance_lk{i}"])
        out = out.drop(columns=[f"stance_lk{i}"])

    return out

# ----------------------------
# 1) Load datasets and filter to Claim
# ----------------------------
d1 = read_csv_str(DATASET1_PATH)
d2 = read_csv_str(DATASET2_PATH)

# Ensure required columns exist; if not, KeyError 
d1_claims = d1[d1["claim"].astype(str).str.strip() == "Claim"][["cid", "text", "claim"]].copy()
d2_claims = d2[d2["claim"].astype(str).str.strip() == "Claim"][["cid", "text", "claim"]].copy()

combined = pd.concat([d1_claims, d2_claims], ignore_index=True)
combined["cid"] = combined["cid"].astype(str).str.strip()

# de-dup combined by cid (keep first occurrence)
combined = combined.drop_duplicates(subset=["cid"], keep="first")

print(f"Combined Claim rows: {len(combined):,}")

# ----------------------------
# 2) Load and normalize lookup tables (cid -> stance)
# ----------------------------
t1 = read_csv_str(TABLE1_PATH)  # columns: cid,text,stance
t2 = read_csv_str(TABLE2_PATH)  # columns: cid,Stance,text
t3 = read_csv_str(TABLE3_PATH)  # columns: cid,text,Stance,...
t4 = read_csv_str(TABLE4_PATH)  # columns: ...,cid,...,predicted_label,...
t5 = read_csv_str(TABLE5_PATH)  # columns: cid,...,predicted_label,...

lk1 = normalize_lookup(t1, cid_col="cid", stance_col="stance")
lk2 = normalize_lookup(t2, cid_col="cid", stance_col="Stance")
lk3 = normalize_lookup(t3, cid_col="cid", stance_col="Stance")
lk4 = normalize_lookup(t4, cid_col="cid", stance_col="predicted_label")
lk5 = normalize_lookup(t5, cid_col="cid", stance_col="predicted_label")

# ----------------------------
# 3) Attach stance (priority: table1 > table2 > table3 > table4 > table5)
# ----------------------------
final = attach_stance_priority(combined, [lk1, lk2, lk3, lk4, lk5])

# ----------------------------
# 4) Report match stats
# ----------------------------
n_total = len(final)
n_matched = final["stance"].notna().sum()
n_unmatched = n_total - n_matched

print(f"Matched to a stance: {n_matched:,} / {n_total:,} ({(n_matched/n_total*100 if n_total else 0):.2f}%)")
print(f"Unmatched:          {n_unmatched:,}")

# stance dist
print("\nStance distribution (matched only):")
print(final.dropna(subset=["stance"])["stance"].value_counts(dropna=False))

# ----------------------------
# 5) Save output
# ----------------------------
final_out = final[["cid", "text", "claim", "stance"]].copy()
final_out.to_csv(OUT_PATH, index=False)
print(f"\nWrote: {OUT_PATH}")
