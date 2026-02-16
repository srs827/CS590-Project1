#!/usr/bin/env python3
"""
Train on 500 Pro-Climate + 500 Pro-Energy + 150 Neutral sampled from labeled CSVs,
then classify a separate full dataset CSV with 3-class logistic regression,
excluding any rows whose id overlaps with the training set ids.

Supports different full-dataset schemas by configuring:
  - FULL_ID_COL (e.g., "cid" or "ad_archive_id")
  - FULL_TEXT_COL (e.g., "text" or "ad_creative_bodies")

Outputs:
- train_1150.csv
- full_stance_<name>.csv (standardized to columns "cid" and "text")
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# -------------------
# TRAINING PATHS
# -------------------
PROCLIMATE_CSV = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/proclimate.csv"
PROENERGY_CSV  = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/proenergy.csv"
NEUTRAL_CSV    = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/neutral.csv"

# -------------------
# FULL DATASET PATH 
# -------------------
FULL_DATASET = "/Users/skamanski/Documents/GitHub/CS590-Project1/Full_Datasets/meta_data.csv"

# Toggle these depending on the full dataset you’re classifying:
#   Bluesky: FULL_ID_COL="cid", FULL_TEXT_COL="text"
#   Meta:    FULL_ID_COL="ad_archive_id", FULL_TEXT_COL="ad_creative_bodies"
FULL_ID_COL   = "ad_archive_id"
FULL_TEXT_COL = "ad_creative_bodies"

# -------------------
# SETTINGS
# -------------------
N_PRO_CLIMATE = 500
N_PRO_ENERGY  = 500
N_NEUTRAL     = 150
RANDOM_SEED   = 42

OUT_TRAIN_CSV = "train_1150_2.csv"
OUT_PRED_CSV  = "full_stance_meta.csv"

# Training files are expected to use these column names
TRAIN_ID_COL   = "cid"
TRAIN_TEXT_COL = "text"

LABELS_ALLOWED = {"Pro-Climate", "Pro-Energy", "Neutral"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stance column naming."""
    df = df.copy()
    if "Stance" in df.columns and "stance" not in df.columns:
        df.rename(columns={"Stance": "stance"}, inplace=True)
    return df


def require_cols(df: pd.DataFrame, cols: set, name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise SystemExit(
            f"{name} is missing required columns: {sorted(missing)}. Found: {list(df.columns)}"
        )


def filter_to_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only allowed labels."""
    df = df.copy()
    df = df[df["stance"].isin(LABELS_ALLOWED)]
    return df


def standardize_full_schema(df_full: pd.DataFrame, full_id_col: str, full_text_col: str) -> pd.DataFrame:
    """
    Rename the chosen id/text columns from the full dataset to following names:
      - "cid"
      - "text"
    """
    df_full = df_full.copy()

    # Ensure required cols exist before renaming
    require_cols(df_full, {full_id_col, full_text_col}, "FULL_DATASET")

    rename_map = {}
    if full_id_col != "cid":
        rename_map[full_id_col] = "cid"
    if full_text_col != "text":
        rename_map[full_text_col] = "text"

    if rename_map:
        df_full.rename(columns=rename_map, inplace=True)

    return df_full


def main() -> int:
    # ---- Load labeled training sources
    df_pc = pd.read_csv(PROCLIMATE_CSV, dtype={TRAIN_ID_COL: "string"})
    df_pe = pd.read_csv(PROENERGY_CSV, dtype={TRAIN_ID_COL: "string"})
    df_ne = pd.read_csv(NEUTRAL_CSV,   dtype={TRAIN_ID_COL: "string"})

    df_pc = normalize_columns(df_pc)
    df_pe = normalize_columns(df_pe)
    df_ne = normalize_columns(df_ne)

    require_cols(df_pc, {TRAIN_ID_COL, TRAIN_TEXT_COL, "stance"}, "PROCLIMATE_CSV")
    require_cols(df_pe, {TRAIN_ID_COL, TRAIN_TEXT_COL, "stance"}, "PROENERGY_CSV")
    require_cols(df_ne, {TRAIN_ID_COL, TRAIN_TEXT_COL, "stance"}, "NEUTRAL_CSV")

    df_pc = filter_to_labels(df_pc)
    df_pe = filter_to_labels(df_pe)
    df_ne = filter_to_labels(df_ne)

    # ---- Sample training rows by label 
    df_pc_only = df_pc[df_pc["stance"] == "Pro-Climate"]
    df_pe_only = df_pe[df_pe["stance"] == "Pro-Energy"]
    df_ne_only = df_ne[df_ne["stance"] == "Neutral"]

    if len(df_pc_only) < N_PRO_CLIMATE:
        raise SystemExit(f"Not enough Pro-Climate rows to sample {N_PRO_CLIMATE}. Found {len(df_pc_only)}.")
    if len(df_pe_only) < N_PRO_ENERGY:
        raise SystemExit(f"Not enough Pro-Energy rows to sample {N_PRO_ENERGY}. Found {len(df_pe_only)}.")
    if len(df_ne_only) < N_NEUTRAL:
        raise SystemExit(f"Not enough Neutral rows to sample {N_NEUTRAL}. Found {len(df_ne_only)}.")

    df_pc_train = df_pc_only.sample(n=N_PRO_CLIMATE, random_state=RANDOM_SEED)
    df_pe_train = df_pe_only.sample(n=N_PRO_ENERGY, random_state=RANDOM_SEED)
    df_ne_train = df_ne_only.sample(n=N_NEUTRAL, random_state=RANDOM_SEED)

    train_df = pd.concat([df_pc_train, df_pe_train, df_ne_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save training set
    train_df.to_csv(OUT_TRAIN_CSV, index=False)

    train_ids = set(train_df[TRAIN_ID_COL].dropna().astype("string"))

    # ---- Load full dataset to classify
    df_full_raw = pd.read_csv(FULL_DATASET, dtype={FULL_ID_COL: "string"})
    df_full = standardize_full_schema(df_full_raw, FULL_ID_COL, FULL_TEXT_COL)

    # Exclude overlaps (compare standardized df_full["cid"] to training ids)
    before = len(df_full)
    df_full = df_full[~df_full["cid"].isin(train_ids)].reset_index(drop=True)
    after = len(df_full)

    # ---- Train logistic regression model (3-class)
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["stance"])

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.95,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            multi_class="auto",
        ))
    ])

    model.fit(train_df[TRAIN_TEXT_COL].fillna(""), y_train)

    # ---- Predict on full dataset
    X_full = df_full["text"].fillna("")
    probs = model.predict_proba(X_full)
    pred_ids = model.predict(X_full)

    df_full["predicted_label"] = le.inverse_transform(pred_ids)

    # Add per-class probability columns
    class_to_col = {cls: i for i, cls in enumerate(le.classes_)}
    if "Pro-Climate" in class_to_col:
        df_full["prob_pro_climate"] = probs[:, class_to_col["Pro-Climate"]]
    if "Pro-Energy" in class_to_col:
        df_full["prob_pro_energy"] = probs[:, class_to_col["Pro-Energy"]]
    if "Neutral" in class_to_col:
        df_full["prob_neutral"] = probs[:, class_to_col["Neutral"]]

    # ---- Save predictions (with standardized "cid" + "text")
    df_full.to_csv(OUT_PRED_CSV, index=False)

    # ---- Report
    print("=== DONE ===")
    print(f"Training set saved: {OUT_TRAIN_CSV}")
    print(f"  Train size: {len(train_df)} (Pro-Climate={N_PRO_CLIMATE}, Pro-Energy={N_PRO_ENERGY}, Neutral={N_NEUTRAL})")
    print(f"Full dataset loaded: {before} rows")
    print(f"Excluded due to train ID overlap: {before - after} rows")
    print(f"Predictions saved: {OUT_PRED_CSV}")
    print("Pred label counts:")
    print(df_full["predicted_label"].value_counts(dropna=False).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
