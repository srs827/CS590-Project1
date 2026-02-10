#!/usr/bin/env python3
"""
Train on 500 Pro-Climate + 500 Pro-Energy + 150 Neutral sampled from labeled CSVs,
then classify a separate full dataset CSV with 3-class logistic regression,
excluding any rows whose cid overlaps with the training set.

Outputs:
- train_1150.csv
- full_stance_bluesky.csv
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# -------------------
# PATHS (your paths)
# -------------------
PROCLIMATE_CSV = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/proclimate.csv"
PROENERGY_CSV  = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/proenergy.csv"
NEUTRAL_CSV    = "/Users/skamanski/Documents/GitHub/CS590-Project1/Stance_Classification/training_data/neutral.csv"
DATASET        = "/Users/skamanski/Documents/GitHub/CS590-Project1/Full_Datasets/bluesky_data.csv"

# -------------------
# SETTINGS
# -------------------
N_PRO_CLIMATE = 500
N_PRO_ENERGY  = 500
N_NEUTRAL     = 150
RANDOM_SEED   = 42

OUT_TRAIN_CSV = "train_1150.csv"
OUT_PRED_CSV  = "full_stance_bluesky.csv"

CID_COL  = "cid"
TEXT_COL = "text"

LABELS_ALLOWED = {"Pro-Climate", "Pro-Energy", "Neutral"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stance column naming and enforce cid/text existence."""
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
    """Keep only allowed labels; drop rows missing cid/text/stance."""
    df = df.copy()
    df = df[df["stance"].isin(LABELS_ALLOWED)]
    return df


def main() -> int:
    # ---- Load labeled training sources
    df_pc = pd.read_csv(PROCLIMATE_CSV, dtype={CID_COL: "string"})
    df_pe = pd.read_csv(PROENERGY_CSV, dtype={CID_COL: "string"})
    df_ne = pd.read_csv(NEUTRAL_CSV,   dtype={CID_COL: "string"})

    df_pc = normalize_columns(df_pc)
    df_pe = normalize_columns(df_pe)
    df_ne = normalize_columns(df_ne)

    require_cols(df_pc, {CID_COL, TEXT_COL, "stance"}, "PROCLIMATE_CSV")
    require_cols(df_pe, {CID_COL, TEXT_COL, "stance"}, "PROENERGY_CSV")
    require_cols(df_ne, {CID_COL, TEXT_COL, "stance"}, "NEUTRAL_CSV")

    df_pc = filter_to_labels(df_pc)
    df_pe = filter_to_labels(df_pe)
    df_ne = filter_to_labels(df_ne)

    # ---- Sample training rows
    if len(df_pc) < N_PRO_CLIMATE:
        raise SystemExit(f"Not enough Pro-Climate rows to sample {N_PRO_CLIMATE}. Found {len(df_pc)}.")
    if len(df_pe) < N_PRO_ENERGY:
        raise SystemExit(f"Not enough Pro-Energy rows to sample {N_PRO_ENERGY}. Found {len(df_pe)}.")
    if len(df_ne) < N_NEUTRAL:
        raise SystemExit(f"Not enough Neutral rows to sample {N_NEUTRAL}. Found {len(df_ne)}.")

    df_pc_train = df_pc[df_pc["stance"] == "Pro-Climate"].sample(n=N_PRO_CLIMATE, random_state=RANDOM_SEED)
    df_pe_train = df_pe[df_pe["stance"] == "Pro-Energy"].sample(n=N_PRO_ENERGY, random_state=RANDOM_SEED)
    df_ne_train = df_ne[df_ne["stance"] == "Neutral"].sample(n=N_NEUTRAL, random_state=RANDOM_SEED)

    train_df = pd.concat([df_pc_train, df_pe_train, df_ne_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save training set
    train_df.to_csv(OUT_TRAIN_CSV, index=False)

    train_cids = set(train_df[CID_COL].dropna().astype("string"))

    # ---- Load full dataset to classify
    df_full = pd.read_csv(DATASET, dtype={CID_COL: "string"})
    df_full = normalize_columns(df_full)
    require_cols(df_full, {CID_COL, TEXT_COL}, "DATASET")

    # Exclude any rows whose cid appears in training
    before = len(df_full)
    df_full = df_full[~df_full[CID_COL].isin(train_cids)].reset_index(drop=True)
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

    model.fit(train_df[TEXT_COL].fillna(""), y_train)

    # ---- Predict on full dataset
    probs = model.predict_proba(df_full[TEXT_COL].fillna(""))
    pred_ids = model.predict(df_full[TEXT_COL].fillna(""))

    df_full["predicted_label"] = le.inverse_transform(pred_ids)

    # Add per-class probability columns (by class name)
    class_to_col = {cls: i for i, cls in enumerate(le.classes_)}
    if "Pro-Climate" in class_to_col:
        df_full["prob_pro_climate"] = probs[:, class_to_col["Pro-Climate"]]
    if "Pro-Energy" in class_to_col:
        df_full["prob_pro_energy"] = probs[:, class_to_col["Pro-Energy"]]
    if "Neutral" in class_to_col:
        df_full["prob_neutral"] = probs[:, class_to_col["Neutral"]]

    # Save predictions
    df_full.to_csv(OUT_PRED_CSV, index=False)

    # ---- Report
    print("=== DONE ===")
    print(f"Training set saved: {OUT_TRAIN_CSV}")
    print(f"  Train size: {len(train_df)} (Pro-Climate={N_PRO_CLIMATE}, Pro-Energy={N_PRO_ENERGY}, Neutral={N_NEUTRAL})")
    print(f"Full dataset loaded: {before} rows")
    print(f"Excluded due to train CID overlap: {before - after} rows")
    print(f"Predictions saved: {OUT_PRED_CSV}")
    print("Pred label counts:")
    print(df_full["predicted_label"].value_counts(dropna=False).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
