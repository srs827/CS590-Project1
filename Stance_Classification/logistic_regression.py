#!/usr/bin/env python3
"""
Train on 200 Pro-Climate + 200 Pro-Energy sampled from two labeled CSVs,
then classify a separate full dataset CSV with logistic regression,
excluding any rows whose cid overlaps with the training set.

Outputs:
- train_400.csv
- dataset_no_train_cids_predictions.csv
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# -------------------
# PATHS (your paths)
# -------------------
PROCLIMATE_CSV = "/Users/skamanski/Desktop/cs590/sem_2/stance_extraction/meta_stance/proclimate.csv"
PROENERGY_CSV  = "/Users/skamanski/Desktop/cs590/sem_2/stance_extraction/meta_stance/proenergy.csv"
DATASET        = "/Users/skamanski/Desktop/cs590/sem_1/pipeline/preprocessing/bluesky_preprocessing/combined_deduped_sbert080.csv"

# -------------------
# SETTINGS
# -------------------
N_PER_CLASS = 200
RANDOM_SEED = 42

OUT_TRAIN_CSV = "/Users/skamanski/Desktop/cs590/sem_2/stance_extraction/meta_stance/train_400.csv"
OUT_PRED_CSV  = "/Users/skamanski/Desktop/cs590/sem_2/stance_extraction/meta_stance/dataset_no_train_cids_predictions.csv"

CID_COL = "cid"
TEXT_COL = "text"

LABELS_ALLOWED = {"Pro-Climate", "Pro-Energy"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stance column naming and enforce cid/text existence."""
    df = df.copy()
    if "Stance" in df.columns and "stance" not in df.columns:
        df.rename(columns={"Stance": "stance"}, inplace=True)
    return df


def require_cols(df: pd.DataFrame, cols: set, name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise SystemExit(f"{name} is missing required columns: {sorted(missing)}. Found: {list(df.columns)}")


def main() -> int:
    # ---- Load labeled training sources
    df_pc = pd.read_csv(PROCLIMATE_CSV, dtype={CID_COL: "string"})
    df_pe = pd.read_csv(PROENERGY_CSV, dtype={CID_COL: "string"})

    df_pc = normalize_columns(df_pc)
    df_pe = normalize_columns(df_pe)

    require_cols(df_pc, {CID_COL, TEXT_COL, "stance"}, "PROCLIMATE_CSV")
    require_cols(df_pe, {CID_COL, TEXT_COL, "stance"}, "PROENERGY_CSV")

    # keep only the two labels (in case file contains noise)
    df_pc = df_pc[df_pc["stance"].isin(LABELS_ALLOWED)].copy()
    df_pe = df_pe[df_pe["stance"].isin(LABELS_ALLOWED)].copy()

    # sample training rows
    if len(df_pc) < N_PER_CLASS:
        raise SystemExit(f"Not enough Pro-Climate rows to sample {N_PER_CLASS}. Found {len(df_pc)}.")
    if len(df_pe) < N_PER_CLASS:
        raise SystemExit(f"Not enough Pro-Energy rows to sample {N_PER_CLASS}. Found {len(df_pe)}.")

    df_pc_train = df_pc.sample(n=N_PER_CLASS, random_state=RANDOM_SEED)
    df_pe_train = df_pe.sample(n=N_PER_CLASS, random_state=RANDOM_SEED)

    train_df = pd.concat([df_pc_train, df_pe_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # save training set 
    train_df.to_csv(OUT_TRAIN_CSV, index=False)

    train_cids = set(train_df[CID_COL].dropna().astype("string"))

    # ---- Load full dataset to classify
    df_full = pd.read_csv(DATASET, dtype={CID_COL: "string"})
    df_full = normalize_columns(df_full)

    require_cols(df_full, {CID_COL, TEXT_COL}, "DATASET")

    # exclude any rows whose cid appears in training
    before = len(df_full)
    df_full = df_full[~df_full[CID_COL].isin(train_cids)].reset_index(drop=True)
    after = len(df_full)

    # ---- Train logistic regression model
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
        ))
    ])

    model.fit(train_df[TEXT_COL].fillna(""), y_train)

    # ---- Predict on full dataset
    probs = model.predict_proba(df_full[TEXT_COL].fillna(""))
    pred_ids = model.predict(df_full[TEXT_COL].fillna(""))

    df_full["predicted_label"] = le.inverse_transform(pred_ids)

    # add per-class probability columns (by class name)
    class_to_col = {cls: i for i, cls in enumerate(le.classes_)}
    if "Pro-Climate" in class_to_col:
        df_full["prob_pro_climate"] = probs[:, class_to_col["Pro-Climate"]]
    if "Pro-Energy" in class_to_col:
        df_full["prob_pro_energy"] = probs[:, class_to_col["Pro-Energy"]]

    df_full.to_csv(OUT_PRED_CSV, index=False)

    # ---- Report
    print("=== DONE ===")
    print(f"Training set saved: {OUT_TRAIN_CSV}")
    print(f"  Train size: {len(train_df)} ({N_PER_CLASS} per class)")
    print(f"Full dataset loaded: {before} rows")
    print(f"Excluded due to train CID overlap: {before - after} rows")
    print(f"Predictions saved: {OUT_PRED_CSV}")
    print("Pred label counts:")
    print(df_full["predicted_label"].value_counts(dropna=False).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
