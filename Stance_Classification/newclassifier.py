#!/usr/bin/env python3
"""
Train on 500 Pro-Climate + 500 Pro-Energy + 150 Neutral sampled from labeled CSVs,
then classify a separate full dataset CSV with 3-class classification,
excluding any rows whose id overlaps with the training set ids.
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib  # for saving/loading models

# -------------------
# TRAINING PATHS
# -------------------
PROCLIMATE_CSV = "/Users/edwardwang/Downloads/CS590-Project1-main/Stance_Classification/training_data/proclimate.csv"
PROENERGY_CSV = "/Users/edwardwang/Downloads/CS590-Project1-main/Stance_Classification/training_data/proenergy.csv"
NEUTRAL_CSV = "/Users/edwardwang/Downloads/CS590-Project1-main/Stance_Classification/training_data/neutral.csv"

# -------------------
# FULL DATASET PATHS (YOU CAN RUN MULTIPLE)
# -------------------
FULL_DATASETS = {
    'meta': {
        'path': "/Users/edwardwang/Downloads/CS590-Project1-main/Full_Datasets/meta_data.csv",
        'id_col': 'ad_archive_id',
        'text_col': 'ad_creative_bodies'
    },
    'bluesky': {
        'path': "/Users/edwardwang/Downloads/CS590-Project1-main/Full_Datasets/bluesky_data.csv",
        'id_col': 'cid',
        'text_col': 'text'
    }
    # Add more datasets as needed
}

# -------------------
# SETTINGS
# -------------------
# Choose your best model: 'random_forest' or 'logistic_regression'
BEST_MODEL = 'random_forest'  # Change to 'logistic_regression' if that performed better

N_PRO_CLIMATE = 500
N_PRO_ENERGY = 500
N_NEUTRAL = 150
RANDOM_SEED = 42

# Output files
OUT_TRAIN_CSV = "train_1150.csv"
OUT_MODEL_FILE = f"best_model_{BEST_MODEL}.pkl"

# Training files column names
TRAIN_ID_COL = "cid"
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
        raise SystemExit(f"{name} is missing required columns: {sorted(missing)}")


def filter_to_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only allowed labels."""
    return df[df["stance"].isin(LABELS_ALLOWED)].copy()


def create_model(model_type='random_forest'):
    """Create a pipeline with the specified model"""
    # RANDOM FOREST FOUND TO BE MOST ACCURATE;
    # HOWEVER, LOGISTIC REGRESSION IS NOT FAR BEHIND IN CV ACCURACY
    # classifier able to run both
    tfidf_params = {
        'max_features': 5000, #7000 for logistic regression
        'ngram_range': (1, 2),
        'min_df': 3, #2 for log regression
        'max_df': 0.95,
        'stop_words': 'english'
    }

    if model_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    elif model_type == 'logistic_regression':
        classifier = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', classifier)
    ])

    return pipeline


def load_and_sample_training_data():
    """Load and sample the training data"""

    # Load labeled training sources
    df_pc = pd.read_csv(PROCLIMATE_CSV, dtype={TRAIN_ID_COL: "string"})
    df_pe = pd.read_csv(PROENERGY_CSV, dtype={TRAIN_ID_COL: "string"})
    df_ne = pd.read_csv(NEUTRAL_CSV, dtype={TRAIN_ID_COL: "string"})

    # Normalize and filter
    df_pc = filter_to_labels(normalize_columns(df_pc))
    df_pe = filter_to_labels(normalize_columns(df_pe))
    df_ne = filter_to_labels(normalize_columns(df_ne))

    # Sample by label
    df_pc_train = df_pc[df_pc["stance"] == "Pro-Climate"].sample(n=N_PRO_CLIMATE, random_state=RANDOM_SEED)
    df_pe_train = df_pe[df_pe["stance"] == "Pro-Energy"].sample(n=N_PRO_ENERGY, random_state=RANDOM_SEED)
    df_ne_train = df_ne[df_ne["stance"] == "Neutral"].sample(n=N_NEUTRAL, random_state=RANDOM_SEED)

    # Combine and shuffle
    train_df = pd.concat([df_pc_train, df_pe_train, df_ne_train], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return train_df


def train_model(train_df):
    """Train the model on the sampled data"""

    print(f"\nTraining {BEST_MODEL} on {len(train_df)} samples...")

    # Prepare features and labels
    X_train = train_df[TRAIN_TEXT_COL].fillna("")
    y_train = train_df["stance"]

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Create and train model
    model = create_model(BEST_MODEL)
    model.fit(X_train, y_train_encoded)

    # Save model and label encoder
    joblib.dump({'model': model, 'label_encoder': le}, OUT_MODEL_FILE)
    print(f"Model saved to {OUT_MODEL_FILE}")

    return model, le


def classify_full_dataset(full_dataset_config, train_ids, model, le):
    """Classify a full dataset and save results"""

    print(f"\nProcessing {full_dataset_config['path']}...")

    # Load full dataset
    df_full = pd.read_csv(
        full_dataset_config['path'],
        dtype={full_dataset_config['id_col']: "string"}
    )

    # Standardize column names
    df_full = df_full.rename(columns={
        full_dataset_config['id_col']: 'cid',
        full_dataset_config['text_col']: 'text'
    })

    # Save original row count
    original_count = len(df_full)

    # Exclude training IDs
    df_full = df_full[~df_full['cid'].isin(train_ids)].reset_index(drop=True)
    excluded_count = original_count - len(df_full)

    # Prepare text data
    X_full = df_full['text'].fillna("")

    # Make predictions
    predictions = model.predict(X_full)
    probabilities = model.predict_proba(X_full)

    # Add predictions to dataframe
    df_full['predicted_label'] = le.inverse_transform(predictions)

    # Add probability columns
    for i, class_name in enumerate(le.classes_):
        df_full[f'prob_{class_name.lower().replace("-", "_")}'] = probabilities[:, i]

    # Generate output filename
    dataset_name = full_dataset_config['path'].split('/')[-1].replace('.csv', '')
    output_file = f"classified_{dataset_name}_{BEST_MODEL}.csv"

    # Save results
    df_full.to_csv(output_file, index=False)

    # Print summary
    print(f"  Original rows: {original_count}")
    print(f"  Excluded (train overlap): {excluded_count}")
    print(f"  Classified: {len(df_full)}")
    print(f"  Saved to: {output_file}")
    print("\n  Label distribution:")
    print(df_full['predicted_label'].value_counts())

    return df_full


def main():
    # Step 1: Load and sample training data
    print("=" * 50)
    print("STEP 1: Loading and sampling training data")
    print("=" * 50)
    train_df = load_and_sample_training_data()

    # Save training set
    train_df.to_csv(OUT_TRAIN_CSV, index=False)
    print(f"Training data saved to {OUT_TRAIN_CSV}")
    print(f"Training set composition:")
    print(train_df['stance'].value_counts())

    # Get training IDs for exclusion
    train_ids = set(train_df[TRAIN_ID_COL].dropna().astype("string"))

    # Step 2: Train model
    print("\n" + "=" * 50)
    print("STEP 2: Training model")
    print("=" * 50)
    model, le = train_model(train_df)

    # Step 3: Classify all full datasets
    print("\n" + "=" * 50)
    print("STEP 3: Classifying full datasets")
    print("=" * 50)

    for dataset_name, config in FULL_DATASETS.items():
        classify_full_dataset(config, train_ids, model, le)



    print("\n" + "=" * 50)
    print("ALL DONE! Check the output files:")
    print(f"  - {OUT_TRAIN_CSV} (training sample)")
    print(f"  - {OUT_MODEL_FILE} (saved model)")
    for dataset_name in FULL_DATASETS.keys():
        print(f"  - classified_{dataset_name}_{BEST_MODEL}.csv")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())