import os
import re
import regex as re2
from pathlib import Path
import umap

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import hdbscan

# =========================
# Config
# =========================
INPUT_CSV  = "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/prefiltering/claims_with_stance.csv"   
OUT_DIR    = "clustering_out"    

ID_COL     = "cid"
TEXT_COL   = "text"
CLAIM_COL  = "claim"                 
STANCE_COL = "stance"  # Pro-Energy / Neutral / Pro-Climate

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

# HDBSCAN hyperparams
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES      = 3   # None -> defaults to MIN_CLUSTER_SIZE 
CLUSTER_SELECTION_METHOD = "eom"      # "eom" or "leaf"
CLUSTER_SELECTION_EPSILON = 0.0

# Optional: text cleaning
DO_CLEAN_TEXT = True

USE_PCA = True
PCA_COMPONENTS = 100
TOP_K = 5

# =========================
# Text cleaning 
# =========================
URL_RE    = re.compile(r"http\\S+|www\\.\\S+")
SPACE_RE  = re.compile(r"\\s+")
EMOJI_RE  = re2.compile(r"\\p{Extended_Pictographic}")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s

# =========================
# Helpers
# =========================
def ensure_dir(p: str) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb.astype(np.float32, copy=False)

def run_hdbscan(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    embeddings: SBERT embeddings, already L2-normalized if normalize_embeddings=True.
    Returns: labels, probs, X_umap
    """
    X = embeddings

    # PCA (optional)
    if USE_PCA:
        n = X.shape[0]
        n_comp = min(PCA_COMPONENTS, X.shape[1], max(2, n - 1))
        pca = PCA(n_components=n_comp, random_state=42)
        X = pca.fit_transform(X)

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    X_umap = reducer.fit_transform(X)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        core_dist_n_jobs=-1,
        prediction_data=True, 
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
    )
    labels = clusterer.fit_predict(X_umap)
    probs  = clusterer.probabilities_.astype(np.float32)
    return labels, probs, X_umap

def cluster_stats(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n = int(labels.shape[0])
    n_out = int((labels == -1).sum())
    # clusters excluding -1
    uniq = set(labels.tolist())
    n_clusters = int(len(uniq) - (1 if -1 in uniq else 0))

    # cluster size distribution (exclude noise)
    sizes = pd.Series(labels[labels != -1]).value_counts()
    largest = int(sizes.iloc[0]) if len(sizes) else 0
    median = float(sizes.median()) if len(sizes) else 0.0

    return {
        "n_total": n,
        "n_outliers": n_out,
        "pct_outliers": (n_out / n) if n else 0.0,
        "n_clusters": n_clusters,
        "largest_cluster": largest,
        "median_cluster_size": median,
    }

# =========================
# Main
# =========================
def main():
    out_dir = ensure_dir(OUT_DIR)

    df = pd.read_csv(INPUT_CSV)
    all_stats = []
    # sanity checks
    for c in [ID_COL, TEXT_COL, CLAIM_COL, STANCE_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)}")

    # only Claim rows
    df = df[df[CLAIM_COL].astype(str).str.strip() == "Claim"].copy()

    # drop dup ids
    df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)

    # clean text + drop empties
    if DO_CLEAN_TEXT:
        df["_clean_text"] = df[TEXT_COL].map(clean_text)
        df = df[df["_clean_text"].str.len() > 0].reset_index(drop=True)
        text_for_embed_col = "_clean_text"
    else:
        df[TEXT_COL] = df[TEXT_COL].astype(str)
        df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
        text_for_embed_col = TEXT_COL

    print(f"Loaded {len(df)} claim rows for clustering.")

    model = SentenceTransformer(MODEL_NAME)

    # Prepare output columns
    df["cluster_id"] = -999
    df["cluster_prob"] = np.nan

    # Cluster stance-by-stance
    for stance, sdf in df.groupby(STANCE_COL, sort=True):
        stance_safe = re.sub(r"[^A-Za-z0-9_\\-]+", "_", str(stance))
        stance_dir = ensure_dir(out_dir / f"stance={stance_safe}")

        sdf = sdf.copy().reset_index()  # keep original row indices in df via "index"
        texts = sdf[text_for_embed_col].tolist()

        print("\\n" + "="*80)
        print(f"Stance: {stance} | n={len(sdf)}")
        print("="*80)

        # embed
        emb_path = stance_dir / "embeddings.npy"
        if emb_path.exists():
            print(f"Loading existing embeddings: {emb_path}")
            embeddings = np.load(emb_path)
        else:
            print(f"Embedding with {MODEL_NAME} ...")
            embeddings = embed_texts(model, texts)
            np.save(emb_path, embeddings)
            print(f"Saved embeddings: {emb_path} (shape={embeddings.shape})")

        # cluster
        labels, probs, X_umap = run_hdbscan(embeddings)

        st = cluster_stats(labels)
        st["stance"] = stance
        all_stats.append(st)

        print(f"Clusters found: {st['n_clusters']} | outliers (-1): {st['n_outliers']} ({st['pct_outliers']:.1%})")
        print(f"Largest cluster: {st['largest_cluster']} | median cluster size: {st['median_cluster_size']:.1f}")

       # write back to master df using original indices
        orig_idx = sdf["index"].to_numpy()
        
        # Make cluster IDs unique across stances
        # Convert to stance-specific format: "Neutral_0", "Pro-Climate_5", etc.
        unique_labels = np.array([f"{stance}_{lab}" if lab != -1 else "-1" 
                                  for lab in labels])
        
        df.loc[orig_idx, "cluster_id"] = unique_labels
        df.loc[orig_idx, "cluster_prob"] = probs

        # stance-level outputs for inspection
        stance_out = sdf.drop(columns=[text_for_embed_col], errors="ignore")
        stance_out["cluster_id"] = unique_labels  
        stance_out["cluster_prob"] = probs
        # cluster size summary (excluding noise)
        summary = (
            stance_out[stance_out["cluster_id"] != -1]
            .groupby("cluster_id")
            .size()
            .reset_index(name="size")
            .sort_values("size", ascending=False)
        )
        

        # top-k per cluster by probability
        work = stance_out[stance_out["cluster_id"] != -1].copy()
        topk = (
            work.sort_values(["cluster_id", "cluster_prob"], ascending=[True, False])
                .groupby("cluster_id", group_keys=False)
                .head(TOP_K)
                .assign(topk_rank=lambda x: x.groupby("cluster_id")["cluster_prob"]
                                    .rank(method="first", ascending=False).astype(int))
                .sort_values(["cluster_id", "topk_rank"])
                .reset_index(drop=True)
        )

        cols_pref = ["cluster_id", "topk_rank", ID_COL, "cluster_prob", TEXT_COL]
        cols_exist = [c for c in cols_pref if c in topk.columns]
        topk[cols_exist].to_csv(stance_dir / "topk_texts_by_cluster.csv", index=False)

    # final output
    stats_df = pd.DataFrame(all_stats).sort_values("stance")
    stats_path = out_dir / "clustering_stats_by_stance.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote: {stats_path}")
    summary.to_csv(stance_dir / "cluster_summary.csv", index=False)
    final_path = out_dir / "claims_clustered_all_stances.csv"
    df.drop(columns=["_clean_text"], errors="ignore").to_csv(final_path, index=False)
    print("\\nDone.")
    print(f"Wrote: {final_path}")

if __name__ == "__main__":
    main()
