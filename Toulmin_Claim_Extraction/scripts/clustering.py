#!/usr/bin/env python3
"""
Cluster Toulmin claims using the same clustering pipeline as:
Paid Voices vs. Public Feeds (arXiv:2601.13317v1)

Pipeline:
1) Embed claims with SBERT all-MiniLM-L6-v2
2) L2-normalize embeddings
3) PCA (default 100 dims) -> UMAP (default 20 dims)
4) HDBSCAN clustering
5) Per-cluster top-k representatives by membership probability (k=5)

Optional:
6) Merge clusters by cosine similarity threshold using SBERT all-mpnet-base-v2
   (paper uses this for summary similarity; here we do it on cluster centroids
    unless you provide your own summaries externally)

Inputs:
- CSV with a claim column
- JSONL where each row has {"claim": "..."} or nested fields you can map

Outputs:
- <out_prefix>.assignments.csv : per-claim cluster id + prob
- <out_prefix>.clusters.json   : cluster metadata + representatives
- (if --merge) <out_prefix>.merged_assignments.csv + merged clusters json

Example:
python cluster_claims.py \
  --input claims.csv --input-format csv --claim-col claim \
  --out-prefix claim_clusters \
  --batch-size 128 --pca-dim 100 --umap-dim 20 \
  --min-cluster-size 20 --min-samples 5 \
  --merge --merge-threshold 0.8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Core ML stack
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

import umap
import hdbscan


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def read_claims_csv(path: str, claim_col: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows_meta: List[Dict[str, Any]] = []
    claims: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if claim_col not in reader.fieldnames:
            raise ValueError(f"claim-col='{claim_col}' not found. Columns: {reader.fieldnames}")
        for r in reader:
            c = (r.get(claim_col) or "").strip()
            claims.append(c)
            rows_meta.append(r)
    return claims, rows_meta


def read_claims_jsonl(path: str, claim_key: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows_meta: List[Dict[str, Any]] = []
    claims: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # simple dotted key support: e.g. "toulmin.claim"
            cur: Any = obj
            for part in claim_key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            c = (cur or "").strip() if isinstance(cur, str) else ""
            claims.append(c)
            rows_meta.append(obj)
    return claims, rows_meta


def filter_claims(
    claims: List[str],
    meta: List[Dict[str, Any]],
    drop_not_present: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray]:
    keep_idx = []
    for i, c in enumerate(claims):
        if not c:
            continue
        if drop_not_present and c.strip().lower() == "not present":
            continue
        keep_idx.append(i)
    keep_idx_arr = np.array(keep_idx, dtype=int)
    return [claims[i] for i in keep_idx], [meta[i] for i in keep_idx], keep_idx_arr


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize: bool = False,
) -> np.ndarray:
    # sentence-transformers does efficient batching; choose device by model init
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,  # we do our own L2 normalize to be explicit
    )
    return emb.astype(np.float32, copy=False)


def pick_representatives(
    labels: np.ndarray,
    probs: np.ndarray,
    texts: List[str],
    top_k: int = 5,
) -> Dict[int, List[Tuple[int, float, str]]]:
    reps: Dict[int, List[Tuple[int, float, str]]] = {}
    for cid in np.unique(labels):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        # sort by membership probability, descending
        idx_sorted = idx[np.argsort(-probs[idx])]
        chosen = idx_sorted[:top_k]
        reps[cid] = [(int(i), float(probs[i]), texts[i]) for i in chosen]
    return reps


def save_assignments_csv(
    out_path: str,
    texts: List[str],
    orig_indices: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "claim", "cluster_id", "membership_prob"])
        for t, oi, lab, pr in zip(texts, orig_indices.tolist(), labels.tolist(), probs.tolist()):
            w.writerow([oi, t, lab, pr])


def save_clusters_json(
    out_path: str,
    labels: np.ndarray,
    probs: np.ndarray,
    texts: List[str],
    reps: Dict[int, List[Tuple[int, float, str]]],
) -> None:
    clusters = []
    for cid in sorted(reps.keys()):
        idx = np.where(labels == cid)[0]
        clusters.append(
            {
                "cluster_id": int(cid),
                "size": int(len(idx)),
                "avg_membership_prob": float(np.mean(probs[idx])) if len(idx) else 0.0,
                "representatives": [
                    {"local_index": int(i), "membership_prob": float(p), "claim": s}
                    for (i, p, s) in reps[cid]
                ],
            }
        )
    out = {
        "n_points": int(len(texts)),
        "n_clusters": int(len(reps)),
        "n_outliers": int(np.sum(labels < 0)),
        "clusters": clusters,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ---------- Optional merging (paper merges based on similarity threshold=0.8) ----------

def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    x = l2_normalize(x)
    return x @ x.T


def merge_clusters_by_centroid_similarity(
    labels: np.ndarray,
    emb_for_merge: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Compute centroid per cluster, then merge clusters whose centroid cosine similarity >= threshold.
    Returns new_labels and mapping old_cluster -> new_cluster.
    """
    cluster_ids = sorted([c for c in np.unique(labels) if c >= 0])
    if not cluster_ids:
        return labels.copy(), {}

    centroids = []
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        centroids.append(np.mean(emb_for_merge[idx], axis=0))
    centroids = np.asarray(centroids, dtype=np.float32)

    sim = cosine_sim_matrix(centroids)
    n = sim.shape[0]

    # union-find for merging
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    # compress and assign new ids
    root_to_new: Dict[int, int] = {}
    old_to_new: Dict[int, int] = {}
    next_id = 0
    for i, c in enumerate(cluster_ids):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = next_id
            next_id += 1
        old_to_new[c] = root_to_new[r]

    new_labels = labels.copy()
    for i in range(len(new_labels)):
        if new_labels[i] >= 0:
            new_labels[i] = old_to_new[int(new_labels[i])]

    return new_labels, old_to_new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV or JSONL containing claims")
    ap.add_argument("--input-format", choices=["csv", "jsonl"], required=True)
    ap.add_argument("--claim-col", default="claim", help="CSV column name for claim text")
    ap.add_argument("--claim-key", default="claim", help="JSONL key (supports dotted) for claim text")

    ap.add_argument("--out-prefix", required=True, help="Output prefix (no extension)")

    ap.add_argument("--drop-not-present", action="store_true", help='Drop claims exactly "Not present"')
    ap.add_argument("--batch-size", type=int, default=128)

    # Paper: all-MiniLM-L6-v2 + L2 norm + PCA + UMAP + HDBSCAN
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pca-dim", type=int, default=100)
    ap.add_argument("--umap-dim", type=int, default=20)
    ap.add_argument("--umap-n-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min-cluster-size", type=int, default=20)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)

    # Optional merging (paper uses SBERT all-mpnet-base-v2 + cosine sim threshold 0.8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--merge-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--merge-threshold", type=float, default=0.8)

    args = ap.parse_args()

    # ---------- Load ----------
    if args.input_format == "csv":
        claims, meta = read_claims_csv(args.input, args.claim_col)
    else:
        claims, meta = read_claims_jsonl(args.input, args.claim_key)

    claims_f, meta_f, keep_idx = filter_claims(
        claims, meta, drop_not_present=args.drop_not_present
    )
    if len(claims_f) == 0:
        raise ValueError("No claims left after filtering (empty or Not present).")

    # ---------- Embed ----------
    device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or True) else "cpu"
    # sentence-transformers auto-detects cuda if available
    embedder = SentenceTransformer(args.embed_model)
    X = embed_texts(embedder, claims_f, batch_size=args.batch_size, normalize=False)
    X = l2_normalize(X)  # paper: L2 normalization

    # ---------- Reduce (PCA -> UMAP) ----------
    pca_dim = min(args.pca_dim, X.shape[1], X.shape[0] - 1) if X.shape[0] > 1 else 1
    pca = PCA(n_components=max(1, pca_dim), random_state=args.seed)
    X_pca = pca.fit_transform(X)

    reducer = umap.UMAP(
        n_components=args.umap_dim,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    X_umap = reducer.fit_transform(X_pca)
    np.save(f"{args.out_prefix}.umap2d.npy", X_umap.astype(np.float32))

    # ---------- Cluster (HDBSCAN) ----------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X_umap)
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = np.ones_like(labels, dtype=np.float32)

    reps = pick_representatives(labels, probs, claims_f, top_k=args.top_k)

    # ---------- Save ----------
    base_assign_path = f"{args.out_prefix}.assignments.csv"
    base_clusters_path = f"{args.out_prefix}.clusters.json"
    save_assignments_csv(base_assign_path, claims_f, keep_idx, labels, probs)
    save_clusters_json(base_clusters_path, labels, probs, claims_f, reps)

    print(f"[OK] wrote {base_assign_path}")
    print(f"[OK] wrote {base_clusters_path}")

    # ---------- Optional merge ----------
    if args.merge:
        merge_embedder = SentenceTransformer(args.merge_model)
        X_merge = embed_texts(merge_embedder, claims_f, batch_size=args.batch_size, normalize=False)
        X_merge = l2_normalize(X_merge)

        merged_labels, mapping = merge_clusters_by_centroid_similarity(
            labels=labels,
            emb_for_merge=X_merge,
            threshold=args.merge_threshold,
        )

        merged_probs = probs  # keep original membership probs
        merged_reps = pick_representatives(merged_labels, merged_probs, claims_f, top_k=args.top_k)

        m_assign_path = f"{args.out_prefix}.merged_assignments.csv"
        m_clusters_path = f"{args.out_prefix}.merged_clusters.json"
        save_assignments_csv(m_assign_path, claims_f, keep_idx, merged_labels, merged_probs)
        save_clusters_json(m_clusters_path, merged_labels, merged_probs, claims_f, merged_reps)

        with open(f"{args.out_prefix}.merge_mapping.json", "w", encoding="utf-8") as f:
            json.dump({"old_cluster_to_new_cluster": mapping}, f, indent=2)

        print(f"[OK] wrote {m_assign_path}")
        print(f"[OK] wrote {m_clusters_path}")
        print(f"[OK] wrote {args.out_prefix}.merge_mapping.json")


if __name__ == "__main__":
    main()