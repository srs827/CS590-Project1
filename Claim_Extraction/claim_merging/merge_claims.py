# save as: merge_similar_cluster_claims.py
import os, json, math
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize


# -----------------------------
# Config (override via CLI)
# -----------------------------
IN_CSV          = "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/claim_generation/segmented_cluster_claims.csv"
OUT_DIR         = "merged_claims_out"
OUT_PREFIX      = "merged"

EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"
GRID_SPEC       = "0.60:0.90:0.02"   # cosine similarity threshold sweep
DEVICE          = None              # e.g., "cuda", "mps", "cpu", or None
LIMIT_PER_SEG   = None              # for debugging


# -----------------------------
# Utilities
# -----------------------------
def build_grid(spec: str):
    """
    Format: "start:stop:step" or list like "0.7,0.75,0.8".
    Returns list[float] of thresholds in [0,1].
    """
    spec = str(spec).strip()
    if "," in spec:
        vals = sorted(set(float(x) for x in spec.split(",") if x.strip()))
        return [v for v in vals if 0.0 <= v <= 1.0]
    a, b, c = [float(x) for x in spec.split(":")]
    n = max(1, int(round((b - a) / c)) + 1)
    arr = [a + i * c for i in range(n)]
    if arr[-1] < b - 1e-9:
        arr.append(b)
    return [min(1.0, max(0.0, v)) for v in arr]


def threshold_to_labels(sim: np.ndarray, thr: float):
    """
    Greedy merge policy:
    - Start a new group at i
    - Add any j > i with sim[i,j] >= thr (single-link to the seed i)
    """
    n = sim.shape[0]
    labels = -np.ones(n, dtype=int)
    gid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        members = [i]
        for j in range(i + 1, n):
            if labels[j] == -1 and sim[i, j] >= thr:
                members.append(j)
        for idx in members:
            labels[idx] = gid
        gid += 1
    return labels, gid


def eval_sil_dbi(E_unit: np.ndarray, labels: np.ndarray):
    """
    - silhouette (cosine) computed only on non-singletons, requires >= 2 non-singleton clusters
    - DBI computed if >= 2 clusters
    """
    u, count = np.unique(labels, return_counts=True)
    n_clusters = len(u)

    sil = np.nan
    try:
        non_single_mask = np.array([
            count[np.where(u == lb)[0][0]] > 1
            for lb in labels
        ])
        if (
            non_single_mask.sum() >= 2 and
            len(np.unique(labels[non_single_mask])) >= 2
        ):
            sil = silhouette_score(
                E_unit[non_single_mask],
                labels[non_single_mask],
                metric="cosine",
            )
    except Exception:
        pass

    dbi = np.nan
    try:
        if n_clusters >= 2:
            dbi = davies_bouldin_score(E_unit, labels)
    except Exception:
        pass

    return sil, dbi


def grid_search(E_unit: np.ndarray, grid: list[float]):
    """
    For each threshold:
      - create merged labels
      - compute silhouette and DBI
      - rank by normalized (sil up, dbi down)
    Returns: best_row_dict, ranked_df, label_cache
    """
    sim = E_unit @ E_unit.T  # cosine sim because unit vectors
    rows = []
    cache = {}

    for thr in grid:
        labels, _ = threshold_to_labels(sim, float(thr))
        sil, dbi = eval_sil_dbi(E_unit, labels)

        uniq, counts = np.unique(labels, return_counts=True)
        rows.append({
            "threshold": float(thr),
            "silhouette": sil,
            "dbi": dbi,
            "n_groups": int(len(uniq)),
            "n_singletons": int(np.sum(counts == 1)),
        })
        cache[float(thr)] = labels

    ranked = pd.DataFrame(rows)

    def minmax_up(s):
        v = s.dropna()
        if v.empty:
            return s * 0 + np.nan
        lo, hi = v.min(), v.max()
        mm = (v - lo) / (hi - lo) if hi > lo else v / v
        out = s * 0 + np.nan
        out.loc[mm.index] = mm
        return out

    def minmax_down(s):
        v = s.dropna()
        if v.empty:
            return s * 0 + np.nan
        lo, hi = v.min(), v.max()
        mm = (v - lo) / (hi - lo) if hi > lo else v / v
        out = s * 0 + np.nan
        out.loc[mm.index] = 1 - mm
        return out

    ranked["sil_norm"] = minmax_up(ranked["silhouette"])
    ranked["dbi_norm"] = minmax_down(ranked["dbi"])
    ranked["score"] = ranked["sil_norm"].fillna(0.0) + ranked["dbi_norm"].fillna(0.0)

    ranked = ranked.sort_values(["score", "silhouette"], ascending=[False, False]).reset_index(drop=True)
    best = ranked.iloc[0].to_dict()
    return best, ranked, cache


def pick_representatives(df_seg: pd.DataFrame, labels: np.ndarray):
    """
    Choose one representative cluster per merged group.

    Policy:
    - Keep the cluster with the lowest cluster_id lexicographically
    """
    tmp = df_seg[["segment", "cluster_id", "cluster_claim"]].copy()
    tmp["merged_id"] = labels

    reps = (
        tmp.loc[tmp.groupby("merged_id")["cluster_id"].idxmin()]
        .sort_values(["merged_id", "cluster_id"])
        .reset_index(drop=True)
    )
    keep_set = set(reps["cluster_id"].tolist())
    return reps, keep_set, tmp


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge semantically similar cluster claims within each segment.")
    parser.add_argument("--in_csv", type=str, default=IN_CSV)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--out_prefix", type=str, default=OUT_PREFIX)
    parser.add_argument("--embed_model", type=str, default=EMBED_MODEL)
    parser.add_argument("--grid", type=str, default=GRID_SPEC)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--limit_per_seg", type=int, default=0)
    args = parser.parse_args()

    in_csv = args.in_csv
    out_dir = args.out_dir
    out_prefix = args.out_prefix
    embed_model = args.embed_model
    grid_spec = args.grid
    device = args.device
    limit_per_seg = args.limit_per_seg if args.limit_per_seg and args.limit_per_seg > 0 else None

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_csv, dtype=str, keep_default_na=False)
    # Require these
    for col in ["segment", "cluster_id", "cluster_claim"]:
        if col not in df.columns:
            raise ValueError(f"{in_csv} missing required column: {col}")

    # Normalize
    df["segment"] = df["segment"].astype(str).str.strip()
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    df["cluster_claim"] = df["cluster_claim"].astype(str).str.strip()

    # Drop empty / errored claims
    if "error" in df.columns:
        df = df[df["error"].astype(str).str.strip() == ""].copy()
    df = df[df["cluster_claim"].astype(str).str.len() > 0].copy()

    # Load embedder once
    print(f"Loading embedder: {embed_model} on device={device}")
    model = SentenceTransformer(embed_model, device=device)

    grid = build_grid(grid_spec)

    all_kept = []
    all_removed = []
    all_maps = []
    best_rows = []

    for seg_name, df_seg in df.groupby("segment", sort=False):
        df_seg = df_seg.copy().reset_index(drop=True)

        if limit_per_seg is not None:
            df_seg = df_seg.head(limit_per_seg).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Segment: {seg_name} | claims={len(df_seg)}")
        print(f"{'='*60}")

        if len(df_seg) <= 1:
            # Nothing to merge
            df_seg_out = df_seg.copy()
            df_seg_out["merged_id"] = 0
            kept_path = os.path.join(out_dir, f"{out_prefix}__{seg_name}__kept.csv")
            df_seg_out.to_csv(kept_path, index=False)
            print(f"Only {len(df_seg)} claim(s); wrote {kept_path} (no merging)")
            all_kept.append(df_seg_out)
            continue

        texts = df_seg["cluster_claim"].astype(str).tolist()
        E = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        E_unit = normalize(E)

        best, ranked, cache = grid_search(E_unit, grid)

        sweep_path = os.path.join(out_dir, f"{out_prefix}__{seg_name}__threshold_sweep.csv")
        ranked.to_csv(sweep_path, index=False, float_format="%.6f")

        best_thr = float(best["threshold"])
        labels = cache[best_thr]

        print(
            f"Best threshold for {seg_name}: {best_thr:.3f} | "
            f"sil={best['silhouette']} | dbi={best['dbi']} | "
            f"groups={best['n_groups']} | singletons={best['n_singletons']}"
        )

        # cluster_id -> merged_id mapping
        map_df = pd.DataFrame({
            "segment": df_seg["segment"].to_numpy(),
            "cluster_id": df_seg["cluster_id"].to_numpy(),
            "merged_id": labels.astype(int),
            "cluster_claim": df_seg["cluster_claim"].to_numpy(),
        })
        map_path = os.path.join(out_dir, f"{out_prefix}__{seg_name}__cluster_to_merged.csv")
        map_df.to_csv(map_path, index=False)

        # representatives + pruned set
        reps_df, keep_set, tmp_df = pick_representatives(df_seg, labels)

        kept_df = df_seg[df_seg["cluster_id"].isin(keep_set)].copy()
        kept_df = kept_df.sort_values("cluster_id").reset_index(drop=True)

        removed_df = df_seg[~df_seg["cluster_id"].isin(keep_set)].copy()
        removed_df = removed_df.sort_values("cluster_id").reset_index(drop=True)

        # write per-segment outputs
        kept_path = os.path.join(out_dir, f"{out_prefix}__{seg_name}__kept.csv")
        removed_path = os.path.join(out_dir, f"{out_prefix}__{seg_name}__removed.csv")
        reps_path = os.path.join(out_dir, f"{out_prefix}__{seg_name}__representatives.csv")

        kept_df.to_csv(kept_path, index=False)
        removed_df[["segment", "cluster_id", "cluster_claim"]].to_csv(removed_path, index=False)
        reps_df.to_csv(reps_path, index=False)

        # accumulate global outputs
        kept_df = kept_df.merge(
            map_df[["segment", "cluster_id", "merged_id"]],
            on=["segment", "cluster_id"],
            how="left",
        )
        all_kept.append(kept_df)
        all_removed.append(removed_df)
        all_maps.append(map_df)

        best_rows.append({
            "segment": seg_name,
            **best
        })

    # Global combined outputs
    if all_maps:
        comb_map = pd.concat(all_maps, ignore_index=True)
        comb_map_path = os.path.join(out_dir, f"{out_prefix}__ALL__cluster_to_merged.csv")
        comb_map.to_csv(comb_map_path, index=False)
        print(f"\nWrote: {comb_map_path}")

    if all_kept:
        comb_kept = pd.concat(all_kept, ignore_index=True)
        comb_kept_path = os.path.join(out_dir, f"{out_prefix}__ALL__kept.csv")
        comb_kept.to_csv(comb_kept_path, index=False)
        print(f"Wrote: {comb_kept_path}")

    if all_removed:
        comb_removed = pd.concat(all_removed, ignore_index=True)
        comb_removed_path = os.path.join(out_dir, f"{out_prefix}__ALL__removed.csv")
        comb_removed.to_csv(comb_removed_path, index=False)
        print(f"Wrote: {comb_removed_path}")

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        best_path = os.path.join(out_dir, f"{out_prefix}__ALL__best_thresholds.csv")
        best_df.to_csv(best_path, index=False, float_format="%.6f")
        print(f"Wrote: {best_path}")


if __name__ == "__main__":
    main()
