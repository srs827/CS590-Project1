#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_assignments(assign_csv: str) -> pd.DataFrame:
    df = pd.read_csv(assign_csv)
    required = {"row_index", "claim", "cluster_id", "membership_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {assign_csv}: {missing}")
    return df


def load_clusters_json(clusters_json: str) -> dict:
    with open(clusters_json, "r", encoding="utf-8") as f:
        return json.load(f)


def load_umap2d(umap_path: str, n_rows: int) -> np.ndarray:
    X = np.load(umap_path)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"UMAP array must be (N,2+) but got {X.shape}")
    if X.shape[0] != n_rows:
        raise ValueError(f"UMAP rows {X.shape[0]} != assignments rows {n_rows}.")
    return X[:, :2]


def plot_umap_scatter(
    df: pd.DataFrame,
    xy: np.ndarray,
    title: str,
    out_path: Optional[str] = None,
    max_points: int = 200_000,
    point_size: float = 3.0,
    alpha: float = 0.5,
    show_legend: bool = False,
):
    """
    Efficient scatter for big N:
    - Optionally subsample to max_points
    - Outliers cluster_id == -1 in gray
    """
    if len(df) != xy.shape[0]:
        raise ValueError("df and xy length mismatch")

    # subsample if needed (uniform)
    if len(df) > max_points:
        idx = np.random.RandomState(0).choice(len(df), size=max_points, replace=False)
        dfp = df.iloc[idx].reset_index(drop=True)
        xyp = xy[idx]
    else:
        dfp = df.reset_index(drop=True)
        xyp = xy

    labels = dfp["cluster_id"].to_numpy()
    outliers = labels < 0
    inliers = ~outliers

    fig = plt.figure()
    ax = plt.gca()

    # outliers first (gray)
    if np.any(outliers):
        ax.scatter(
            xyp[outliers, 0],
            xyp[outliers, 1],
            s=point_size,
            alpha=min(alpha, 0.35),
        )

    # inliers colored by cluster
    # Matplotlib will auto-cycle colors; for many clusters it's still fine visually.
    # For speed, we plot cluster by cluster only for "big" clusters; else, use one call with colormap.
    if np.any(inliers):
        # Use numeric colormap mapping for speed
        # Remap cluster IDs to consecutive ints (0..K-1)
        uniq = np.unique(labels[inliers])
        remap = {cid: i for i, cid in enumerate(uniq)}
        cvals = np.array([remap.get(c, -1) for c in labels], dtype=np.int32)

        sc = ax.scatter(
            xyp[inliers, 0],
            xyp[inliers, 1],
            c=cvals[inliers],
            s=point_size,
            alpha=alpha,
        )
        if show_legend and len(uniq) <= 20:
            # small cluster count only
            handles, _ = sc.legend_elements(num=len(uniq))
            ax.legend(handles, [str(c) for c in uniq], title="cluster_id", loc="best", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linewidth=0.2, alpha=0.3)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_top_cluster_sizes(
    df: pd.DataFrame,
    title: str,
    out_path: Optional[str] = None,
    top_n: int = 25,
    include_outliers: bool = False,
):
    counts = df["cluster_id"].value_counts()
    if not include_outliers:
        counts = counts[counts.index >= 0]
    counts = counts.head(top_n)

    fig = plt.figure()
    ax = plt.gca()
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(i) for i in counts.index], rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.3)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_platform_prevalence_side_by_side(
    meta_df: pd.DataFrame,
    blue_df: pd.DataFrame,
    title: str,
    out_path: Optional[str] = None,
    top_n: int = 20,
):
    """
    Shows top clusters per platform side-by-side.
    Note: cluster ids are not comparable across platforms unless you matched/merged them.
    This plot is just "within-platform" prevalence.
    """
    meta_counts = meta_df["cluster_id"].value_counts()
    meta_counts = meta_counts[meta_counts.index >= 0].head(top_n)

    blue_counts = blue_df["cluster_id"].value_counts()
    blue_counts = blue_counts[blue_counts.index >= 0].head(top_n)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(range(len(meta_counts)), meta_counts.values)
    ax1.set_xticks(range(len(meta_counts)))
    ax1.set_xticklabels([str(i) for i in meta_counts.index], rotation=45, ha="right")
    ax1.set_title("Meta: top clusters")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y", linewidth=0.2, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(range(len(blue_counts)), blue_counts.values)
    ax2.set_xticks(range(len(blue_counts)))
    ax2.set_xticklabels([str(i) for i in blue_counts.index], rotation=45, ha="right")
    ax2.set_title("Bluesky: top clusters")
    ax2.grid(True, axis="y", linewidth=0.2, alpha=0.3)

    fig.suptitle(title)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_matched_cluster_ratio(
    matches_csv: str,
    title: str,
    out_path: Optional[str] = None,
    top_n: int = 30,
):
    """
    Optional: If you have a matches file like:
      meta_cluster_id,blue_cluster_id,similarity,meta_size,blue_size
    Plot log ratio and sizes.
    """
    m = pd.read_csv(matches_csv)
    required = {"meta_cluster_id", "blue_cluster_id", "similarity", "meta_size", "blue_size"}
    if not required.issubset(set(m.columns)):
        raise ValueError(f"{matches_csv} missing columns. Need {required}")

    # focus on the strongest matches
    m = m.sort_values(["similarity", "meta_size"], ascending=False).head(top_n).copy()
    m["log_ratio_meta_over_blue"] = np.log((m["meta_size"] + 1) / (m["blue_size"] + 1))

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.bar(range(len(m)), m["log_ratio_meta_over_blue"].values)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(
        [f"M{a}-B{b}" for a, b in zip(m["meta_cluster_id"], m["blue_cluster_id"])],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("log((meta_size+1)/(blue_size+1))")
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.2, alpha=0.3)

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", help="Path to <prefix>.assignments.csv", required=True)
    ap.add_argument("--clusters-json", help="Path to <prefix>.clusters.json", required=False)
    ap.add_argument("--umap2d-npy", help="Path to <prefix>.umap2d.npy", required=True)
    ap.add_argument("--out-dir", default="viz_out", help="Output directory")
    ap.add_argument("--title", default="Claim clusters", help="Plot title prefix")
    ap.add_argument("--max-points", type=int, default=200_000)
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top-n", type=int, default=25)

    # optional multi-platform / matching
    ap.add_argument("--meta-assignments", help="Meta assignments csv (optional)")
    ap.add_argument("--meta-umap2d-npy", help="Meta UMAP2D npy (optional)")
    ap.add_argument("--blue-assignments", help="Bluesky assignments csv (optional)")
    ap.add_argument("--blue-umap2d-npy", help="Bluesky UMAP2D npy (optional)")
    ap.add_argument("--matches-csv", help="Matched clusters csv (optional)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # single dataset plots
    df = load_assignments(args.assignments)
    xy = load_umap2d(args.umap2d_npy, n_rows=len(df))

    plot_umap_scatter(
        df,
        xy,
        title=f"{args.title} — UMAP scatter",
        out_path=str(out_dir / "umap_scatter.png"),
        max_points=args.max_points,
        point_size=args.point_size,
        alpha=args.alpha,
        show_legend=False,
    )

    plot_top_cluster_sizes(
        df,
        title=f"{args.title} — top cluster sizes",
        out_path=str(out_dir / "top_cluster_sizes.png"),
        top_n=args.top_n,
    )

    # optional: two platform side-by-side scatter + bars
    if args.meta_assignments and args.meta_umap2d_npy and args.blue_assignments and args.blue_umap2d_npy:
        meta_df = load_assignments(args.meta_assignments)
        meta_xy = load_umap2d(args.meta_umap2d_npy, n_rows=len(meta_df))
        blue_df = load_assignments(args.blue_assignments)
        blue_xy = load_umap2d(args.blue_umap2d_npy, n_rows=len(blue_df))

        plot_umap_scatter(
            meta_df,
            meta_xy,
            title=f"{args.title} — Meta UMAP scatter",
            out_path=str(out_dir / "meta_umap_scatter.png"),
            max_points=args.max_points,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        plot_umap_scatter(
            blue_df,
            blue_xy,
            title=f"{args.title} — Bluesky UMAP scatter",
            out_path=str(out_dir / "bluesky_umap_scatter.png"),
            max_points=args.max_points,
            point_size=args.point_size,
            alpha=args.alpha,
        )

        plot_platform_prevalence_side_by_side(
            meta_df,
            blue_df,
            title=f"{args.title} — prevalence (within-platform)",
            out_path=str(out_dir / "platform_prevalence.png"),
            top_n=min(args.top_n, 30),
        )

    # optional: matched cluster ratio plot
    if args.matches_csv:
        plot_matched_cluster_ratio(
            args.matches_csv,
            title=f"{args.title} — matched cluster prevalence ratio",
            out_path=str(out_dir / "matched_cluster_ratio.png"),
            top_n=min(args.top_n, 50),
        )

    print(f"[OK] wrote plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()