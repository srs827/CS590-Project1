"""
Topic Extraction Pipeline for Cross-Platform Claim Comparison
=============================================================
Embeds claims from both platforms jointly, clusters them into topics,
labels clusters via LLM (Mistral via httpx), and produces a representation
matrix for comparison.

Usage:
    python topic_extraction.py \
        --platform_a meta_claims.csv \
        --platform_b bluesky_claims.csv \
        --label_a Meta \
        --label_b Bluesky \
        --output_dir ./topic_output

Install:
    pip install sentence-transformers umap-learn hdbscan httpx matplotlib seaborn
"""

import argparse
import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import httpx

# ── Mistral API config ────────────────────────────────────────────────────────
API_KEY     = os.getenv("MISTRAL_API_KEY")
BASE_URL    = "https://api.mistral.ai/v1"
API_URL     = f"{BASE_URL}/chat/completions"
MODEL_ID    = "mistral-large-2407"
TEMPERATURE = 0.2
TOP_P       = 0.9
MAX_TOKENS  = 200


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_claims(claims: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a list of claim strings using a sentence-transformer model.
    Swap model_name for 'climatebert/distilroberta-base-climate-sentiment'
    or any HuggingFace model if you want a domain-specific encoder.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(claims, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


# ── Dimensionality Reduction ──────────────────────────────────────────────────
def reduce_dimensions(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    UMAP reduction before clustering. n_components=10 preserves more structure
    than 2D; use n_components=2 only for visualization.
    """
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=5,       # small: dataset is small (50-100 claims)
        min_dist=0.0,        # tighter clusters
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


# ── Clustering ────────────────────────────────────────────────────────────────
def cluster_claims(
    reduced: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> np.ndarray:
    """
    HDBSCAN clustering. Returns an array of integer cluster labels;
    -1 indicates noise (unclustered claims).

    Tune min_cluster_size down to 2 if your claim set is small and you
    want finer-grained topics, or up to 5 for broader categories.
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(reduced)


# ── LLM Caller ───────────────────────────────────────────────────────────────
def call_llm(
    messages: list[dict],
    max_retries: int = 5,
    base_sleep: float = 1.0,
    max_sleep: float = 10.0,
) -> str:
    """
    Send a chat completion request to Mistral via httpx with exponential
    backoff retry logic. Matches the call_llm pattern used in the main pipeline.
    """
    if API_KEY in (None, "", "REPLACE_ME"):
        raise RuntimeError("MISTRAL_API_KEY not set; export it before running.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code == 429:
                raise RuntimeError("RATE_LIMITED_429")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            last_err = f"{e.response.status_code}: {e.response.text[:500]}"
            # Non-retriable client errors
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                break

        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = str(e)

        except RuntimeError:
            raise  # propagate rate-limit signal upward

        if attempt < max_retries:
            sleep = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            jitter = sleep * (0.8 + 0.4 * random.random())
            print(f"  [LLM RETRY] attempt={attempt} sleep={jitter:.1f}s error={last_err}")
            time.sleep(jitter)

    raise RuntimeError(last_err or "unknown error")


# ── LLM Topic Labeling ────────────────────────────────────────────────────────
def label_clusters_with_llm(
    df: pd.DataFrame,
    claim_col: str = "cluster_claim",
    cluster_col: str = "topic_cluster",
) -> dict[int, str]:
    """
    For each cluster, send all member claims to Mistral via httpx and ask for a
    short topic label. Returns a dict mapping cluster_id -> topic_label.
    """
    cluster_ids = sorted([c for c in df[cluster_col].unique() if c != -1])
    topic_labels = {}

    for cluster_id in cluster_ids:
        claims_in_cluster = df[df[cluster_col] == cluster_id][claim_col].tolist()
        claims_formatted = "\n".join(f"- {c}" for c in claims_in_cluster)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a climate discourse researcher. Given a set of thematically "
                    "related claims extracted from social media, produce a short topic label "
                    "(3-6 words) that captures their shared theme. Respond with ONLY the label, "
                    "no explanation or punctuation at the end."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here are the claims in this cluster:\n{claims_formatted}\n\n"
                    "What is the shared topic? Give a 3-6 word label."
                ),
            },
        ]

        label = call_llm(messages).strip()
        topic_labels[int(cluster_id)] = label
        print(f"  Cluster {cluster_id}: {label}  ({len(claims_in_cluster)} claims)")

    # Noise points get their own label
    topic_labels[-1] = "Uncategorized"
    return topic_labels


# ── Representation Matrix ─────────────────────────────────────────────────────
def build_representation_matrix(
    df: pd.DataFrame,
    platform_col: str = "platform",
    topic_col: str = "topic_label",
    segment_col: str = "segment",
) -> pd.DataFrame:
    """
    Builds a topic x platform matrix showing count and percentage of claims
    per topic on each platform. Excludes uncategorized noise points.
    """
    df_filtered = df[df[topic_col] != "Uncategorized"].copy()

    counts = (
        df_filtered.groupby([topic_col, platform_col])
        .size()
        .unstack(fill_value=0)
    )

    # Percentage within each platform
    pct = counts.div(counts.sum(axis=0), axis=1) * 100

    # Combine into a readable multi-level frame
    platforms = counts.columns.tolist()
    combined = pd.DataFrame(index=counts.index)
    for p in platforms:
        combined[f"{p}_n"] = counts[p]
        combined[f"{p}_pct"] = pct[p].round(1)

    # Add absolute difference column if exactly 2 platforms
    if len(platforms) == 2:
        combined["pct_diff"] = (
            pct[platforms[0]] - pct[platforms[1]]
        ).round(1)

    combined = combined.sort_values(
        f"{platforms[0]}_pct", ascending=False
    )
    return combined


def build_segment_breakdown(
    df: pd.DataFrame,
    platform_col: str = "platform",
    topic_col: str = "topic_label",
    segment_col: str = "segment",
) -> pd.DataFrame:
    """
    Breaks down topic representation by stance segment (Pro-Climate,
    Pro-Energy, Neutral) per platform. Useful for seeing whether topic
    differences are stance-conditional.
    """
    df_filtered = df[df[topic_col] != "Uncategorized"].copy()
    counts = (
        df_filtered.groupby([topic_col, platform_col, segment_col])
        .size()
        .reset_index(name="n")
    )
    return counts


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_representation_matrix(
    matrix: pd.DataFrame,
    platform_labels: list[str],
    output_path: str,
):
    """
    Grouped bar chart of topic representation (%) per platform.
    """
    topics = matrix.index.tolist()
    x = np.arange(len(topics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        x - width / 2,
        matrix[f"{platform_labels[0]}_pct"],
        width,
        label=platform_labels[0],
        color="#4C9BE8",
    )
    ax.bar(
        x + width / 2,
        matrix[f"{platform_labels[1]}_pct"],
        width,
        label=platform_labels[1],
        color="#E8834C",
    )

    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("% of Platform Claims", fontsize=12)
    ax.set_title("Topic Representation by Platform", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=35, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved bar chart -> {output_path}")


def plot_heatmap(
    matrix: pd.DataFrame,
    platform_labels: list[str],
    output_path: str,
):
    """
    Heatmap of topic x platform percentages, with a diverging colormap
    highlighting overrepresentation on either platform.
    """
    pct_data = matrix[[f"{p}_pct" for p in platform_labels]].copy()
    pct_data.columns = platform_labels

    fig, ax = plt.subplots(figsize=(6, max(6, len(pct_data) * 0.45)))
    sns.heatmap(
        pct_data,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=pct_data.values.mean(),
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "% of Platform Claims"},
    )
    ax.set_title("Topic x Platform Representation (%)", fontsize=13)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {output_path}")


def plot_2d_embedding(
    reduced_2d: np.ndarray,
    df: pd.DataFrame,
    output_path: str,
    topic_col: str = "topic_label",
    platform_col: str = "platform",
):
    """
    2D scatter of claim embeddings colored by topic, shaped by platform.
    Useful for a qualitative sanity check of cluster quality.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 7))
    topics = df[topic_col].unique()
    platforms = df[platform_col].unique()
    markers = ["o", "^", "s", "D"]
    palette = sns.color_palette("tab10", n_colors=len(topics))
    color_map = dict(zip(topics, palette))
    marker_map = dict(zip(platforms, markers))

    for platform in platforms:
        mask = df[platform_col] == platform
        ax.scatter(
            reduced_2d[mask, 0],
            reduced_2d[mask, 1],
            c=[color_map[t] for t in df[mask][topic_col]],
            marker=marker_map[platform],
            label=platform,
            alpha=0.75,
            s=80,
            edgecolors="white",
            linewidths=0.4,
        )

    topic_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[t], markersize=9, label=t)
        for t in topics if t != "Uncategorized"
    ]
    platform_handles = [
        Line2D([0], [0], marker=marker_map[p], color="grey",
               markersize=9, label=p)
        for p in platforms
    ]
    ax.legend(
        handles=topic_handles + platform_handles,
        bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8,
    )
    ax.set_title("Claim Embeddings by Topic and Platform (2D UMAP)", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D scatter -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform_a", required=True, help="Path to first platform CSV")
    parser.add_argument("--platform_b", required=True, help="Path to second platform CSV")
    parser.add_argument("--label_a", default="Platform_A", help="Name for platform A")
    parser.add_argument("--label_b", default="Platform_B", help="Name for platform B")
    parser.add_argument("--claim_col", default="cluster_claim", help="Column with claim text")
    parser.add_argument("--segment_col", default="segment", help="Column with stance segment")
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--output_dir", default="./topic_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load both CSVs and tag with platform label
    print("Loading data...")
    df_a = pd.read_csv(args.platform_a)
    df_b = pd.read_csv(args.platform_b)
    df_a["platform"] = args.label_a
    df_b["platform"] = args.label_b
    df = pd.concat([df_a, df_b], ignore_index=True)

    claims = df[args.claim_col].tolist()
    print(f"  Total claims: {len(claims)} ({len(df_a)} {args.label_a} + {len(df_b)} {args.label_b})")

    # 2. Embed
    print("\nEmbedding claims...")
    embeddings = embed_claims(claims)

    # 3. Reduce for clustering (higher dims) and visualization (2D)
    print("\nReducing dimensions for clustering (10D)...")
    reduced_10d = reduce_dimensions(embeddings, n_components=10)

    print("Reducing dimensions for visualization (2D)...")
    reduced_2d = reduce_dimensions(embeddings, n_components=2)

    # 4. Cluster
    print("\nClustering...")
    labels = cluster_claims(reduced_10d, min_cluster_size=args.min_cluster_size)
    df["topic_cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} topic clusters, {n_noise} noise points")

    # 5. LLM labeling
    print("\nLabeling clusters with LLM...")
    topic_labels = label_clusters_with_llm(df, claim_col=args.claim_col)
    df["topic_label"] = df["topic_cluster"].astype(int).map(topic_labels)

    # Save labeled claims
    labeled_path = os.path.join(args.output_dir, "claims_with_topics.csv")
    df.to_csv(labeled_path, index=False)
    print(f"\nSaved labeled claims -> {labeled_path}")

    # 6. Representation matrix
    print("\nBuilding representation matrix...")
    matrix = build_representation_matrix(df, platform_col="platform", topic_col="topic_label")
    matrix_path = os.path.join(args.output_dir, "representation_matrix.csv")
    matrix.to_csv(matrix_path)
    print(matrix.to_string())
    print(f"Saved matrix -> {matrix_path}")

    # Segment breakdown
    breakdown = build_segment_breakdown(df)
    breakdown_path = os.path.join(args.output_dir, "segment_breakdown.csv")
    breakdown.to_csv(breakdown_path, index=False)

    # 7. Save topic label mapping
    label_map_path = os.path.join(args.output_dir, "topic_label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({int(k): v for k, v in topic_labels.items()}, f, indent=2)

    # 8. Visualizations
    print("\nGenerating visualizations...")
    plot_representation_matrix(
        matrix,
        platform_labels=[args.label_a, args.label_b],
        output_path=os.path.join(args.output_dir, "topic_representation_bar.png"),
    )
    plot_heatmap(
        matrix,
        platform_labels=[args.label_a, args.label_b],
        output_path=os.path.join(args.output_dir, "topic_representation_heatmap.png"),
    )
    plot_2d_embedding(
        reduced_2d,
        df,
        output_path=os.path.join(args.output_dir, "claim_embedding_scatter.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
