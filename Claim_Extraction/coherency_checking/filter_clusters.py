#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def read_csv_force_str(path: str) -> pd.DataFrame:
    """
    Read CSV while forcing cluster_id + segment-like columns to be strings
    so that -1 vs "-1" mismatches don't break joins/filters.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Normalize common columns 
    for col in ["cluster_id", "segment", "stance"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmented", required=True, help="segmented_coherency_results.csv")
    ap.add_argument("--claims", required=True, help="claims_clustered_all_stances.csv")

    ap.add_argument("--topk_pro_energy", help="Pro-Energy top_texts_by_cluster.csv")
    ap.add_argument("--topk_pro_climate", help="Pro-Climate top_texts_by_cluster.csv")
    ap.add_argument("--topk_neutral", help="Neutral top_texts_by_cluster.csv")
    ap.add_argument("--topk_dir", help="Directory containing the 3 top_texts_by_cluster.csv files")

    ap.add_argument("--out_dir", default=".", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load coherency results and keep only coherent
    seg = read_csv_force_str(args.segmented)

    required_cols = {"segment", "cluster_id", "verdict"}
    missing = required_cols - set(seg.columns)
    if missing:
        raise ValueError(f"{args.segmented} is missing columns: {sorted(missing)}")

    seg["verdict"] = seg["verdict"].astype(str).str.strip()
    coherent_pairs = seg.loc[seg["verdict"] == "Coherent", ["segment", "cluster_id"]].drop_duplicates()

    # Build per-segment coherent cluster_id sets 
    coherent_by_segment = (
        coherent_pairs.groupby("segment")["cluster_id"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    # 2) Filter claims_clustered_all_stances.csv by coherent cluster_ids (within its stance/segment)
    claims = read_csv_force_str(args.claims)

    required_claims_cols = {"stance", "cluster_id"}
    missing = required_claims_cols - set(claims.columns)
    if missing:
        raise ValueError(f"{args.claims} is missing columns: {sorted(missing)}")

    # Only keep rows where (stance, cluster_id) is in coherent_pairs
    claims_filtered = claims.merge(
        coherent_pairs.rename(columns={"segment": "stance"}),
        on=["stance", "cluster_id"],
        how="inner",
    )

    claims_out = os.path.join(args.out_dir, "claims_clustered_all_stances__coherent_only.csv")
    claims_filtered.to_csv(claims_out, index=False)

    # 3) Determine topk input paths
    topk_paths = {}

    if args.topk_dir:

        candidates = {
            "Pro-Energy": [
                os.path.join(args.topk_dir, "Pro-Energy", "top_texts_by_cluster.csv"),
                os.path.join(args.topk_dir, "top_texts_by_cluster_Pro-Energy.csv"),
                os.path.join(args.topk_dir, "top_texts_by_cluster__Pro-Energy.csv"),
            ],
            "Pro-Climate": [
                os.path.join(args.topk_dir, "Pro-Climate", "top_texts_by_cluster.csv"),
                os.path.join(args.topk_dir, "top_texts_by_cluster_Pro-Climate.csv"),
                os.path.join(args.topk_dir, "top_texts_by_cluster__Pro-Climate.csv"),
            ],
            "Neutral": [
                os.path.join(args.topk_dir, "Neutral", "top_texts_by_cluster.csv"),
                os.path.join(args.topk_dir, "top_texts_by_cluster_Neutral.csv"),
                os.path.join(args.topk_dir, "top_texts_by_cluster__Neutral.csv"),
            ],
        }

        for stance, paths in candidates.items():
            found = next((p for p in paths if os.path.exists(p)), None)
            if not found:
                raise FileNotFoundError(
                    f"Could not find {stance} topk file in --topk_dir. Tried:\n" + "\n".join(paths)
                )
            topk_paths[stance] = found
    else:
        # Explicit file args
        if not (args.topk_pro_energy and args.topk_pro_climate and args.topk_neutral):
            raise ValueError(
                "Provide either --topk_dir OR all of: --topk_pro_energy --topk_pro_climate --topk_neutral"
            )
        topk_paths = {
            "Pro-Energy": args.topk_pro_energy,
            "Pro-Climate": args.topk_pro_climate,
            "Neutral": args.topk_neutral,
        }

    # 4) Filter each topk file by coherent cluster_ids for that segment
    for stance, path in topk_paths.items():
        topk = read_csv_force_str(path)

        if "cluster_id" not in topk.columns:
            raise ValueError(f"{path} is missing column: cluster_id")

        allowed = coherent_by_segment.get(stance, set())
        topk_filtered = topk[topk["cluster_id"].isin(allowed)].copy()

        out_path = os.path.join(args.out_dir, f"top_texts_by_cluster__{stance}__coherent_only.csv")
        topk_filtered.to_csv(out_path, index=False)

    # 5) summary
    print("Wrote:")
    print(" -", claims_out)
    for stance in ["Pro-Energy", "Pro-Climate", "Neutral"]:
        print(" -", os.path.join(args.out_dir, f"top_texts_by_cluster__{stance}__coherent_only.csv"))
    print("\nCounts:")
    print("claims_clustered_all_stances:", len(claims), "->", len(claims_filtered))
    for stance, path in topk_paths.items():
        topk = read_csv_force_str(path)
        allowed = coherent_by_segment.get(stance, set())
        print(f"{stance} topk: {len(topk)} -> {sum(topk['cluster_id'].isin(allowed))} (allowed clusters: {len(allowed)})")


if __name__ == "__main__":
    main()
