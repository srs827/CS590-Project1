# save as: run_segmented_cluster_claims.py
import os, json, time, re, traceback, httpx
import pandas as pd
import argparse

# Paths for the three segments
SEGMENT_PATHS = {
    "Pro-Energy": "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/coherency_checking/coherent_outputs/top_texts_by_cluster__Pro-Energy__coherent_only.csv",
    "Pro-Climate": "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/coherency_checking/coherent_outputs/top_texts_by_cluster__Pro-Climate__coherent_only.csv",
    "Neutral": "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/coherency_checking/coherent_outputs/top_texts_by_cluster__Neutral__coherent_only.csv",
}

OUT_PATH = "segmented_cluster_claims.jsonl"

MODEL_ID = "mistral-large-2407"
BASE_URL = "https://api.mistral.ai/v1"
API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = f"{BASE_URL}/chat/completions"

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 96
ENV_LIMIT = 0  # 0 = no limit per segment
SLEEP_BETWEEN = 0.4


def truncate(s: str, max_chars=2500) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_chars:
        return s[:max_chars] + " …"
    return s


def format_cluster_texts(texts):
    lines = []
    for i, text in enumerate(texts, start=1):
        lines.append(f"Text {i}: {truncate(text)}")
    return "\n".join(lines)


def make_claim_prompt(texts):
    instruction = """Given the following texts, identify the single central claim they collectively support.

The claim must:
- Be a single, concise declarative sentence
- Express a clear position, evaluation, recommendation, or causal assertion
- Be something a reader could agree or disagree with

Do NOT:
- Summarize the texts
- List multiple claims
- Add evidence, examples, or qualifiers
- Refer to the texts themselves

Output only the claim sentence.

Input:
"""
    return instruction + format_cluster_texts(texts) + "\n\nOutput:"


def postprocess_claim(resp: str) -> str:
    """
    Try to coerce response to a single sentence claim:
    - strip quotes/bullets/prefixes
    - take first non-empty line
    - if multiple sentences, keep first sentence 
    """
    t = (resp or "").strip()

    # take first non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        t = lines[0]

    # remove common prefixes/bullets
    t = re.sub(r"^(claim\s*:\s*)", "", t, flags=re.I)
    t = re.sub(r"^[-•\*\d\)\.]+\s*", "", t).strip()

    # strip wrapping quotes
    t = t.strip().strip('"').strip("'").strip()

    # if model returned multiple sentences in one line, keep the first sentence
    m = re.match(r"^(.+?[.!?])\s+.+$", t)
    if m:
        t = m.group(1).strip()

    return t


def call_model(prompt, retries=4, sleep=0.8):
    last_err = None
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "stop": ["\n\n", "\nText 2:", "\nText 3:", "\nText 4:", "\nText 5:", "\nInput:"],
    }

    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code != 200:
                print(f"[API Error] HTTP {r.status_code}: {r.text[:800]}")
                r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"Attempt {attempt}/{retries} failed: {last_err}")
            time.sleep(sleep * attempt)

    raise RuntimeError(last_err or "unknown error")


def load_segment_clusters(csv_path):
    """
    Load clusters from top_texts_by_cluster__<segment>.csv
    Format: cluster_id,topk_rank,cid,cluster_prob,text

    Returns: list[(cluster_id:str, texts:list[str])]
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return []

    # force string to avoid -1 numeric issues
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # normalize
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    if "topk_rank" in df.columns:
        # numeric sort if possible
        df["topk_rank_num"] = pd.to_numeric(df["topk_rank"], errors="coerce")
        df = df.sort_values(["cluster_id", "topk_rank_num"], na_position="last").reset_index(drop=True)
    else:
        df = df.sort_values(["cluster_id"]).reset_index(drop=True)

    clusters = []
    for cluster_id, group in df.groupby("cluster_id", sort=False):
        texts = group.get("text", pd.Series([], dtype=str)).fillna("").astype(str).tolist()[:5]
        texts = (texts + [""] * 5)[:5]

        # Skip if noise or empty
        if cluster_id.strip() in {"-1", "None", ""}:
            continue
        if all(not t.strip() for t in texts):
            continue

        clusters.append((cluster_id, texts))

    return clusters


def main():
    global ENV_LIMIT

    if not API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable not set")

    print("Starting segmented cluster-claim extraction...")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUT_PATH}\n")

    all_results = []

    for segment_name, csv_path in SEGMENT_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_name}")
        print(f"{'='*60}")

        clusters = load_segment_clusters(csv_path)
        if not clusters:
            print(f"No clusters found for {segment_name}, skipping...")
            continue

        print(f"Found {len(clusters)} clusters in {segment_name}")

        if ENV_LIMIT > 0:
            clusters = clusters[:ENV_LIMIT]
            print(f"Limiting to {ENV_LIMIT} clusters for testing")

        for idx, (cluster_id, texts) in enumerate(clusters, start=1):
            prompt = make_claim_prompt(texts)

            try:
                model_resp = call_model(prompt)
                claim = postprocess_claim(model_resp)

                obj = {
                    "segment": segment_name,
                    "cluster_id": cluster_id,
                    "cluster_claim": claim,
                    "raw_response": model_resp,
                    "model_used": MODEL_ID,
                    "num_texts": len([t for t in texts if t.strip()]),
                }
            except Exception as e:
                traceback.print_exc()
                obj = {
                    "segment": segment_name,
                    "cluster_id": cluster_id,
                    "cluster_claim": None,
                    "error": f"request_error: {e}",
                    "model_used": MODEL_ID,
                }

            all_results.append(obj)
            time.sleep(SLEEP_BETWEEN)

            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(clusters)} clusters in {segment_name}...")

        print(f"Completed {segment_name}: {len(clusters)} clusters processed")

    # Write JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! Wrote {len(all_results)} cluster claims to {OUT_PATH}")

    # Also write CSV
    try:
        csv_path = OUT_PATH.replace(".jsonl", ".csv")
        out_rows = []
        for r in all_results:
            out_rows.append({
                "segment": r.get("segment"),
                "cluster_id": r.get("cluster_id"),
                "cluster_claim": r.get("cluster_claim"),
                "num_texts": r.get("num_texts"),
                "error": r.get("error"),
                "model_used": r.get("model_used"),
            })
        pd.DataFrame(out_rows).to_csv(csv_path, index=False)
        print(f"Also saved to {csv_path}")

        # quick summary
        df = pd.DataFrame(out_rows)
        ok = df["cluster_claim"].notna().sum()
        err = df["error"].notna().sum()
        print(f"Successful: {ok} | Errors: {err}")

    except Exception as e:
        print(f"Failed to save CSV summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one central claim per cluster")
    parser.add_argument("--limit", type=int, default=0, help="Limit clusters per segment (0 = no limit)")
    parser.add_argument(
        "--pro_energy", type=str, default=None,
        help="Override path for Pro-Energy CSV"
    )
    parser.add_argument(
        "--pro_climate", type=str, default=None,
        help="Override path for Pro-Climate CSV"
    )
    parser.add_argument(
        "--neutral", type=str, default=None,
        help="Override path for Neutral CSV"
    )
    args = parser.parse_args()

    ENV_LIMIT = args.limit

    # Optional CLI overrides
    if args.pro_energy:
        SEGMENT_PATHS["Pro-Energy"] = args.pro_energy
    if args.pro_climate:
        SEGMENT_PATHS["Pro-Climate"] = args.pro_climate
    if args.neutral:
        SEGMENT_PATHS["Neutral"] = args.neutral

    main()
