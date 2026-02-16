# save as: run_segmented_coherency.py
import os, json, time, re, traceback, httpx
import pandas as pd
import argparse
from pathlib import Path

# Paths for the three segments
SEGMENT_PATHS = {
    "Pro-Energy": "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/clustering/clustering_out/stance=Pro-Energy/topk_texts_by_cluster.csv",
    "Pro-Climate": "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/clustering/clustering_out/stance=Pro-Climate/topk_texts_by_cluster.csv",
    "Neutral": "/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/clustering/clustering_out/stance=Neutral/topk_texts_by_cluster.csv",
}

OUT_PATH = "segmented_coherency_results.jsonl"

MODEL_ID = "mistral-large-2407"
BASE_URL = "https://api.mistral.ai/v1"
API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = f"{BASE_URL}/chat/completions"

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 512
ENV_LIMIT = 0  # 0 = no limit per segment
SLEEP_BETWEEN = 0.4

VALID_LABELS = {"coherent", "incoherent"}


def extract_verdict(text: str) -> str:
    """
    Extract coherent/incoherent label from model response.
    """
    t = (text or "").strip().lower()
    
    # Look for the exact labels we asked for
    if "coherent" in t and "incoherent" not in t:
        return "Coherent"
    if "incoherent" in t:
        return "Incoherent"
    
    # Default to incoherent if unclear
    return "Incoherent"


def truncate(s: str, max_chars=2000) -> str:
    """
    Basic truncation/cleanup for text.
    """
    s = (s or "").strip()
    if len(s) > max_chars:
        return s[:max_chars] + " …"
    return s


def format_cluster_texts(texts):
    """
    Format texts as numbered list:
    Text 1: <text>
    Text 2: <text>
    ...
    """
    lines = []
    for i, text in enumerate(texts, start=1):
        clean_text = re.sub(r"\s+", " ", text).strip()
        clean_text = truncate(clean_text)
        lines.append(f"Text {i}: {clean_text}")
    return "\n".join(lines)


def make_coherency_prompt(texts):
    """
    Create the coherency evaluation prompt.
    """
    instruction = """Q: Are the following texts coherent as a single cluster?

A cluster is considered **coherent** if all texts express a similar central claim or argument, even if phrased differently or supported by different examples.

A cluster is **incoherent** if the texts advance very different claims, vastly conflicting arguments, or only share a topic without asserting a similar position.

Answer with exactly one label:
- "Coherent"
- "Incoherent"

Input:
"""
    
    formatted_texts = format_cluster_texts(texts)
    
    return instruction + formatted_texts + "\n\nAnswer:"


def call_model(prompt, retries=4, sleep=0.8):
    """
    Call Mistral API with retry logic.
    """
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
        "stop": ["\n\n", "\nQ:", "\nText"],
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
    Load clusters from a topk_texts_by_cluster.csv file.
    Format: cluster_id,topk_rank,cid,cluster_prob,text
    
    Returns: list[(cluster_id:str, texts:list[str])]
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    
    # Sort by cluster and topk_rank to ensure correct ordering
    df = df.sort_values(["cluster_id", "topk_rank"]).reset_index(drop=True)
    
    clusters = []
    for cluster_id, group in df.groupby("cluster_id", sort=False):
        # Take top 5 texts
        texts = group["text"].fillna("").astype(str).tolist()[:5]
        
        # Pad to 5 if needed
        texts = (texts + [""] * 5)[:5]
        
        # Skip if cluster is noise (-1) or all texts are empty
        if cluster_id == -1 or all(not t.strip() for t in texts):
            continue
            
        clusters.append((str(cluster_id), texts))
    
    return clusters


def main():
    if not API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    print(f"Starting segmented coherency analysis...")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUT_PATH}\n")
    
    all_results = []
    
    # Process each segment
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
        
        # Process each cluster
        for idx, (cluster_id, texts) in enumerate(clusters, start=1):
            prompt = make_coherency_prompt(texts)
            
            try:
                model_resp = call_model(prompt)
                verdict = extract_verdict(model_resp)
                
                obj = {
                    "segment": segment_name,
                    "cluster_id": cluster_id,
                    "verdict": verdict,
                    "raw_response": model_resp.strip(),
                    "model_used": MODEL_ID,
                    "num_texts": len([t for t in texts if t.strip()]),
                }
            except Exception as e:
                traceback.print_exc()
                obj = {
                    "segment": segment_name,
                    "cluster_id": cluster_id,
                    "verdict": None,
                    "error": f"request_error: {e}",
                    "model_used": MODEL_ID,
                }
            
            all_results.append(obj)
            
            time.sleep(SLEEP_BETWEEN)
            
            if idx % 5 == 0:
                print(f"  Processed {idx}/{len(clusters)} clusters in {segment_name}...")
        
        print(f"Completed {segment_name}: {len(clusters)} clusters processed")
    
    # Write results to JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"Done! Wrote {len(all_results)} cluster results to {OUT_PATH}")
    
    # Also save as CSV for easier analysis
    try:
        csv_path = OUT_PATH.replace(".jsonl", ".csv")
        out_rows = []
        for r in all_results:
            out_rows.append({
                "segment": r.get("segment"),
                "cluster_id": r.get("cluster_id"),
                "verdict": r.get("verdict"),
                "num_texts": r.get("num_texts"),
                "error": r.get("error"),
            })
        pd.DataFrame(out_rows).to_csv(csv_path, index=False)
        print(f"Also saved to {csv_path}")
        
        # Print summary statistics
        df_summary = pd.DataFrame(out_rows)
        print(f"\n{'='*60}")
        print("Summary by Segment:")
        print(f"{'='*60}")
        for segment in df_summary["segment"].unique():
            seg_df = df_summary[df_summary["segment"] == segment]
            coherent = (seg_df["verdict"] == "Coherent").sum()
            incoherent = (seg_df["verdict"] == "Incoherent").sum()
            errors = seg_df["error"].notna().sum()
            total = len(seg_df)
            
            print(f"\n{segment}:")
            print(f"  Total clusters: {total}")
            print(f"  Coherent: {coherent} ({100*coherent/total:.1f}%)")
            print(f"  Incoherent: {incoherent} ({100*incoherent/total:.1f}%)")
            if errors > 0:
                print(f"  Errors: {errors}")
        
    except Exception as e:
        print(f"Failed to save CSV summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate coherency of segmented clusters"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of clusters per segment (0 = no limit)"
    )
    args = parser.parse_args()
    
    ENV_LIMIT = args.limit
    
    main()
