import os
import re
import json
import argparse
import pandas as pd

def load_claims(claims_csv: str) -> list[dict]:
    df = pd.read_csv(claims_csv)
    needed = ["segment", "cluster_id", "cluster_claim"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"{claims_csv} missing required column '{c}'")
    df = df.copy()
    df["segment"] = df["segment"].fillna("").astype(str).str.strip()
    df["cluster_id"] = df["cluster_id"].fillna("").astype(str).str.strip()
    df["cluster_claim"] = df["cluster_claim"].fillna("").astype(str).str.strip()
    df = df[(df["cluster_id"] != "") & (df["cluster_claim"] != "")]
    return [
        {"segment": r.segment, "cluster_id": str(r.cluster_id), "cluster_claim": r.cluster_claim}
        for r in df.itertuples(index=False)
    ]

def extract_cids_from_prompt(prompt_text: str) -> list[str]:
    cids = []
    for ln in prompt_text.splitlines():
        m = re.search(r"\[cid=([^\]]+)\]", ln)
        if m:
            cids.append(m.group(1).strip())
    return cids

def parse_numbers_only_response(resp_text: str, k: int) -> list[int | None]:
    """
    Parse k indices (1..N). If missing/invalid -> None.
    """
    lines = [ln.strip() for ln in (resp_text or "").splitlines() if ln.strip()]

    # comma-separated fallback
    if len(lines) == 1 and "," in lines[0]:
        parts = [p.strip() for p in lines[0].split(",") if p.strip()]
        if len(parts) > 1:
            lines = parts

    # pad/truncate
    if len(lines) < k:
        lines += [""] * (k - len(lines))
    elif len(lines) > k:
        lines = lines[:k]

    out: list[int | None] = []
    for ln in lines:
        # allow "1)" or "1." etc
        m = re.match(r"^\s*(\d+)", ln)
        if not m:
            out.append(None)
            continue
        out.append(int(m.group(1)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_dir", required=True, help="e.g., llm_batch_debug_claims")
    ap.add_argument("--claims_csv", required=True, help="e.g., /mnt/data/merged__ALL__kept.csv")
    ap.add_argument("--out_prefix", required=True, help="e.g., outputs/recovered_bluesky_allclaims")
    args = ap.parse_args()

    claims = load_claims(args.claims_csv)
    n_claims = len(claims)
    print(f"[LOAD] claims: {n_claims}")

    # Find all batch indices for which we have prompt and response (retry0 only)
    prompt_pat = re.compile(r"^prompt_batch(\d+)_retry0\.txt$")
    response_pat = re.compile(r"^response_batch(\d+)_retry0\.txt$")

    files = os.listdir(args.debug_dir)
    prompts = {int(m.group(1)): f for f in files if (m := prompt_pat.match(f))}
    responses = {int(m.group(1)): f for f in files if (m := response_pat.match(f))}

    common = sorted(set(prompts.keys()) & set(responses.keys()))
    print(f"[FOUND] prompt batches: {len(prompts)} | response batches: {len(responses)} | paired: {len(common)}")

    if not common:
        raise RuntimeError(
            "No paired prompt/response files found (response_batch*.txt missing?). "
            "Run: ls -1 llm_batch_debug_claims | grep '^response_' | head"
        )

    rows_out = []
    seen = set()

    for b in common:
        with open(os.path.join(args.debug_dir, prompts[b]), "r", encoding="utf-8") as f:
            prompt_txt = f.read()
        with open(os.path.join(args.debug_dir, responses[b]), "r", encoding="utf-8") as f:
            resp_txt = f.read()

        cids = extract_cids_from_prompt(prompt_txt)
        if not cids:
            print(f"[WARN] batch {b}: no cids parsed from prompt; skipping")
            continue

        idxs = parse_numbers_only_response(resp_txt, k=len(cids))

        for cid, idx in zip(cids, idxs):
            # keep first occurrence (retry handling can be added if needed)
            if cid in seen:
                continue
            seen.add(cid)

            if idx is None or idx < 1 or idx > n_claims:
                rows_out.append({
                    "cid": cid,
                    "assigned_cluster_id": None,
                    "assigned_claim": None,
                    "assigned_claim_segment": None,
                    "assign_reason": "recovered_missing_or_invalid",
                    "llm_label": None if resp_txt is None else "see_debug_response",
                    "match_status": "non_matching",
                    "debug_batch": b,
                })
            else:
                chosen = claims[idx - 1]
                rows_out.append({
                    "cid": cid,
                    "assigned_cluster_id": chosen["cluster_id"],
                    "assigned_claim": chosen["cluster_claim"],
                    "assigned_claim_segment": chosen["segment"],
                    "assign_reason": "recovered_from_debug",
                    "llm_label": str(idx),
                    "match_status": "matched",
                    "debug_batch": b,
                })

    # ensure output directory exists
    out_jsonl = f"{args.out_prefix}.jsonl"
    out_csv = f"{args.out_prefix}.csv"
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(out_csv, index=False)

    print("=== RECOVERY SUMMARY ===")
    print(f"Recovered rows: {len(df_out)}")
    print(f"Matched: {(df_out['match_status'] == 'matched').sum()}")
    print(f"Missing/invalid: {(df_out['match_status'] != 'matched').sum()}")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()