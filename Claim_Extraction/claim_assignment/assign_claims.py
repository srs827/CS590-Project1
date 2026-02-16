# save as: assign_with_llm_claim_batches.py

import os
import json
import time
import random
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import httpx
from textwrap import shorten


# -----------------------------
# Mistral API config
# -----------------------------
API_KEY   = os.getenv("MISTRAL_API_KEY")
BASE_URL  = "https://api.mistral.ai/v1"
API_URL   = f"{BASE_URL}/chat/completions"
MODEL_ID  = "mistral-large-2407"

TEMPERATURE = 0.2
TOP_P       = 0.9
MAX_TOKENS  = 512

# batching
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 0.5

# debug
DEBUG_VERBOSE      = True
DEBUG_DUMP_TO_DISK = True
DEBUG_DIR          = "llm_batch_debug_claims"


# -----------------------------
# Utilities
# -----------------------------
def _ensure_debug_dir():
    if DEBUG_DUMP_TO_DISK:
        os.makedirs(DEBUG_DIR, exist_ok=True)

def _dump_text(tag: str, batch_idx: int, retry_num: int, content: str):
    if not DEBUG_DUMP_TO_DISK:
        return
    _ensure_debug_dir()
    suffix = f"batch{batch_idx:05d}_retry{retry_num}"
    path = os.path.join(DEBUG_DIR, f"{tag}_{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")

def strip_md_quotes(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'^\*{1,3}\s*', '', s)
    s = re.sub(r'\s*\*{1,3}$', '', s)
    s = s.strip("`'\"“”‘’")
    return s.strip()

def normalize_label(s: str) -> str:
    # normalize stance/segment labels (keeps case-insensitive matching)
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def safe_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


# -----------------------------
# Load claims
# -----------------------------
def load_claims(claims_csv: str) -> pd.DataFrame:
    """
    Expected columns:
      - segment
      - cluster_id
      - cluster_claim
    """
    df = pd.read_csv(claims_csv)
    needed = ["segment", "cluster_id", "cluster_claim"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"{claims_csv} missing required column '{c}'")

    df["segment"] = df["segment"].fillna("").astype(str).str.strip()
    df["cluster_id"] = df["cluster_id"].fillna("").astype(str).str.strip()
    df["cluster_claim"] = df["cluster_claim"].fillna("").astype(str).str.strip()

    df = df[df["cluster_id"] != ""].copy()
    df = df[df["cluster_claim"] != ""].copy()

    return df


# -----------------------------
# Load datasets (meta/bluesky/common)
# -----------------------------
def load_text_dataset(csv_path: str, dataset_kind: str) -> pd.DataFrame:
    """
    dataset_kind:
      - bluesky_raw: expects cid, text
      - meta_raw: expects ad_archive_id, ad_creative_bodies (renamed to cid, text)
      - common: expects cid, text
    """
    df = pd.read_csv(csv_path)

    if dataset_kind == "meta_raw":
        required = ["ad_archive_id", "ad_creative_bodies"]
        for c in required:
            if c not in df.columns:
                raise RuntimeError(f"{csv_path} missing '{c}' required for meta_raw")
        df = df.rename(columns={"ad_archive_id": "cid", "ad_creative_bodies": "text"})
    elif dataset_kind in ("bluesky_raw", "common"):
        required = ["cid", "text"]
        for c in required:
            if c not in df.columns:
                raise RuntimeError(f"{csv_path} missing '{c}' required for {dataset_kind}")
    else:
        raise RuntimeError(f"Unknown dataset_kind: {dataset_kind}")

    df["cid"] = df["cid"].astype(str)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    return df


def ensure_segment_column(df: pd.DataFrame, segment_col: str = "predicted_label") -> None:
    if segment_col not in df.columns:
        raise RuntimeError(
            f"Segment-labeled mode requires column '{segment_col}', but it was not found."
        )


# -----------------------------
# Prompt building
# -----------------------------
def build_batch_prompt(
    items_batch: List[Dict[str, str]],
    claims_list: List[Dict[str, str]],
    instruction_prefix: str = "Choose, for each text, the single best-fitting claim from the numbered list below."
) -> str:
    """
    items_batch: [{"cid": ..., "text": ...}, ...]
    claims_list: [{"cluster_id": ..., "cluster_claim": ..., "segment": ...}, ...]
    Output: EXACTLY k lines, line i is ONLY a number.
    """

    claim_lines = []
    for idx, row in enumerate(claims_list, start=1):
        # include segment for context; keep it compact
        seg = row.get("segment", "")
        claim = safe_one_line(row.get("cluster_claim", ""))
        claim_lines.append(f"{idx}. [{seg}] {claim}")

    item_lines = []
    for idx, it in enumerate(items_batch, start=1):
        text_one_line = safe_one_line(it["text"])
        item_lines.append(f'{idx}. [cid={it["cid"]}] {text_one_line}')
    k = len(items_batch)
    instruction_prefix = (
        "Task: For each text, select the SINGLE best-fitting claim from the numbered list.\n\n"
        "How to decide (follow in order):\n"
        "1) Match the *core proposition* (what is being asserted), not just shared keywords.\n"
        "2) Prefer claims that preserve polarity/stance (supports vs criticizes) and direction of causality.\n"
        "3) If multiple claims are similar, choose the most specific one that still fully covers the text.\n"
        "4) Ignore hashtags, URLs, emojis, and boilerplate unless they change meaning.\n"
        "5) If the text contains multiple ideas, choose the claim that best captures the main point.\n\n"
        "Output rules (MUST follow exactly):\n"
        f"- Output EXACTLY {k} lines.\n"
        "- Line i: output ONLY the NUMBER of the chosen claim for text i.\n"
        "- No other text. No punctuation. No explanations."
    )
    
    prompt = (
        instruction_prefix + "\n\n"
        "Allowed claims:\n" + "\n".join(claim_lines) + "\n\n"
        "Texts:\n" + "\n".join(item_lines) + "\n\n"
        "Response (numbers only, one per line):\n"
    )
    return prompt


# -----------------------------
# Parse LLM response
# -----------------------------
def parse_llm_batch_response(
    raw: str,
    items_batch: List[Dict[str, str]],
    claims_list: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Accept numeric indices 1..N mapping to claims_list.
    If a line is missing/invalid -> mark needs_retry.
    """
    idx_to_claim = {str(i): claims_list[i - 1] for i in range(1, len(claims_list) + 1)}

    lines = [ln for ln in (raw or "").splitlines() if ln.strip()]

    # comma-separated fallback
    if len(lines) == 1 and ("," in lines[0]):
        parts = [p.strip() for p in lines[0].split(",") if p.strip()]
        if len(parts) > 1:
            lines = parts

    # pad/truncate
    if len(lines) < len(items_batch):
        lines += [None] * (len(items_batch) - len(lines))
    elif len(lines) > len(items_batch):
        lines = lines[:len(items_batch)]

    out: Dict[str, Dict[str, Any]] = {}

    for it, raw_line in zip(items_batch, lines):
        cleaned = (raw_line or "").strip()
        cleaned = re.sub(r'^\s*(\[\d+\]|[-*]|\d+[\)\].:])\s*', '', cleaned).strip()

        chosen: Optional[Dict[str, str]] = None
        m = re.match(r'^\s*(\d+)\s*$', cleaned) or re.match(r'^\s*(\d+)', cleaned)
        if m:
            idx = m.group(1)
            if idx in idx_to_claim:
                chosen = idx_to_claim[idx]

        if chosen is not None:
            out[it["cid"]] = {
                "chosen_cluster_id": chosen["cluster_id"],
                "chosen_claim": chosen["cluster_claim"],
                "chosen_segment": chosen.get("segment"),
                "llm_label": raw_line,
                "needs_retry": False,
                "match_status": "matched",
                "cleaned_attempt": cleaned,
            }
        else:
            out[it["cid"]] = {
                "chosen_cluster_id": None,
                "chosen_claim": None,
                "chosen_segment": None,
                "llm_label": raw_line,
                "needs_retry": True,
                "match_status": "non_matching" if raw_line else "no_output",
                "cleaned_attempt": cleaned or None,
            }

    return out


def print_batch_diagnostics(
    batch_idx: int,
    retry_num: int,
    raw: str,
    items_batch: List[Dict[str, str]],
    batch_map: Dict[str, Dict[str, Any]],
):
    if not DEBUG_VERBOSE:
        return

    failed = [(it["cid"], batch_map.get(it["cid"], {})) for it in items_batch
              if not batch_map.get(it["cid"], {}).get("chosen_cluster_id")]
    matched = [(it["cid"], batch_map.get(it["cid"], {})) for it in items_batch
               if batch_map.get(it["cid"], {}).get("chosen_cluster_id")]

    for (cid, rec) in matched[:3]:
        print(f"[OK] {cid} -> {rec.get('chosen_cluster_id')}")

    for (cid, rec) in failed[:5]:
        cleaned = rec.get("cleaned_attempt")
        raw_label = rec.get("llm_label")
        text = next((a["text"] for a in items_batch if a["cid"] == cid), "")
        snippet = shorten(text.replace("\n", " "), width=120, placeholder="…")
        print(f"[FAIL] cid={cid} cleaned='{cleaned}' raw='{raw_label}' text='{snippet}'")


# -----------------------------
# LLM caller
# -----------------------------
def call_llm(messages_or_text, max_retries=5, base_sleep=1.0, max_sleep=10.0):
    if API_KEY in (None, "", "REPLACE_ME"):
        raise RuntimeError("MISTRAL_API_KEY not set; export it before running.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if isinstance(messages_or_text, str):
        payload_messages = [{"role": "user", "content": messages_or_text}]
    else:
        payload_messages = messages_or_text

    base_payload = {
        "model": MODEL_ID,
        "messages": payload_messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.post(API_URL, json=base_payload, headers=headers, timeout=120)
            if r.status_code == 429:
                body = r.text if hasattr(r, "text") else "<no-body>"
                print(f"[HTTP 429] {len(body)} bytes: {body[:500]}")
                raise RuntimeError("RATE_LIMITED_429")

            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            resp = e.response
            body = resp.text if hasattr(resp, "text") else ""
            last_err = f"{type(e).__name__} {resp.status_code}: {body[:1000]}"
            print(f"[HTTP ERROR] status={resp.status_code} body_len={len(body)}\n{body[:1000]}")
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
            if attempt < max_retries:
                sleep_time = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                jitter = sleep_time * (0.8 + 0.4 * random.random())
                print(f"[HTTP RETRY] attempt={attempt} sleep={jitter:.2f}s")
                time.sleep(jitter)

        except (httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[HTTP EXC] {last_err}")
            if attempt < max_retries:
                sleep_time = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                jitter = sleep_time * (0.8 + 0.4 * random.random())
                print(f"[HTTP RETRY] attempt={attempt} sleep={jitter:.2f}s")
                time.sleep(jitter)

        except Exception as e:
            last_err = f"Unexpected: {e}"
            print(f"[HTTP UNEXPECTED] {last_err}")
            if "RATE_LIMITED_429" in str(e):
                raise
            if attempt < max_retries:
                sleep_time = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                jitter = sleep_time * (0.8 + 0.4 * random.random())
                print(f"[HTTP RETRY] attempt={attempt} sleep={jitter:.2f}s")
                time.sleep(jitter)

    raise RuntimeError(last_err or "unknown error")


# -----------------------------
# Core assignment runner
# -----------------------------
def run_assignment(
    df_items: pd.DataFrame,
    claims_df: pd.DataFrame,
    candidate_claims: List[Dict[str, str]],
    out_prefix: str,
    per_batch_max_retries: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assign each row in df_items (must have cid,text) to one claim from candidate_claims.
    Writes:
      - {out_prefix}.jsonl
      - {out_prefix}.csv
    Returns:
      - output dataframe
      - summary dict
    """
    _ensure_debug_dir()

    rows = list(df_items.itertuples(index=False))
    total = len(rows)

    llm_assignments: Dict[str, Dict[str, Any]] = {}
    hit_rate_limit = False

    for start in range(0, total, BATCH_SIZE):
        if hit_rate_limit:
            break

        batch_rows = rows[start:start + BATCH_SIZE]
        items_batch = [{"cid": str(r.cid), "text": str(r.text)} for r in batch_rows]
        batch_idx = start // BATCH_SIZE

        print(f"[BATCH] sending {len(items_batch)} items ({start}..{start+len(items_batch)-1})")
        prompt = build_batch_prompt(items_batch, candidate_claims)
        _dump_text("prompt", batch_idx, 0, prompt)

        try:
            raw = call_llm(prompt)
            _dump_text("response", batch_idx, 0, raw)
            batch_map = parse_llm_batch_response(raw, items_batch, candidate_claims)
            llm_assignments.update(batch_map)
            print(f"[BATCH] got {len(batch_map)} assignments")
            print_batch_diagnostics(batch_idx, 0, raw, items_batch, batch_map)

        except RuntimeError as e:
            if str(e) == "RATE_LIMITED_429":
                print("[BATCH] hit 429 rate limit. Stopping early and saving progress.")
                hit_rate_limit = True
            else:
                print(f"[BATCH] fatal error: {e}")
            break
        except Exception as e:
            print(f"[BATCH] error: {e}")
            time.sleep(SLEEP_BETWEEN_BATCHES)
            continue

        # Retry failures inside this batch
        id_to_text = {a["cid"]: a["text"] for a in items_batch}
        failed_ids = [cid for cid, rec in batch_map.items() if rec.get("chosen_cluster_id") is None]

        retry_num = 0
        while failed_ids and retry_num < per_batch_max_retries and not hit_rate_limit:
            retry_num += 1
            print(f"[BATCH] retry #{retry_num}: reattempting {len(failed_ids)} failed items")

            retry_batch = [{"cid": cid, "text": id_to_text[cid]} for cid in failed_ids]
            retry_prompt = build_batch_prompt(retry_batch, candidate_claims)
            _dump_text("prompt", batch_idx, retry_num, retry_prompt)

            try:
                raw_retry = call_llm(retry_prompt)
                _dump_text("response", batch_idx, retry_num, raw_retry)
                retry_map = parse_llm_batch_response(raw_retry, retry_batch, candidate_claims)
                llm_assignments.update(retry_map)
                print_batch_diagnostics(batch_idx, retry_num, raw_retry, retry_batch, retry_map)
                failed_ids = [cid for cid, rec in retry_map.items() if rec.get("chosen_cluster_id") is None]

            except RuntimeError as e:
                if str(e) == "RATE_LIMITED_429":
                    print("[BATCH] retry hit 429; stopping.")
                    hit_rate_limit = True
                    break
                else:
                    print(f"[BATCH] retry fatal error: {e}")
                    break
            except Exception as e:
                print(f"[BATCH] retry error: {e}")

            time.sleep(min(10.0, SLEEP_BETWEEN_BATCHES * (2 ** (retry_num - 1))))

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Build final output rows
    final_rows = []
    attempted_ids = set(llm_assignments.keys())

    for r in df_items.itertuples(index=False):
        cid = str(r.cid)
        text = str(r.text)
        rec = llm_assignments.get(cid, {})

        if cid in attempted_ids and rec.get("chosen_cluster_id") is not None:
            final_rows.append({
                "cid": cid,
                "text": text,
                "assigned_cluster_id": rec.get("chosen_cluster_id"),
                "assigned_claim": rec.get("chosen_claim"),
                "assigned_claim_segment": rec.get("chosen_segment"),
                "assign_reason": "llm_claim_assign",
                "llm_label": rec.get("llm_label"),
                "match_status": rec.get("match_status"),
            })
        else:
            final_rows.append({
                "cid": cid,
                "text": text,
                "assigned_cluster_id": None,
                "assigned_claim": None,
                "assigned_claim_segment": None,
                "assign_reason": "llm_missing",
                "llm_label": rec.get("llm_label") if rec else None,
                "match_status": rec.get("match_status") if rec else None,
            })

    out_df = pd.DataFrame(final_rows)

    # Write
    out_jsonl = f"{out_prefix}.jsonl"
    out_csv   = f"{out_prefix}.csv"

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print(f"[WRITE] {out_jsonl}")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[WRITE] {out_csv}")
    out_df.to_csv(out_csv, index=False)

    summary = {
        "total": len(out_df),
        "assigned": int((out_df["assign_reason"] == "llm_claim_assign").sum()),
        "missing": int((out_df["assign_reason"] == "llm_missing").sum()),
        "hit_rate_limit": hit_rate_limit,
        "out_jsonl": out_jsonl,
        "out_csv": out_csv,
    }

    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return out_df, summary


# -----------------------------
# Segment-restricted runner
# -----------------------------
def run_segment_restricted_assignment(
    df_items: pd.DataFrame,
    claims_df: pd.DataFrame,
    segment_col: str,
    out_prefix: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    For each segment value in df_items[segment_col], run assignment only using
    claims where claims_df['segment'] matches that segment.
    """
    ensure_segment_column(df_items, segment_col)

    # Build mapping: normalized segment -> original segment strings in claims
    claims_df = claims_df.copy()
    claims_df["_seg_norm"] = claims_df["segment"].map(normalize_label)

    # Candidate lists by segment
    seg_to_candidates: Dict[str, List[Dict[str, str]]] = {}
    for seg_norm, sub in claims_df.groupby("_seg_norm"):
        seg_to_candidates[seg_norm] = [
            {"segment": str(r.segment), "cluster_id": str(r.cluster_id), "cluster_claim": str(r.cluster_claim)}
            for r in sub.itertuples(index=False)
        ]

    # Assign per segment group
    df_items = df_items.copy()
    df_items["_seg_norm"] = df_items[segment_col].map(normalize_label)

    all_outputs = []
    hit_rate_limit_any = False
    assigned_total = 0
    missing_total = 0

    for seg_norm, df_seg in df_items.groupby("_seg_norm"):
        candidates = seg_to_candidates.get(seg_norm, [])
        print(f"[SEGMENT] '{seg_norm}' -> {len(df_seg)} items, {len(candidates)} candidate claims")

        if not candidates:
            # no candidates -> all missing for this segment
            tmp = df_seg.copy()
            tmp["assigned_cluster_id"] = None
            tmp["assigned_claim"] = None
            tmp["assigned_claim_segment"] = None
            tmp["assign_reason"] = "no_claims_for_segment"
            tmp["llm_label"] = None
            tmp["match_status"] = None
            all_outputs.append(tmp[[
                "cid", "text", "assigned_cluster_id", "assigned_claim",
                "assigned_claim_segment", "assign_reason", "llm_label", "match_status"
            ]])
            missing_total += len(tmp)
            continue

        seg_out_prefix = f"{out_prefix}__segment={seg_norm}"
        out_df_seg, summary_seg = run_assignment(
            df_seg[["cid", "text"]].copy(),
            claims_df=claims_df,
            candidate_claims=candidates,
            out_prefix=seg_out_prefix,
        )

        hit_rate_limit_any = hit_rate_limit_any or bool(summary_seg.get("hit_rate_limit"))
        assigned_total += int(summary_seg.get("assigned", 0))
        missing_total += int(summary_seg.get("missing", 0))

        all_outputs.append(out_df_seg)

        if hit_rate_limit_any:
            print("[SEGMENT] hit rate limit in a segment run; stopping remaining segments early.")
            break

    merged = pd.concat(all_outputs, ignore_index=True) if all_outputs else pd.DataFrame()

    # Write merged outputs
    out_jsonl = f"{out_prefix}.jsonl"
    out_csv   = f"{out_prefix}.csv"
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    print(f"[WRITE] merged {out_jsonl}")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in merged.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[WRITE] merged {out_csv}")
    merged.to_csv(out_csv, index=False)

    summary = {
        "total": len(merged),
        "assigned": assigned_total,
        "missing": missing_total,
        "hit_rate_limit": hit_rate_limit_any,
        "out_jsonl": out_jsonl,
        "out_csv": out_csv,
    }

    print("=== SEGMENT-RESTRICTED SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return merged, summary


# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to meta/bluesky dataset CSV")
    parser.add_argument("--dataset_kind", required=True, choices=["bluesky_raw", "meta_raw", "common"],
                        help="Input dataset schema type")
    parser.add_argument("--claims_csv", default="/Users/skamanski/Documents/GitHub/CS590-Project1/Claim_Extraction/claim_merging/merged_claims_out/merged__ALL__kept.csv",
                        help="Claims CSV (default points to your uploaded file)")
    parser.add_argument("--out_prefix", required=True, help="Output prefix (no extension)")

    parser.add_argument("--mode", required=True, choices=["plain", "segment"],
                        help="plain: dataset has cid,text; segment: dataset has cid,text,predicted_label")
    parser.add_argument("--segment_col", default="predicted_label",
                        help="Column name for segment label in segment mode (default predicted_label)")

    parser.add_argument("--assign", required=True, choices=["all", "segment_only", "both"],
                        help="Assignment method(s). In plain mode, only 'all' is valid (script will enforce).")

    parser.add_argument("--max_items", type=int, default=None, help="Optional cap for testing")
    args = parser.parse_args()

    _ensure_debug_dir()

    claims_df = load_claims(args.claims_csv)
    print(f"[LOAD] claims: {len(claims_df)} rows from {args.claims_csv}")

    df = load_text_dataset(args.input_csv, args.dataset_kind)
    print(f"[LOAD] items: {len(df)} rows from {args.input_csv} (after renaming if meta_raw)")

    if args.mode == "segment":
        df_full = pd.read_csv(args.input_csv)
        if args.dataset_kind == "meta_raw":
            df_full = df_full.rename(columns={"ad_archive_id": "cid", "ad_creative_bodies": "text"})
        # ensure cid/text cleaned
        if "cid" not in df_full.columns or "text" not in df_full.columns:
            raise RuntimeError("segment mode requires cid and text columns after any renaming.")
        df_full["cid"] = df_full["cid"].astype(str)
        df_full["text"] = df_full["text"].fillna("").astype(str).str.strip()
        ensure_segment_column(df_full, args.segment_col)
        df = df_full
        print(f"[LOAD] segment mode confirmed; found '{args.segment_col}'")

    if args.max_items is not None:
        df = df.head(args.max_items).copy()
        print(f"[INFO] limiting to first {len(df)} rows via --max_items")

    # Build global candidate claims list 
    all_candidates = [
        {"segment": str(r.segment), "cluster_id": str(r.cluster_id), "cluster_claim": strip_md_quotes(str(r.cluster_claim))}
        for r in claims_df.itertuples(index=False)
    ]

    if args.mode == "plain":
        if args.assign != "all":
            raise RuntimeError("plain mode only supports --assign all (no predicted_label to restrict).")

        run_assignment(
            df_items=df[["cid", "text"]].copy(),
            claims_df=claims_df,
            candidate_claims=all_candidates,
            out_prefix=args.out_prefix + "__allclaims",
        )
        return

    # segment mode
    if args.assign in ("all", "both"):
        out_all, _ = run_assignment(
            df_items=df[["cid", "text"]].copy(),
            claims_df=claims_df,
            candidate_claims=all_candidates,
            out_prefix=args.out_prefix + "__allclaims",
        )
    else:
        out_all = None

    if args.assign in ("segment_only", "both"):
        out_seg, _ = run_segment_restricted_assignment(
            df_items=df[["cid", "text", args.segment_col]].copy(),
            claims_df=claims_df,
            segment_col=args.segment_col,
            out_prefix=args.out_prefix + "__segmentclaims",
        )
    else:
        out_seg = None

    # If both, write a single combined CSV that includes both assignments side-by-side
    if args.assign == "both":
        # merge on cid
        comb = df[["cid", "text", args.segment_col]].copy()

        out_all_small = out_all.rename(columns={
            "assigned_cluster_id": "all_assigned_cluster_id",
            "assigned_claim": "all_assigned_claim",
            "assigned_claim_segment": "all_assigned_claim_segment",
            "assign_reason": "all_assign_reason",
            "llm_label": "all_llm_label",
            "match_status": "all_match_status",
        })[[
            "cid",
            "all_assigned_cluster_id",
            "all_assigned_claim",
            "all_assigned_claim_segment",
            "all_assign_reason",
            "all_llm_label",
            "all_match_status",
        ]]

        out_seg_small = out_seg.rename(columns={
            "assigned_cluster_id": "seg_assigned_cluster_id",
            "assigned_claim": "seg_assigned_claim",
            "assigned_claim_segment": "seg_assigned_claim_segment",
            "assign_reason": "seg_assign_reason",
            "llm_label": "seg_llm_label",
            "match_status": "seg_match_status",
        })[[
            "cid",
            "seg_assigned_cluster_id",
            "seg_assigned_claim",
            "seg_assigned_claim_segment",
            "seg_assign_reason",
            "seg_llm_label",
            "seg_match_status",
        ]]

        comb = comb.merge(out_all_small, on="cid", how="left").merge(out_seg_small, on="cid", how="left")

        out_csv = args.out_prefix + "__COMBINED__bothmethods.csv"
        out_jsonl = args.out_prefix + "__COMBINED__bothmethods.jsonl"
        print(f"[WRITE] combined {out_csv}")
        comb.to_csv(out_csv, index=False)

        print(f"[WRITE] combined {out_jsonl}")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in comb.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
