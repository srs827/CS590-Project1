import os, json, time, re
import pandas as pd
import httpx
import numpy as np
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# Config
# ---------------------------
MODEL_ID   = os.getenv("MODEL_ID", "mistral-large-2407")
BASE_URL   = os.getenv("BASE_URL", "https://api.mistral.ai/v1")
API_KEY    = os.getenv("MISTRAL_API_KEY")
API_URL    = f"{BASE_URL}/chat/completions"

TEMPERATURE = 0.0
TOP_P       = 1.0
MAX_TOKENS  = 128

# concurrency + throttling
MAX_WORKERS = 16
SLEEP_BETWEEN_REQ = 0.0

# parsing helpers
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# ---------------------------
# I/O paths 
# ---------------------------
in_csv    = "/Users/skamanski/Documents/GitHub/CS590-Project1/Full_Datasets/meta_data.csv"
out_jsonl = "meta_results.jsonl"
out_csv   = "meta_results.csv"

# Dataset columns 
cid_col  = "ad_archive_id"
text_col = "ad_creative_bodies"

max_chars = 1200
limit = 0
resume = True

# ---------------------------
# Utils
# ---------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def extract_first_json_relaxed(s: str) -> str:
    """Extract first JSON object from a string; tolerant of code fences."""
    s = (s or "").strip()
    m = JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON start")

    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return s[start:] + ("}" * max(depth, 0))

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split())

def truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    return s[:max_chars] + (" …" if len(s) > max_chars else "")

def clamp_int(x, lo=0, hi=100, default=0) -> int:
    try:
        v = int(x)
    except Exception:
        v = default
    return max(lo, min(hi, v))

# ---------------------------
# Prompting
# ---------------------------

SYSTEM_PROMPT = """You are a careful annotator.

Task: Determine whether the given text contains an explicit or implicit claim—i.e., a declarative statement expressing a position, evaluation, recommendation, or causal assertion that could reasonably be supported or contested.

Answer with exactly one label:
- "Claim": the text asserts a position, judgment, recommendation, or causal relationship.
- "No Claim": the text is purely descriptive, factual reporting, rhetorical without asserting a position, or does not take a position.

Return STRICT JSON ONLY with keys:
claim: "Claim" or "No Claim"
confidence: integer 0-100
evidence: short quote from the text (<=20 words) justifying the label, or "" if none
reason: one short sentence justification
"""

FEWSHOT_EXAMPLES = [
    # Claim
    {
        "post": "Net zero policies will raise energy prices and hurt working families.",
        "claim": "Claim",
        "confidence": 90,
        "evidence": "will raise energy prices and hurt working families",
        "reason": "Makes a causal assertion that can be contested."
    },
    {
        "post": "We should expand solar and wind to cut emissions and improve air quality.",
        "claim": "Claim",
        "confidence": 88,
        "evidence": "should expand solar and wind",
        "reason": "Recommends an action and gives a causal rationale."
    },
    {
        "post": "The EPA’s new rule is dangerous and unnecessary.",
        "claim": "Claim",
        "confidence": 89,
        "evidence": "dangerous and unnecessary",
        "reason": "Expresses an evaluative judgment."
    },
    # No Claim
    {
        "post": "Join us this Thursday at 6pm for a community meeting about local policy updates.",
        "claim": "No Claim",
        "confidence": 85,
        "evidence": "Join us this Thursday at 6pm",
        "reason": "Administrative event information without a contestable assertion."
    },
    {
        "post": "Breaking: The bill passed the Senate 51–49 on Tuesday.",
        "claim": "No Claim",
        "confidence": 80,
        "evidence": "bill passed the Senate 51–49",
        "reason": "Factual reporting without a stated position."
    },
    {
        "post": "Wow. Just wow.",
        "claim": "No Claim",
        "confidence": 78,
        "evidence": "Wow. Just wow.",
        "reason": "Rhetorical reaction without a clear claim."
    },
]

def build_messages(post_text: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": f'Text:\n"""{ex["post"]}"""'})
        messages.append({"role": "assistant", "content": json.dumps({
            "claim": ex["claim"],
            "confidence": ex["confidence"],
            "evidence": ex["evidence"],
            "reason": ex["reason"],
        }, ensure_ascii=False)})
    messages.append({"role": "user", "content": f'Text:\n"""{post_text}"""'})
    return messages

# ---------------------------
# API call
# ---------------------------

def call_model(messages, retries=4, sleep=1.2) -> str:
    if not API_KEY:
        raise RuntimeError("Missing MISTRAL_API_KEY in environment")

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
        # try JSON mode; if unsupported, fall back
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code in (400, 404) and "response_format" in r.text.lower():
                payload.pop("response_format", None)
                r = httpx.post(API_URL, json=payload, headers=headers, timeout=120)

            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(sleep * attempt)

    raise RuntimeError(last_err or "unknown error")

def parse_response(content: str) -> Dict[str, Any]:
    raw = content or ""
    js = extract_first_json_relaxed(raw)
    obj = json.loads(js)

    label = str(obj.get("claim", "")).strip()
    if label.lower() in ("claim",):
        label = "Claim"
    elif label.lower() in ("no claim", "noclaim", "no_claim", "none"):
        label = "No Claim"
    else:
        # conservative fallback
        label = "No Claim"

    conf = clamp_int(obj.get("confidence", 0), 0, 100, 0)
    evidence = str(obj.get("evidence", "") or "")
    reason = str(obj.get("reason", "") or "")

    return {
        "claim": label,
        "confidence": conf,
        "evidence": evidence,
        "reason": reason,
        "raw": raw,
    }

# ---------------------------
# Row processing
# ---------------------------

def classify_one(row_id: int, cid: str, text: str, max_chars: int) -> Dict[str, Any]:
    text_clean = clean_text(text)
    post_text = truncate(text_clean, max_chars=max_chars)

    if not post_text:
        return {
            "row_id": row_id,
            "cid": cid,
            "text": "",
            "claim": "No Claim",
            "confidence": 0,
            "evidence": "",
            "reason": "Empty text.",
            "error": None,
        }

    try:
        messages = build_messages(post_text)
        out = call_model(messages)
        parsed = parse_response(out)
        return {
            "row_id": row_id,
            "cid": cid,
            "text": post_text,
            "claim": parsed["claim"],
            "confidence": parsed["confidence"],
            "evidence": parsed["evidence"],
            "reason": parsed["reason"],
            "error": None,
        }
    except Exception as e:
        return {
            "row_id": row_id,
            "cid": cid,
            "text": post_text,
            "claim": None,
            "confidence": None,
            "evidence": None,
            "reason": None,
            "error": f"{type(e).__name__}: {e}",
        }

# ---------------------------
# Main
# ---------------------------

def main():
    df = pd.read_csv(in_csv)
    if limit and limit > 0:
        df = df.head(limit).copy()

    seen = set()
    if resume and os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("cid") is not None:
                        seen.add(str(obj["cid"]))
                except Exception:
                    pass
        print(f"Resume enabled: {len(seen)} already processed.")

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex, open(out_jsonl, "a", encoding="utf-8") as out_f:
        for i, r in df.iterrows():
            cid = str(r.get(cid_col, "")) if cid_col in df.columns else str(i)
            if resume and cid in seen:
                continue
            text = str(r.get(text_col, ""))

            futures.append(ex.submit(classify_one, i, cid, text, max_chars))

        done = 0
        for fut in as_completed(futures):
            obj = fut.result()
            out_f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
            out_f.flush()
            done += 1
            if SLEEP_BETWEEN_REQ > 0:
                time.sleep(SLEEP_BETWEEN_REQ)
            if done % 200 == 0:
                print(f"Processed {done}/{len(futures)}...")

    # Build CSV from JSONL
    rows = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Done. Wrote JSONL -> {out_jsonl} and CSV -> {out_csv}")

if __name__ == "__main__":
    main()
