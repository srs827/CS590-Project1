#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from openai import AsyncOpenAI
from openai import RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, retry_if_exception, wait_random


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SYSTEM_PROMPT = (
    "You summarize clusters of short claims. Return JSON only with keys: "
    "topic (short noun phrase), summary (1-2 sentences), keywords (3-6 short strings). "
    "Do not include markdown or extra text."
)


def load_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"claim", "cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def pick_claims(
    df: pd.DataFrame,
    max_claims: int,
    seed: int,
    sample_mode: str,
) -> List[str]:
    claims = df["claim"].fillna("").astype(str).tolist()
    claims = [c.strip() for c in claims if c and c.strip()]
    if not claims:
        return []

    if len(claims) <= max_claims:
        return claims

    if sample_mode == "top" and "membership_prob" in df.columns:
        top_df = df.sort_values("membership_prob", ascending=False)
        claims = top_df["claim"].fillna("").astype(str).tolist()
        claims = [c.strip() for c in claims if c and c.strip()]
        return claims[:max_claims]

    rng = random.Random(seed)
    return rng.sample(claims, max_claims)


def build_user_prompt(cluster_id: int, size: int, claims: List[str]) -> str:
    items = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    return (
        f"Cluster {cluster_id} has {size} claims. "
        f"Here is a sample of {len(claims)} claims:\n{items}\n\n"
        "Return JSON only."
    )


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        topic = obj.get("topic", "").strip() or "Unknown"
        summary = obj.get("summary", "").strip() or "Unknown"
        keywords = obj.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        if not isinstance(keywords, list):
            keywords = []
        obj["topic"] = topic
        obj["summary"] = summary
        obj["keywords"] = keywords
        return obj
    except Exception:
        return None


def is_quota_or_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True
    return False


@retry(
    retry=retry_if_exception(is_quota_or_rate_limit),
    wait=wait_random(min=30, max=60),
    stop=stop_after_attempt(20),
    reraise=True,
)
async def call_openai_quota_retry(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> str:
    resp = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text.strip()


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, (APITimeoutError, APIConnectionError))),
    wait=wait_random(min=1, max=5),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def call_openai(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> str:
    try:
        return await call_openai_quota_retry(client, model, system_prompt, user_prompt, max_output_tokens)
    except RateLimitError:
        raise
    except APIStatusError:
        raise


async def summarize_clusters(df: pd.DataFrame, args) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(args.concurrency)
    results: Dict[int, Dict[str, Any]] = {}

    async with AsyncOpenAI(api_key=args.api_key) as client:

        async def worker(cluster_id: int, group: pd.DataFrame):
            size = len(group)
            claims = pick_claims(group, args.max_claims_per_cluster, args.seed, args.sample_mode)
            user_prompt = build_user_prompt(cluster_id, size, claims)
            async with sem:
                try:
                    raw = await call_openai(
                        client,
                        args.model,
                        args.system_prompt,
                        user_prompt,
                        args.max_output_tokens,
                    )
                    parsed = safe_parse_json(raw)
                    if parsed is None:
                        results[cluster_id] = {
                            "cluster_id": cluster_id,
                            "size": size,
                            "sample_size": len(claims),
                            "topic": "",
                            "summary": "",
                            "keywords": json.dumps([], ensure_ascii=False),
                            "raw": raw,
                            "error": "json_parse_failed",
                        }
                    else:
                        results[cluster_id] = {
                            "cluster_id": cluster_id,
                            "size": size,
                            "sample_size": len(claims),
                            "topic": parsed["topic"],
                            "summary": parsed["summary"],
                            "keywords": json.dumps(parsed["keywords"], ensure_ascii=False),
                            "raw": raw,
                            "error": "",
                        }
                except Exception as e:
                    results[cluster_id] = {
                        "cluster_id": cluster_id,
                        "size": size,
                        "sample_size": len(claims),
                        "topic": "",
                        "summary": "",
                        "keywords": json.dumps([], ensure_ascii=False),
                        "raw": "",
                        "error": f"{type(e).__name__}: {e}",
                    }

        tasks = []
        for cluster_id, group in df.groupby("cluster_id"):
            if not args.include_outliers and cluster_id < 0:
                continue
            if len(group) < args.min_cluster_size:
                continue
            tasks.append(asyncio.create_task(worker(int(cluster_id), group)))

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Summarizing"):
            await fut

    rows = [results[cid] for cid in sorted(results)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", required=True, help="Path to <prefix>.assignments.csv")
    ap.add_argument("--out-csv", required=True, help="Output CSV for cluster summaries")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-claims-per-cluster", type=int, default=40)
    ap.add_argument("--min-cluster-size", type=int, default=2)
    ap.add_argument("--include-outliers", action="store_true")
    ap.add_argument("--sample-mode", choices=["top", "random"], default="top")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--api-key", default=None, help="Optional OpenAI API key override")
    args = ap.parse_args()

    assignments_path = Path(args.assignments).expanduser().resolve()
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")

    df = load_assignments(assignments_path)
    rows = asyncio.run(summarize_clusters(df, args))

    out_path = Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] wrote summaries to: {out_path}")


if __name__ == "__main__":
    main()
