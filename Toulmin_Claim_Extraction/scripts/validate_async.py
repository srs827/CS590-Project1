import os
import json
import argparse
import asyncio
import random
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

from pathlib import Path
from openai import AsyncOpenAI
from openai import RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
from tenacity import retry, stop_after_attempt, retry_if_exception, wait_random


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "validate_ground_system_prompt.txt"
DEFAULT_FEWSHOT_BLOCK_PATH = BASE_DIR / "prompts" / "validate_ground_fewshot_block.txt"


def read_prompt_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt file not found: {path}") from exc


SYSTEM_PROMPT = None
FEWSHOT_BLOCK = None


def resolve_prompt_path(path_str: Optional[str], default_path: Path) -> Path:
    if not path_str:
        return default_path
    path = Path(path_str)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def build_user_prompt(ad_text: str) -> str:
    if FEWSHOT_BLOCK is None:
        raise RuntimeError("FEWSHOT_BLOCK is not loaded. Check prompt paths.")
    return f"""{FEWSHOT_BLOCK}

Now analyze the following text.

{ad_text}
Answer:
""".strip()


def is_quota_or_rate_limit(exc: Exception) -> bool:
    # OpenAI SDK raises RateLimitError for 429 conditions
    if isinstance(exc, RateLimitError):
        return True
    # Sometimes 429 can appear as APIStatusError with status_code
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True
    return False


# Requirement: if quota limit => wait 30-60 seconds, then retry.
# We implement that by retrying only on rate limit/quota errors with wait_random(30,60).
@retry(
    retry=retry_if_exception(is_quota_or_rate_limit),
    wait=wait_random(min=30, max=60),
    stop=stop_after_attempt(20),
    reraise=True,
)
async def call_openai_extract_quota_retry(
    client: AsyncOpenAI,
    model: str,
    ad_text: str,
    max_output_tokens: int = 200,
) -> Dict[str, Any]:
    resp = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ad_text)},
        ],
        max_output_tokens=max_output_tokens,
    )
    out_text = resp.output_text.strip()
    return {"raw": out_text}


# For transient network issues/timeouts, do a smaller exponential retry (fast)
@retry(
    retry=retry_if_exception(lambda e: isinstance(e, (APITimeoutError, APIConnectionError))),
    wait=wait_random(min=1, max=5),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def call_openai_extract(
    client: AsyncOpenAI,
    model: str,
    ad_text: str,
    max_output_tokens: int = 200,
) -> Dict[str, Any]:
    # First handle quota/rate-limit with the special retry behavior
    try:
        return await call_openai_extract_quota_retry(client, model, ad_text, max_output_tokens)
    except RateLimitError:
        # Should already be retried; reraise for visibility if exhausted
        print("Rate Limit!")
        raise
    except APIStatusError:
        # If it's 429, already handled. Otherwise propagate.
        print("App Status Error!")
        raise


async def process_df(client: AsyncOpenAI, df: pd.DataFrame, args):
    sem = asyncio.Semaphore(args.concurrency)

    results_raw = [None] * len(df)
    results_err = [None] * len(df)

    async def worker(i: int, text: str):
        async with sem:
            try:
                r = await call_openai_extract(client, args.model, text, args.max_output_tokens)
                results_raw[i] = r["raw"]
                results_err[i] = ""
            except Exception as e:
                # request failed -> write fallback + error message
                results_raw[i] = ""
                results_err[i] = f"{type(e).__name__}: {e}"

    texts = df[args.text_col].fillna("").astype(str).tolist()

    tasks = []
    for i, text in enumerate(texts):
        if text.strip() == "":
            results_raw[i] = ""
            continue
        tasks.append(asyncio.create_task(worker(i, text)))

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Labeling"):
        await f

    df[args.out_col] = results_raw
    df[args.out_error_col] = results_err


async def run(args):
    global SYSTEM_PROMPT, FEWSHOT_BLOCK

    system_prompt_path = resolve_prompt_path(args.system_prompt_path, DEFAULT_SYSTEM_PROMPT_PATH)
    fewshot_block_path = resolve_prompt_path(args.fewshot_block_path, DEFAULT_FEWSHOT_BLOCK_PATH)

    SYSTEM_PROMPT = read_prompt_text(system_prompt_path)
    FEWSHOT_BLOCK = read_prompt_text(fewshot_block_path)

    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    df_all = pd.read_csv(input_path)

    if args.text_col not in df_all.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Columns: {list(df_all.columns)}")

    # Apply offset + limit
    start = max(args.offset, 0)
    end = len(df_all) if args.limit is None else min(len(df_all), start + args.limit)
    df = df_all.iloc[start:end].copy()

    async with AsyncOpenAI() as client:
        await process_df(client, df, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved subset [{start}:{end}] -> {output_path}")

    if args.merge_back:
        df_all.loc[df.index, args.out_col] = df[args.out_col].values

        merged_path = output_path.with_name(output_path.stem + ".merged_full.csv")
        df_all.to_csv(merged_path, index=False)
        print(f"Saved merged full file -> {merged_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--text-col", default="input_text")
    ap.add_argument(
        "--system-prompt-path",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help="Path to system prompt text file.",
    )
    ap.add_argument(
        "--fewshot-block-path",
        default=str(DEFAULT_FEWSHOT_BLOCK_PATH),
        help="Path to few-shot block text file.",
    )

    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-output-tokens", type=int, default=256)

    # subset controls
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows after offset.")
    ap.add_argument("--offset", type=int, default=0, help="Start from this row index.")
    ap.add_argument("--merge-back", action="store_true", help="Also write outputs into the full dataset and save.")

    ap.add_argument("--out-col", default="gpt_output")
    ap.add_argument("--out-error-col", default="gpt_error")

    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
