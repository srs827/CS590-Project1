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


SYSTEM_PROMPT = """You are an information extraction system.

For the given text:
- Identify the main statement being promoted.
- Identify any explicit reason or evidence mentioned.
- Identify any explicit explanation that connects the reason to the statement.

Rules:
- Extract only what is explicitly written.
- Do not add or infer information.
- If something is missing, write "Not present".
- If the text is only descriptive or factual, the main statement is "Not present".

Return STRICT JSON only:
{"claim": "...", "ground": "...", "warrant": "..."}"""

FEWSHOT_BLOCK = """
Example 1
Text:
Upgrade your Teaching Toolbox! Uncover free interactive tools with standout resources for teaching fiscal policy. Transform your lessons into engaging, unforgettable experiences.

Output:
{"claim":"Free interactive tools can transform fiscal policy lessons into engaging and effective learning experiences.","ground":"Not present","warrant":"Not present"}

Example 2
Text:
Accelerating the future of energy, together

Output:
{"claim":"Not present","ground":"Not present","warrant":"Not present"}

Example 3
Text:
Our spray foam insulation helps reduce energy waste and supports a more sustainable future.

Output:
{"claim":"Using this insulation supports a more sustainable future.","ground":"Our spray foam insulation helps reduce energy waste.","warrant":"Not present"}

Example 4 (explicit warrant case)
Text:
Our insulation reduces energy waste. Because reducing energy waste lowers carbon emissions, it helps protect the environment.

Output:
{"claim":"Using this insulation helps protect the environment.","ground":"Our insulation reduces energy waste.","warrant":"Reducing energy waste lowers carbon emissions."}
""".strip()


def build_user_prompt(ad_text: str) -> str:
    return f"""{FEWSHOT_BLOCK}

Now analyze the following text.

Text:
{ad_text}

Output:
""".strip()


def safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None
        for k in ["claim", "ground", "warrant"]:
            obj.setdefault(k, "Not present")
        return obj
    except Exception:
        return None


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
    parsed = safe_parse_json(out_text)
    return {"raw": out_text, "parsed": parsed}


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
    except RateLimitError as e:
        # Should already be retried; reraise for visibility if exhausted
        print("Rate Limit!")
        raise
    except APIStatusError as e:
        # If it's 429, already handled. Otherwise propagate.
        print("App Status Error!")
        raise


async def process_df(client: AsyncOpenAI, df: pd.DataFrame, args):
    sem = asyncio.Semaphore(args.concurrency)

    results_raw = [None] * len(df)
    results_json = [None] * len(df)
    results_err = [None] * len(df)

    async def worker(i: int, text: str):
        async with sem:
            try:
                r = await call_openai_extract(client, args.model, text, args.max_output_tokens)
                results_raw[i] = r["raw"]

                if r["parsed"] is None:
                    # parsing failed → write a safe fallback JSON
                    results_json[i] = json.dumps(
                        {"claim":"Not present","ground":"Not present","warrant":"Not present"},
                        ensure_ascii=False
                    )
                    results_err[i] = "json_parse_failed"
                else:
                    results_json[i] = json.dumps(r["parsed"], ensure_ascii=False)
                    results_err[i] = ""
            except Exception as e:
                # request failed → write fallback JSON + error message
                results_raw[i] = ""
                results_json[i] = json.dumps(
                    {"claim":"Not present","ground":"Not present","warrant":"Not present"},
                    ensure_ascii=False
                )
                results_err[i] = f"{type(e).__name__}: {e}"


    texts = df[args.text_col].fillna("").astype(str).tolist()

    tasks = []
    for i, text in enumerate(texts):
        if text.strip() == "":
            results_raw[i] = ""
            results_json[i] = json.dumps({"claim": "Not present", "ground": "Not present", "warrant": "Not present"})
            continue
        tasks.append(asyncio.create_task(worker(i, text)))

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Labeling"):
        await f

    df[args.out_raw_col] = results_raw
    df[args.out_json_col] = results_json
    df[args.out_error_col] = results_err


async def run(args):
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
        df_all.loc[df.index, args.out_raw_col] = df[args.out_raw_col].values
        df_all.loc[df.index, args.out_json_col] = df[args.out_json_col].values

        merged_path = output_path.with_name(output_path.stem + ".merged_full.csv")
        df_all.to_csv(merged_path, index=False)
        print(f"Saved merged full file -> {merged_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--text-col", default="ad_text")

    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-output-tokens", type=int, default=256)

    # NEW: subset controls
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows after offset.")
    ap.add_argument("--offset", type=int, default=0, help="Start from this row index.")
    ap.add_argument("--merge-back", action="store_true", help="Also write outputs into the full dataset and save.")

    ap.add_argument("--out-raw-col", default="gpt_raw")
    ap.add_argument("--out-json-col", default="gpt_toulmin_json")
    ap.add_argument("--out-error-col", default="gpt_error")

    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
