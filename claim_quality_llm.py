"""
Claim Quality Evaluator
=======================
Evaluates claims against source text on 4 metrics from Ullrich et al. 2025:
  - Atomicity           (0/1): Does the claim describe a single entity, relation, or process?
  - Fluency             (0/1): Is the claim grammatically correct and intelligible?
  - Decontextualization (0/1): Can the claim be interpreted without the source document?
  - Faithfulness        (0/1): Does the claim contain only info consistent with the source?

Input CSV must have columns: cid, clean_text, assigned_claim  (plus any others, preserved)
Output CSV appends: atomicity, fluency, decontextualization, faithfulness, eval_reasoning
"""

import csv
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "/Users/skamanski/Documents/GitHub/CS590-Project1/Evaluation/llm_human_eval/direct_claim_results_bluesky_gpt.csv"           # change to your file path
OUTPUT_CSV = "bluesky_claims_direct_evaluated.csv"
MODEL      = "gpt-4o-mini"
SLEEP_SEC  = 0.5                    # polite pause between API calls

SYSTEM_PROMPT = """You are a rigorous NLP evaluation assistant. You will be given:
- SOURCE: the original text a claim was derived from
- CLAIM: a statement assigned to that source

Score the CLAIM on exactly these four binary metrics:

1. atomicity (0 or 1)
   1 = the claim describes exactly ONE entity, relation, or process
   0 = the claim bundles multiple independent facts

2. fluency (0 or 1)
   1 = the claim is grammatically correct and fully intelligible
   0 = it contains grammatical errors or is hard to understand

3. decontextualization (0 or 1)
   1 = the claim can be correctly interpreted WITHOUT the source text (no dangling pronouns, no relative terms that need the source)
   0 = the claim requires the source to be understood

4. faithfulness (0 or 1)
   1 = every part of the claim is consistent with and supported by the source
   0 = the claim contains information not in or contradicted by the source

Return ONLY valid JSON in this exact schema — no markdown, no preamble:
{
  "atomicity": <0 or 1>,
  "fluency": <0 or 1>,
  "decontextualization": <0 or 1>,
  "faithfulness": <0 or 1>,
  "reasoning": "<one concise sentence explaining any 0 scores, or 'All criteria met.' if all 1>"
}"""


def build_user_message(source: str, claim: str) -> str:
    return f"SOURCE:\n{source}\n\nCLAIM:\n{claim}"


def evaluate_claim(client: OpenAI, source: str, claim: str) -> dict:
    """Call GPT-4o-mini and return parsed metric dict. Returns error dict on failure."""
    raw = ""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},  # enforces JSON output
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_message(source, claim)},
            ],
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        # Coerce to int in case the model returns booleans
        for key in ("atomicity", "fluency", "decontextualization", "faithfulness"):
            result[key] = int(result[key])
        return result
    except json.JSONDecodeError as e:
        return {
            "atomicity": -1, "fluency": -1,
            "decontextualization": -1, "faithfulness": -1,
            "reasoning": f"JSON parse error: {e} | raw: {raw[:200]}",
        }
    except Exception as e:
        return {
            "atomicity": -1, "fluency": -1,
            "decontextualization": -1, "faithfulness": -1,
            "reasoning": f"API error: {e}",
        }


def main():
    input_path  = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    client = OpenAI()  # reads OPENAI_API_KEY from environment

    new_cols = ["atomicity", "fluency", "decontextualization", "faithfulness", "eval_reasoning"]

    with input_path.open(newline="", encoding="utf-8") as fin, \
         output_path.open("w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        if "clean_text" not in reader.fieldnames or "generated_claim" not in reader.fieldnames:
            print("[ERROR] CSV must contain 'clean_text' and 'generated_claim' columns.")
            sys.exit(1)

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames + new_cols)
        writer.writeheader()

        for i, row in enumerate(reader, start=1):
            source = row["clean_text"].strip()
            claim  = row["generated_claim"].strip()
            cid    = row.get("cid", i)

            print(f"[{i}] cid={cid} — evaluating...", end=" ", flush=True)
            metrics = evaluate_claim(client, source, claim)

            row["atomicity"]           = metrics["atomicity"]
            row["fluency"]             = metrics["fluency"]
            row["decontextualization"] = metrics["decontextualization"]
            row["faithfulness"]        = metrics["faithfulness"]
            row["eval_reasoning"]      = metrics.get("reasoning", "")

            writer.writerow(row)
            print(
                f"A={metrics['atomicity']} F={metrics['fluency']} "
                f"D={metrics['decontextualization']} Faith={metrics['faithfulness']}"
            )
            time.sleep(SLEEP_SEC)

    print(f"\n Done. Results written to: {output_path}")


if __name__ == "__main__":
    main()
