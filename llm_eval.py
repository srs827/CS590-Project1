import os
import time
import httpx
import pandas as pd

INPUT_CSV  = "/Users/skamanski/Documents/GitHub/CS590-Project1/Evaluation/llm_human_eval/meta_sample.csv"
OUTPUT_CSV = "claim_eval_results_meta.csv"

API_KEY  = os.getenv("MISTRAL_API_KEY")
BASE_URL = "https://api.mistral.ai/v1"
API_URL  = f"{BASE_URL}/chat/completions"
MODEL_ID = "mistral-large-2407"
SLEEP    = 0.1

CLAIM_SYSTEM = (
    "A 'claim' is a specific assertable statement about the world — it has a subject, a predicate, "
    "and a concrete point being made (e.g. 'Fossil fuel subsidies are driving up carbon emissions and should be eliminated').\n\n"
    "Your task: determine whether the text expresses the same or a related assertion as the given claim.\n\n"
    "Output 'Yes' if any of the following are true:\n"
    "  - The text makes the same or a similar point as the claim.\n"
    "  - The text addresses the same subject and implies a stance consistent with the claim.\n"
    "  - The claim captures the general thrust of what the text is about, even if the text is less specific.\n"
    "  - The text argues against a position that the claim opposes, making the claim the logical subtext.\n\n"
    "Output 'No' if:\n"
    "  - The text is entirely about a different subject with no meaningful overlap with the claim.\n"
    "  - The text is a call to action, event announcement, or general statement with no assertable content matching the claim.\n"
    "  - The claim is absent, generic, or cannot be inferred from the text.\n\n"
    "Output only Yes or No."
)


def is_empty(val) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "n/a", "na", "[]", "{}", "()", "empty"}


def yesno(reply: str) -> str:
    if not reply:
        return "No"
    return "Yes" if "yes" in reply.strip().lower() else "No"


def ask(system_prompt: str, user_prompt: str) -> str:
    if not API_KEY:
        raise RuntimeError("Missing MISTRAL_API_KEY. Set it with: export MISTRAL_API_KEY='...'")

    payload = {
        "model": MODEL_ID,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }

    resp = httpx.post(API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

    results = []

    for i, row in df.iterrows():
        cid   = row.get("cid", str(i))
        text  = row.get("clean_text", "")
        claim = row.get("assigned_claim", "")

        if is_empty(text) or is_empty(claim):
            verdict = "No"
            print(f"[{i+1}/{len(df)}] {cid} -> No (empty text or claim)")
        else:
            user_prompt = f"text:\n{text}\n\nassigned_claim: {claim}"
            verdict = yesno(ask(CLAIM_SYSTEM, user_prompt))
            print(f"[{i+1}/{len(df)}] {cid} -> {verdict}")
            time.sleep(SLEEP)

        results.append(verdict)

    df["llm_judgment"] = results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()