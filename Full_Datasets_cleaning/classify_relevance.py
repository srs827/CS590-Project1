import os
import time
import pandas as pd
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # set your key as an env variable
MODEL           = "gpt-4o-mini"                 # cheap + fast; swap for gpt-4o if needed
BATCH_SIZE      = 20                            # posts per API call (reduces cost)
MAX_RETRIES     = 3
RETRY_DELAY     = 5                             # seconds between retries

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Few-shot examples ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
You are classifying social media posts as RELEVANT or IRRELEVANT to climate change.

A post is RELEVANT if it discusses topics such as: climate change, global warming, carbon emissions,
fossil fuels, renewable/clean/green energy, deforestation, greenhouse gases, environmental pollution,
climate policy, climate misinformation/denial, or related scientific and political topics.

A post is IRRELEVANT if it only superficially matches climate keywords but is actually about
something unrelated — e.g. personal lifestyle, general disinformation not about climate,
financial advice, or unrelated uses of words like "clean", "green", "energy", "sustainable", "drilling".

Here are examples:

IRRELEVANT: "My bears coming over and I'm struggling to find the energy to clean my room"
IRRELEVANT: "Manifesting the best energy to you all, you're amazing and strong and this week will go well!"
IRRELEVANT: "Decoding the affluent mindset: It's not just about wealth, but a strategic approach to financial freedom and sustainable growth."
IRRELEVANT: "Disinformation in Spanish spreads faster, is less scrutinized, and is more effective than misleading posts in English."
IRRELEVANT: "Upgrade your Teaching Toolbox! Uncover free interactive tools with standout resources for teaching fiscal policy."

RELEVANT: "The archaeology of climate change: a blueprint for integrating environmental and cultural systems"
RELEVANT: "China is making significant efforts to address climate change, aiming to peak its carbon emissions before 2030 and achieve carbon neutrality by 2060"
RELEVANT: "Follow to stay informed about news and other information concerning the climate!"
RELEVANT: "The net result of German politicians' shortsightedness in phasing out nuclear power is a vastly pricier grid. The new analysis shows that if Germans simply maintained their 2002 fleet of reactors through 2022, they could have saved themselves roughly $600 billion Euros. Why so much? Well, in addition to their construction costs, renewables required expensive grid upgrades and subsidies."
RELEVANT: "I'm Rep. Diana DeGette and I want to hear from you. Is Climate Change an urgent issue? Take this 1-minute survey to share your feedback."
""".strip()


def build_prompt(texts: list[str]) -> str:
    """Build a batch classification prompt for a list of texts."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "Now classify the following posts. "
        "Reply with ONLY a numbered list matching the input, one per line, "
        "in the format:  <number>. RELEVANT  or  <number>. IRRELEVANT\n\n"
        f"{numbered}"
    )


def classify_batch(texts: list[str]) -> list[str]:
    """Send a batch of texts to GPT and return a list of labels."""
    prompt = build_prompt(texts)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            return parse_labels(raw, len(texts))
        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    # Fallback: mark all as UNKNOWN if all retries fail
    return ["UNKNOWN"] * len(texts)


def parse_labels(raw: str, expected: int) -> list[str]:
    """Parse GPT's numbered response into a list of labels."""
    labels = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if "IRRELEVANT" in upper:
            labels.append("IRRELEVANT")
        elif "RELEVANT" in upper:
            labels.append("RELEVANT")
    # Safety: pad or trim to expected length
    if len(labels) < expected:
        print(f"  Warning: got {len(labels)} labels, expected {expected}. Padding with UNKNOWN.")
        labels += ["UNKNOWN"] * (expected - len(labels))
    return labels[:expected]


def classify_dataframe(df: pd.DataFrame, text_col: str, label: str) -> pd.DataFrame:
    """Classify all rows in a dataframe and add a 'relevance' column."""
    texts   = df[text_col].fillna("").tolist()
    total   = len(texts)
    all_labels = []

    print(f"\nClassifying {total:,} posts from {label}...")
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        batch_labels = classify_batch(batch)
        all_labels.extend(batch_labels)

        done = min(start + BATCH_SIZE, total)
        print(f"  {done:,} / {total:,} classified", end="\r")
        time.sleep(0.3)  # light throttle

    print(f"\n  Done. Label breakdown:")
    result = df.copy()
    result["relevance"] = all_labels
    print(result["relevance"].value_counts().to_string())
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Bluesky
    bsky = pd.read_csv("bluesky_clean_filtered.csv")
    bsky_classified = classify_dataframe(bsky, text_col="clean_text", label="bluesky_clean_filtered.csv")
    bsky_classified.to_csv("bluesky_clean_filtered_classified.csv", index=False)
    print("  → Saved to bluesky_clean_filtered_classified.csv")

    # Meta
    meta = pd.read_csv("meta_clean_filtered.csv")
    meta_classified = classify_dataframe(meta, text_col="clean_text", label="meta_clean_filtered.csv")
    meta_classified.to_csv("meta_clean_filtered_classified.csv", index=False)
    print("  → Saved to meta_clean_filtered_classified.csv")

    print("\nAll done.")