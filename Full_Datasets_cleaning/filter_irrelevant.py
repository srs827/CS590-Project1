import pandas as pd
import re

# ── Load irrelevant keywords ──────────────────────────────────────────────────
with open("irrelevant_keywords.txt", "r", encoding="utf-8") as f:
    raw_keywords = [line.strip().lower() for line in f if line.strip()]

# Build a single regex that matches any irrelevant keyword as a phrase.
# Using word boundaries where the keyword starts/ends with a word character,
# otherwise plain substring match (handles hashtags like #oc, #art).
def build_pattern(keywords):
    parts = []
    for kw in keywords:
        escaped = re.escape(kw)
        # Add word boundaries only on alphanumeric edges
        left  = r'\b' if re.match(r'\w', kw[0])  else ''
        right = r'\b' if re.match(r'\w', kw[-1]) else ''
        parts.append(f"{left}{escaped}{right}")
    return re.compile('|'.join(parts), re.IGNORECASE)

pattern = build_pattern(raw_keywords)

def is_irrelevant(text):
    """Return True if the text contains any irrelevant keyword."""
    if not isinstance(text, str):
        return False
    return bool(pattern.search(text))

# ── Filter bluesky_clean.csv ──────────────────────────────────────────────────
print("Loading bluesky_clean.csv...")
bsky = pd.read_csv("bluesky_clean.csv")

before = len(bsky)
bsky_filtered = bsky[~bsky["clean_text"].apply(is_irrelevant)].reset_index(drop=True)
after = len(bsky_filtered)
removed = before - after

bsky_filtered.to_csv("bluesky_clean_filtered.csv", index=False)
print(f"  Rows before : {before:,}")
print(f"  Rows removed: {removed:,}  ({removed/before*100:.1f}%)")
print(f"  Rows after  : {after:,}")
print(f"  → Saved to bluesky_clean_filtered.csv\n")

# ── Filter meta_clean.csv ─────────────────────────────────────────────────────
print("Loading meta_clean.csv...")
meta = pd.read_csv("meta_clean.csv")

before = len(meta)
meta_filtered = meta[~meta["clean_text"].apply(is_irrelevant)].reset_index(drop=True)
after = len(meta_filtered)
removed = before - after

meta_filtered.to_csv("meta_clean_filtered.csv", index=False)
print(f"  Rows before : {before:,}")
print(f"  Rows removed: {removed:,}  ({removed/before*100:.1f}%)")
print(f"  Rows after  : {after:,}")
print(f"  → Saved to meta_clean_filtered.csv\n")

print("Done.")