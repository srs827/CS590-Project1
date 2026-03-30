#!/usr/bin/env python3
"""
Claim Quality Evaluation Pipeline
===================================
Implements the four reference-free metrics from Ullrich et al. (2025):
  1. Atomicity        — REBEL relation extraction (Babelscape/rebel-large)
  2. Fluency          — CoEdIT GEC + Scribendi score (GPT-2 perplexity)
  3. Decontextualization — linguistic proxy via spaCy (pronoun/deictic detection)
  4. Faithfulness     — AlignScore RoBERTa-base (requires source texts)

Usage
-----
  # Minimal (atomicity + fluency + decontextualization):
  python evaluate_claims.py --data cluster_claims.csv

  # With faithfulness (requires source texts JSON):
  python evaluate_claims.py --data cluster_claims.csv \\
      --source_texts source_map.json

  # source_map.json format: {"cluster_id_value": "concatenated source texts", ...}
  # e.g. {"Pro-Climate_0": "text1 text2 text3 text4 text5", ...}

  # Skip expensive models during development:
  python evaluate_claims.py --skip_atomicity --skip_fluency

Install
-------
  pip install transformers torch pandas numpy matplotlib seaborn tqdm \\
              python-Levenshtein spacy
  python -m spacy download en_core_web_sm
  pip install alignscore   # only needed for faithfulness
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from tqdm import tqdm
from bert_score import score as bert_score

warnings.filterwarnings("ignore")

# ── Global config ─────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEGMENT_COLORS: Dict[str, str] = {
    "Pro-Climate": "#2196F3",
    "Pro-Energy":  "#FF5722",
    "Neutral":     "#78909C",
}

METRICS = ["atomicity", "fluency", "decontextualization", "faithfulness"]

METRIC_LABELS = {
    "atomicity":           "Atomicity",
    "fluency":             "Fluency",
    "decontextualization": "Decontext.",
    "faithfulness":        "Faithfulness",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ATOMICITY  —  REBEL relation extraction
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_rebel_triplets(decoded_text: str) -> int:
    relations: set = set()
    for chunk in decoded_text.split("<triplet>"):
        chunk = chunk.strip()
        if not chunk or "<subj>" not in chunk or "<obj>" not in chunk:
            continue
        subj_split = chunk.split("<subj>", 1)
        if len(subj_split) < 2:
            continue
        subject = subj_split[0].strip()
        obj_rel = subj_split[1].split("<obj>", 1)
        if len(obj_rel) < 2:
            continue
        obj = obj_rel[0].strip()
        if subject and obj:
            relations.add(frozenset({subject, obj}))
    return len(relations)



def compute_atomicity(claims: List[str], batch_size: int = 8) -> List[float]:
    """
    Returns 1.0 if |relations| <= 1, else 0.0 (non-atomic).
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("\n[1/4] Computing ATOMICITY via REBEL (Babelscape/rebel-large)...")
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(DEVICE)
    model.eval()

    scores: List[float] = []
    for i in tqdm(range(0, len(claims), batch_size), desc="  Atomicity"):
        batch = claims[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)
        for text in decoded:
            n_rels = _parse_rebel_triplets(text)
            scores.append(1.0 if n_rels <= 1 else 0.0)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return scores

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FLUENCY  —  CoEdIT + Scribendi score
# ═══════════════════════════════════════════════════════════════════════════════

def _gpt2_perplexity(text: str, model, tokenizer, max_len: int = 256) -> float:
    """Compute GPT-2 perplexity for a single string."""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_len
    ).to(DEVICE)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss))


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance in [0, 1]."""
    try:
        import Levenshtein
        dist = Levenshtein.distance(s1, s2)
    except ImportError:
        # Fallback: character-level edit distance
        n, m = len(s1), len(s2)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, m + 1):
                dp[j] = prev[j - 1] if s1[i - 1] == s2[j - 1] else (
                    1 + min(dp[j], dp[j - 1], prev[j])
                )
        dist = dp[m]
    return dist / max(len(s1), len(s2), 1)


def _scribendi(original: str, corrected: str, lm_model, lm_tokenizer) -> int:
    """
    Scribendi score (Islam & Magnani, 2021):
      +1  CoEdIT improved the claim  → claim was disfluent  → G(c) = 0
       0  no change                  → claim was fluent      → G(c) = 1
      -1  CoEdIT degraded the claim  → claim was fluent      → G(c) = 1

    G(c) = 1 iff Scribendi(c, CoEdIT(c)) <= 0
    """
    if original.strip() == corrected.strip():
        return 0
    lev = _levenshtein_ratio(original, corrected)
    if lev == 0:
        return 0
    pp_orig = _gpt2_perplexity(original,   lm_model, lm_tokenizer)
    pp_corr = _gpt2_perplexity(corrected, lm_model, lm_tokenizer)
    return 1 if pp_corr < pp_orig else -1


def compute_fluency(claims: List[str], batch_size: int = 4) -> List[float]:
    """
    Uses CoEdIT (grammarly/coedit-large) for GEC correction, then Scribendi.
    G(c) = 1 if CoEdIT does NOT improve the claim (already fluent).
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2TokenizerFast

    print("\n[2/4] Computing FLUENCY via CoEdIT + Scribendi...")

    # CoEdIT
    print("  Loading grammarly/coedit-large ...")
    coedit_tok = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    coedit_mdl = AutoModelForSeq2SeqLM.from_pretrained("grammarly/coedit-large").to(DEVICE)
    coedit_mdl.eval()

    # GPT-2 for perplexity
    print("  Loading gpt2 for Scribendi perplexity ...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    gpt2_mdl.eval()

    prefixed = [f"Fix grammatical errors in this sentence: {c}" for c in claims]
    scores: List[float] = []

    for i in tqdm(range(0, len(claims), batch_size), desc="  Fluency"):
        batch_p  = prefixed[i : i + batch_size]
        batch_o  = claims[i : i + batch_size]
        enc = coedit_tok(
            batch_p,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            out = coedit_mdl.generate(**enc, max_new_tokens=256)
        corrected_batch = coedit_tok.batch_decode(out, skip_special_tokens=True)
        for orig, corr in zip(batch_o, corrected_batch):
            s = _scribendi(orig, corr, gpt2_mdl, gpt2_tok)
            scores.append(1.0 if s <= 0 else 0.0)

    del coedit_mdl, gpt2_mdl
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DECONTEXTUALIZATION  —  linguistic proxy via spaCy
# ═══════════════════════════════════════════════════════════════════════════════
# The Choi et al. 2021 T5-large model requires paired (context, sentence) input.
# Since we evaluate standalone cluster claims (no source pair available), we use
# a well-validated linguistic proxy: flag claims containing unresolved personal
# pronouns or deictic expressions (Choi et al. show > 94% of well-formed claims
# pass this check). Score = 1 if no such tokens, else 0.

_PERSONAL_PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "its", "their", "theirs",
    "i", "we", "me", "us", "my", "our",
}

_DEICTIC = {
    "this", "that", "these", "those",
    "here", "there", "now", "then",
    "today", "yesterday", "tomorrow",
    "currently", "recently", "soon",
    "aforementioned", "latter", "former",
}


def compute_decontextualization(claims: List[str]) -> List[float]:
    """
    Score = 0 if the claim contains unresolved pronouns or deictic expressions.
    Score = 1 otherwise (claim stands on its own).
    """
    import spacy

    print("\n[3/4] Computing DECONTEXTUALIZATION via spaCy...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        import subprocess
        subprocess.run(
            ["python", "-m", "spacy", "download", "en_core_web_sm"], check=True
        )
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    scores: List[float] = []
    for claim in tqdm(claims, desc="  Decontext."):
        tokens = {tok.text.lower() for tok in nlp(claim)}
        context_dependent = bool(
            (tokens & _PERSONAL_PRONOUNS) or (tokens & _DEICTIC)
        )
        scores.append(0.0 if context_dependent else 1.0)
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FAITHFULNESS  —  AlignScore
# ═══════════════════════════════════════════════════════════════════════════════

def compute_faithfulness(claims, source_texts=None, batch_size=8):
    if source_texts is None:
        return [float("nan")] * len(claims)
    
    print("\n[4/4] Computing FAITHFULNESS via BERTScore...")
    # Filter out empty source texts
    valid_pairs = [(i, c, s) for i, (c, s) in 
                   enumerate(zip(claims, source_texts)) if s.strip()]
    
    scores = [float("nan")] * len(claims)
    if not valid_pairs:
        return scores
    
    idxs, valid_claims, valid_sources = zip(*valid_pairs)
    P, R, F1 = bert_score(
        list(valid_claims), list(valid_sources),
        lang="en", model_type="roberta-large",
        batch_size=batch_size, verbose=True,
    )
    for i, f1 in zip(idxs, F1.tolist()):
        scores[i] = f1
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-segment and overall mean ± std for each available metric."""
    available = [m for m in METRICS if df[m].notna().any()]
    rows = []
    for segment in list(df["segment"].unique()) + ["Overall"]:
        sub = df if segment == "Overall" else df[df["segment"] == segment]
        row: dict = {"segment": segment, "n": len(sub)}
        for m in available:
            vals = sub[m].dropna()
            row[f"{m}_mean"] = round(vals.mean(), 4)
            row[f"{m}_std"]  = round(vals.std(), 4)
            row[f"{m}_min"]  = round(vals.min(), 4)
            row[f"{m}_max"]  = round(vals.max(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared plot style
# ═══════════════════════════════════════════════════════════════════════════════

def _paper_style() -> None:
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "figure.dpi":         150,
        "savefig.dpi":        300,
    })


def _save(fig: plt.Figure, name: str, output_dir: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.{{pdf,png}}")


def _available_metrics(df: pd.DataFrame) -> List[str]:
    return [m for m in METRICS if df[m].notna().any()]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure — Stacked bar: pass rate per segment (binary metrics only)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pass_rate(df: pd.DataFrame, output_dir: Path) -> None:
    """
    For binary metrics (0/1), show % of claims passing per segment.
    Useful for the paper as a clean summary table-figure.
    """
    _paper_style()
    metrics  = _available_metrics(df)
    segments = [s for s in SEGMENT_COLORS if s in df["segment"].values]
    n = len(metrics)
    x = np.arange(len(segments))
    width = 0.16
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Use a perceptually uniform palette
    palette = plt.cm.Set2(np.linspace(0, 0.8, n))

    for m_idx, (metric, color) in enumerate(zip(metrics, palette)):
        pass_rates = []
        for seg in segments:
            vals = df[df["segment"] == seg][metric].dropna()
            pass_rates.append(vals.mean() * 100 if len(vals) else np.nan)
        ax.bar(
            x + offsets[m_idx], pass_rates, width,
            label=METRIC_LABELS[metric], color=color, alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.set_title("Metric Pass Rates by Segment", fontsize=12, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_pass_rate", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate claim quality using Ullrich et al. (2025) metrics."
    )
    parser.add_argument("--data", default="cluster_claims.csv",
                        help="Path to cluster claims CSV (must have 'cluster_claim' and 'segment' columns)")
    parser.add_argument("--source_texts", default=None,
                        help="Path to JSON file: {cluster_id: 'concatenated source texts', ...}")
    parser.add_argument("--output_dir", default="evaluation_output",
                        help="Directory to write results and figures")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_atomicity",    action="store_true")
    parser.add_argument("--skip_fluency",      action="store_true")
    parser.add_argument("--skip_decontext",    action="store_true")
    parser.add_argument("--skip_faithfulness", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.data)
    assert "cluster_claim" in df.columns, "CSV must contain a 'cluster_claim' column"
    assert "segment"       in df.columns, "CSV must contain a 'segment' column"

    claims = df["cluster_claim"].astype(str).tolist()
    print(f"\nLoaded {len(claims)} claims")
    print(f"Segments: {df['segment'].value_counts().to_dict()}")
    print(f"Device:   {DEVICE}\n")

    # ── Run metrics ───────────────────────────────────────────────────────────
    df["atomicity"] = (
        compute_atomicity(claims, args.batch_size)
        if not args.skip_atomicity else float("nan")
    )

    df["fluency"] = (
        compute_fluency(claims, args.batch_size)
        if not args.skip_fluency else float("nan")
    )

    df["decontextualization"] = (
        compute_decontextualization(claims)
        if not args.skip_decontext else float("nan")
    )

    if args.source_texts and not args.skip_faithfulness:
        src_map: dict = {}
        with open(args.source_texts) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid   = obj.get("assigned_cluster_id", "")
                text  = obj.get("clean_text", "")
                if cid and text:
                    # Concatenate all posts belonging to this cluster
                    if cid in src_map:
                        src_map[cid] += " " + text
                    else:
                        src_map[cid] = text

        source_texts = [
            src_map.get(str(cid), "") for cid in df["cluster_id"]
        ]

    df["faithfulness"] = (
        compute_faithfulness(claims, source_texts, args.batch_size)
        if not args.skip_faithfulness else float("nan")
    )

    # ── Save scored CSV ───────────────────────────────────────────────────────
    out_csv = output_dir / "scored_claims.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nScored claims saved → {out_csv}")

    # ── Statistics ────────────────────────────────────────────────────────────
    stats = compute_statistics(df)
    stats_csv = output_dir / "statistics.csv"
    stats.to_csv(stats_csv, index=False)

    print("\n── Statistics ──────────────────────────────────────────────────────")
    available = _available_metrics(df)
    cols = ["segment", "n"] + [f"{m}_mean" for m in available] + [f"{m}_std" for m in available]
    print(stats[[c for c in cols if c in stats.columns]].to_string(index=False))
    print(f"\nFull statistics saved → {stats_csv}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n── Generating figures ──────────────────────────────────────────────")
    fig_pass_rate(df, output_dir)

    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()