"""
Psycholinguistic Analysis for Climate Discourse
================================================
Compares Meta ads vs Bluesky posts using three free, citable lexicons:

  1. NRC Emotion Lexicon (EmoLex) — Mohammad & Turney (2013)
     8 emotions + positive/negative valence; loaded via the `nrclex` package.
     pip install nrclex

  2. Moral Foundations Dictionary 2 (MFD2) — Frimer et al. (2019)
     care/harm, fairness/cheating, loyalty/betrayal, authority/subversion,
     purity/degradation; fetched automatically from the public GitHub repo
     and cached in ~/.cache/climate_liwc/. No account or licence required.

  3. VADER sentiment — Hutto & Gilbert (2014)
     Compound, positive, negative, neutral scores; designed for social media.
     pip install vaderSentiment

  4. Computational features (no lexicon required)
     Readability (Flesch-Kincaid), lexical diversity (TTR), modal verb rate,
     imperative/CTA rate, hashtag & mention rate, exclamation rate, named
     entity counts (spaCy).

  5. Optional: proprietary LIWC .dic file (LIWC 2015 / 2022)
     Pass --liwc_dic path/to/file.dic to layer LIWC categories on top of
     everything above.

Usage
-----
pip install nrclex vaderSentiment textstat spacy pandas scipy matplotlib seaborn requests
python -m spacy download en_core_web_sm

# Real data
python liwc_analysis.py --meta meta_ads.csv --bsky bluesky_posts.csv

# With optional LIWC licence file
python liwc_analysis.py --meta meta_ads.csv --bsky bluesky_posts.csv --liwc_dic LIWC2015.dic

# Quick smoke-test on synthetic data
python liwc_analysis.py --demo

Citations
---------
NRC EmoLex : Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a
             word-emotion association lexicon. Computational Intelligence, 29(3).
MFD2       : Frimer, J. A., Boghrati, R., Haidt, J., Graham, J., & Dehghan, M.
             (2019). Moral Foundations Dictionary for Linguistic Analyses 2.0.
             Unpublished manuscript.
VADER      : Hutto, C. J., & Gilbert, E. E. (2014). VADER: A parsimonious
             rule-based model for sentiment analysis of social media text. ICWSM.
"""

import argparse
import os
import re
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

CACHE_DIR = Path.home() / ".cache" / "climate_liwc"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"Meta": "#4A90D9", "Bluesky": "#E07B39"}


# ============================================================================
# 1.  Lexicon loaders
# ============================================================================

# ----------------------------------------------------------------------------
# 1a. NRC Emotion Lexicon
# ----------------------------------------------------------------------------

def load_nrclex() -> dict[str, set[str]]:
    """
    Load NRC EmoLex via the `nrclex` package and invert to
    {category: set_of_words}.  Returns {} if package not installed.
    """
    try:
        from nrclex import NRCLex
        sample    = NRCLex("test")
        raw       = sample.affect_dict        # {word: [emotion, ...]}
        cat2words: dict[str, set[str]] = defaultdict(set)
        for word, cats in raw.items():
            for c in cats:
                cat2words[c].add(word.lower())
        total = sum(len(v) for v in cat2words.values())
        print(f"  NRC EmoLex: {len(cat2words)} categories, {total} word-category pairs.")
        return dict(cat2words)
    except ImportError:
        print("  ⚠  nrclex not installed (pip install nrclex) — skipping NRC EmoLex.")
        return {}


def score_nrc(text: str, cat2words: dict[str, set[str]]) -> dict[str, float]:
    toks = re.findall(r"[a-z]+", text.lower())
    n    = len(toks) or 1
    tok_set = set(toks)
    return {f"nrc_{cat}": len(tok_set & words) / n * 100
            for cat, words in cat2words.items()}


# ----------------------------------------------------------------------------
# 1b. MFD2
# ----------------------------------------------------------------------------

MFD2_URL = (
    "https://raw.githubusercontent.com/"
    "medianlab/mfd2/refs/heads/main/mfd2.0.dic"
)


def _fetch_mfd2() -> str:
    """Download MFD2 .dic and cache it.  Returns raw text content."""
    cache = CACHE_DIR / "mfd2.0.dic"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    print("  Downloading MFD2 from GitHub …", end=" ", flush=True)
    try:
        with urllib.request.urlopen(MFD2_URL, timeout=20) as r:
            content = r.read().decode("utf-8", errors="ignore")
        cache.write_text(content, encoding="utf-8")
        print("done.")
        return content
    except Exception as exc:
        print(f"failed: {exc}")
        return ""


def _parse_dic(content: str) -> dict[str, list[str]]:
    """
    Parse LIWC-style .dic format shared by MFD2 and proprietary LIWC:
        %
        1   category_name
        ...
        %
        word   1
        word*  2 3
    Returns {category_name: [word_pattern, ...]}
    """
    cat_ids: dict[str, str] = {}
    word_map: dict[str, list[str]] = defaultdict(list)
    in_header = True

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "%":
            in_header = not in_header
            continue
        if in_header:
            parts = line.split("\t")
            if len(parts) >= 2:
                cat_ids[parts[0].strip()] = parts[1].strip()
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                word = parts[0].strip().lower()
                for cid in parts[1:]:
                    cid = cid.strip()
                    if cid in cat_ids:
                        word_map[cat_ids[cid]].append(word)
    return dict(word_map)


def _builtin_mfd() -> dict[str, list[str]]:
    """Minimal built-in MF word lists used only if MFD2 download fails."""
    return {
        "care.virtue":      ["care*","protect*","safe*","kind*","compassion*","nurtur*","help*"],
        "care.vice":        ["harm*","hurt*","cruel*","brutal*","suffer*","victim*","abus*"],
        "fairness.virtue":  ["fair*","justice","equal*","right*","honest*","impartial*"],
        "fairness.vice":    ["unfair*","injustice","cheat*","exploit*","discriminat*","bias*"],
        "loyalty.virtue":   ["loyal*","solidar*","together","united","communit*","patriot*"],
        "loyalty.vice":     ["betray*","traitor*","disloyal*","abandon*","selfish*"],
        "authority.virtue": ["authorit*","expert*","law*","rule*","duty*","obey*","respect*"],
        "authority.vice":   ["rebel*","defy*","subvert*","lawless*","corrupt*","violat*"],
        "purity.virtue":    ["pure*","clean*","natural*","sacred*","holy*","healthy*"],
        "purity.vice":      ["pollut*","toxic*","contaminat*","dirty","disgust*","filth*"],
    }


def load_mfd2() -> dict[str, list[str]]:
    content = _fetch_mfd2()
    parsed  = _parse_dic(content) if content else {}
    if parsed:
        total = sum(len(v) for v in parsed.values())
        print(f"  MFD2: {len(parsed)} categories, {total} word entries.")
        return parsed
    print("  ⚠  MFD2 unavailable — using built-in MF word lists.")
    return _builtin_mfd()


def score_dic(text: str, word_map: dict[str, list[str]],
              prefix: str = "") -> dict[str, float]:
    """Score text against a wildcard word-list dictionary."""
    toks = re.findall(r"[a-z]+", text.lower())
    n    = len(toks) or 1
    counts: dict[str, int] = defaultdict(int)
    for tok in toks:
        for cat, patterns in word_map.items():
            for p in patterns:
                matched = (tok.startswith(p[:-1]) if p.endswith("*") else tok == p)
                if matched:
                    counts[cat] += 1
                    break
    return {f"{prefix}{cat}": counts.get(cat, 0) / n * 100 for cat in word_map}


# ----------------------------------------------------------------------------
# 1c. VADER
# ----------------------------------------------------------------------------

def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("  VADER: loaded.")
        return SentimentIntensityAnalyzer()
    except ImportError:
        print("  ⚠  vaderSentiment not installed (pip install vaderSentiment) — skipping.")
        return None


def score_vader(text: str, sia) -> dict[str, float]:
    if sia is None:
        return {}
    return {f"vader_{k}": v for k, v in sia.polarity_scores(text).items()}


# ----------------------------------------------------------------------------
# 1d. Optional proprietary LIWC .dic
# ----------------------------------------------------------------------------

def load_liwc_dic(path: str | None) -> dict[str, list[str]]:
    if not path or not os.path.exists(path):
        return {}
    parsed = _parse_dic(Path(path).read_text(encoding="utf-8", errors="ignore"))
    total  = sum(len(v) for v in parsed.values())
    print(f"  LIWC .dic: {len(parsed)} categories, {total} word entries.")
    return parsed


# ============================================================================
# 2.  Computational / register features  (no lexicon required)
# ============================================================================

MODALS = {"can","could","may","might","must","shall","should","will","would","ought"}
CTAS   = {"join","sign","call","contact","vote","act","help","stop",
          "support","donate","share","fight","demand","urge","write","protect"}
NER_LABELS = ["ORG","PERSON","GPE","LOC","DATE","MONEY","PERCENT","LAW"]


def compute_features(texts: pd.Series) -> pd.DataFrame:
    try:
        import textstat
        has_ts = True
    except ImportError:
        has_ts = False
        print("  ⚠  textstat not installed — skipping readability.")

    try:
        import spacy
        nlp    = spacy.load("en_core_web_sm", disable=["parser"])
        has_sp = True
        print("  spaCy: loaded for NER.")
    except Exception:
        has_sp = False
        print("  ⚠  spaCy unavailable — skipping NER features.")

    rows = []
    for text in texts:
        r: dict = {}
        toks = re.findall(r"[a-z]+", text.lower())
        n    = len(toks) or 1

        r["fk_grade"]        = textstat.flesch_kincaid_grade(text) if has_ts else np.nan
        r["flesch_ease"]     = textstat.flesch_reading_ease(text)  if has_ts else np.nan
        r["avg_word_len"]    = float(np.mean([len(t) for t in toks])) if toks else 0.0
        r["ttr"]             = len(set(toks)) / n
        r["n_words"]         = n
        r["modal_rate"]      = sum(t in MODALS for t in toks) / n * 100
        r["imperative_rate"] = sum(t in CTAS   for t in toks) / n * 100
        r["question_rate"]   = text.count("?") / n * 100
        r["exclaim_rate"]    = text.count("!") / n * 100
        r["hashtag_rate"]    = len(re.findall(r"#\w+", text)) / n * 100
        r["mention_rate"]    = len(re.findall(r"@\w+", text)) / n * 100
        r["has_url"]         = int(bool(re.search(r"https?://", text)))

        if has_sp:
            doc  = nlp(text)
            ner: dict[str, int] = defaultdict(int)
            for ent in doc.ents:
                ner[ent.label_] += 1
            for lbl in NER_LABELS:
                r[f"ner_{lbl.lower()}"] = ner[lbl] / n * 100
        else:
            for lbl in NER_LABELS:
                r[f"ner_{lbl.lower()}"] = np.nan

        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================
# 3.  Statistics
# ============================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def compare_platforms(meta_df: pd.DataFrame,
                       bsky_df: pd.DataFrame,
                       feat_cols: list) -> pd.DataFrame:
    rows = []
    for col in feat_cols:
        a = meta_df[col].dropna().values
        b = bsky_df[col].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"feature":   col,
                     "meta_mean": np.mean(a), "meta_sd": np.std(a, ddof=1),
                     "bsky_mean": np.mean(b), "bsky_sd": np.std(b, ddof=1),
                     "t_stat": t, "p_value": p, "cohens_d": cohens_d(a, b)})
    df = pd.DataFrame(rows).sort_values("cohens_d", key=abs, ascending=False)
    n  = len(df)
    df["p_bonferroni"]   = (df["p_value"] * n).clip(upper=1.0)
    df["sig"]            = df["p_value"].map(
        lambda p: "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns")))
    df["sig_bonferroni"] = df["p_bonferroni"].map(
        lambda p: "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns")))
    return df


# ============================================================================
# 4.  Visualisations
# ============================================================================

def plot_top_features(stats_df: pd.DataFrame, out_path: str, top_n: int = 25):
    """Horizontal paired bar chart sorted by |Cohen's d|."""
    top = stats_df.head(top_n).sort_values("cohens_d")
    fig, ax = plt.subplots(figsize=(10, 0.42*top_n + 2))
    y, h = np.arange(len(top)), 0.35
    ax.barh(y+h/2, top["meta_mean"], h, color=PALETTE["Meta"],    label="Meta Ads",      alpha=0.85)
    ax.barh(y-h/2, top["bsky_mean"], h, color=PALETTE["Bluesky"], label="Bluesky Posts", alpha=0.85)
    xmax = max(top["meta_mean"].max(), top["bsky_mean"].max()) or 1
    for i, (_, row) in enumerate(top.iterrows()):
        if row["sig_bonferroni"] != "ns":
            ax.text(xmax*1.03, i, row["sig_bonferroni"], va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Mean score (% tokens or normalised)", fontsize=10)
    ax.set_title(f"Top {top_n} features by |Cohen's d|  (* Bonferroni-corrected)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_grouped_bars(stats_df: pd.DataFrame, groups: dict, out_path: str):
    """One subplot per lexicon group."""
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:
        axes = [axes]
    for ax, (gname, cats) in zip(axes, groups.items()):
        sub = stats_df[stats_df["feature"].isin(cats)].copy()
        if sub.empty:
            ax.set_title(f"{gname}\n(no data)"); continue
        sub    = sub.sort_values("cohens_d")
        labels = sub["feature"].str.replace(r"^(nrc_|mfd2_|vader_|liwc_)", "", regex=True)
        y, h   = np.arange(len(sub)), 0.35
        ax.barh(y+h/2, sub["meta_mean"], h, color=PALETTE["Meta"],    label="Meta",    alpha=0.85)
        ax.barh(y-h/2, sub["bsky_mean"], h, color=PALETTE["Bluesky"], label="Bluesky", alpha=0.85)
        xmax = max(sub["meta_mean"].max(), sub["bsky_mean"].max()) or 1
        for i, (_, row) in enumerate(sub.iterrows()):
            if row["sig_bonferroni"] != "ns":
                ax.text(xmax*1.05, i, row["sig_bonferroni"], va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels.values, fontsize=8)
        ax.set_title(gname, fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Platform Comparison by Lexicon Group", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_radar(stats_df: pd.DataFrame, out_path: str):
    """Spider chart across a curated cross-lexicon feature set."""
    want = ["nrc_positive","nrc_negative","nrc_fear","nrc_anger","nrc_joy","nrc_trust",
            "mfd2_care.virtue","mfd2_care.vice","mfd2_fairness.virtue",
            "mfd2_authority.virtue","mfd2_purity.virtue",
            "vader_compound","modal_rate","imperative_rate","ttr"]
    present = [c for c in want if c in stats_df["feature"].values]
    if len(present) < 4:
        print("  Not enough categories for radar — skipping."); return

    sub  = stats_df[stats_df["feature"].isin(present)].set_index("feature").reindex(present)
    mins = sub[["meta_mean","bsky_mean"]].min(axis=1)
    rng  = (sub[["meta_mean","bsky_mean"]].max(axis=1) - mins).replace(0, 1)
    mn   = list((sub["meta_mean"]-mins)/rng)
    bn   = list((sub["bsky_mean"]-mins)/rng)
    N    = len(present)
    ang  = [i/N*2*np.pi for i in range(N)] + [0]
    mn  += [mn[0]]; bn += [bn[0]]
    labs = [c.replace("nrc_","").replace("mfd2_","").replace("vader_","")
             .replace("_rate","").replace("_virtue","↑").replace("_vice","↓")
             for c in present]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    ax.plot(ang, mn, "o-", lw=2, color=PALETTE["Meta"],    label="Meta Ads")
    ax.fill(ang, mn, alpha=0.15, color=PALETTE["Meta"])
    ax.plot(ang, bn, "s-", lw=2, color=PALETTE["Bluesky"], label="Bluesky Posts")
    ax.fill(ang, bn, alpha=0.15, color=PALETTE["Bluesky"])
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs, size=8)
    ax.set_yticklabels([])
    ax.set_title("Psycholinguistic Profile\n(per-feature normalised)", size=11, y=1.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_stance_heatmap(df: pd.DataFrame, features: list,
                         out_path: str, title: str = ""):
    if "stance" not in df.columns:
        return
    present = [f for f in features if f in df.columns]
    if not present:
        return
    grouped = df.groupby("stance")[present].mean()
    z = (grouped - grouped.mean()) / grouped.std().replace(0, 1)
    h = max(3, len(grouped) * 0.7)
    w = max(8, len(present) * 0.55)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(z, annot=grouped.round(2), fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, ax=ax,
                cbar_kws={"label": "z-score (column)"})
    ax.set_title(title or "Feature Means by Stance (z-scored per column)", fontsize=11)
    ax.set_xlabel("")
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_theme_boxplot(meta_df: pd.DataFrame, bsky_df: pd.DataFrame,
                        feature: str, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, (df, name, color) in zip(axes, [
            (meta_df, "Meta Ads",      PALETTE["Meta"]),
            (bsky_df, "Bluesky Posts", PALETTE["Bluesky"])]):
        if "theme" not in df.columns or feature not in df.columns:
            ax.set_title(f"{name} — missing data"); continue
        top_th = df["theme"].value_counts().head(10).index
        sub    = df[df["theme"].isin(top_th)]
        order  = sub.groupby("theme")[feature].median().sort_values(ascending=False).index
        sns.boxplot(data=sub, y="theme", x=feature, order=order,
                    color=color, fliersize=2, ax=ax)
        ax.set_title(f"{name}: {feature}", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================================
# 5.  Main pipeline
# ============================================================================

def run_analysis(meta_path: str,
                 bsky_path: str,
                 text_col:  str        = "text",
                 out_dir:   str        = "liwc_results",
                 liwc_path: str | None = None,
                 sample_n:  int | None = None):

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print("\n=== Loading data ===")
    meta = pd.read_csv(meta_path)
    bsky = pd.read_csv(bsky_path)
    print(f"  Meta: {len(meta):,}  |  Bluesky: {len(bsky):,}")

    for df, name in [(meta, "Meta"), (bsky, "Bluesky")]:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not in {name}. "
                             f"Available: {list(df.columns)}")

    if sample_n:
        meta = meta.sample(min(sample_n, len(meta)), random_state=42).reset_index(drop=True)
        bsky = bsky.sample(min(sample_n, len(bsky)),  random_state=42).reset_index(drop=True)
        print(f"  Sampled: {len(meta)} Meta / {len(bsky)} Bluesky")

    meta[text_col] = meta[text_col].fillna("").astype(str)
    bsky[text_col] = bsky[text_col].fillna("").astype(str)

    # --------------------------------------------------------- load lexicons
    print("\n=== Loading lexicons ===")
    nrc   = load_nrclex()
    mfd2  = load_mfd2()
    vader = load_vader()
    liwc  = load_liwc_dic(liwc_path)

    # ------------------------------------------------------- score all texts
    print("\n=== Scoring texts ===")

    def score_all(texts: pd.Series) -> pd.DataFrame:
        records = []
        for text in texts:
            r: dict = {}
            if nrc:
                r.update(score_nrc(text, nrc))
            r.update(score_dic(text, mfd2,  prefix="mfd2_"))
            r.update(score_vader(text, vader))
            if liwc:
                r.update(score_dic(text, liwc, prefix="liwc_"))
            records.append(r)
        return pd.DataFrame(records)

    meta_lex = score_all(meta[text_col])
    bsky_lex = score_all(bsky[text_col])

    # Align columns between platforms
    all_lex_cols = sorted(set(meta_lex.columns) | set(bsky_lex.columns))
    meta_lex = meta_lex.reindex(columns=all_lex_cols, fill_value=0)
    bsky_lex = bsky_lex.reindex(columns=all_lex_cols, fill_value=0)

    # ----------------------------------------- computational features
    print("=== Computing computational features ===")
    meta_feat = compute_features(meta[text_col])
    bsky_feat = compute_features(bsky[text_col])

    # ---------------------------------------------------- combine everything
    def build_full(lex_df, feat_df, source_df):
        full = pd.concat([lex_df.reset_index(drop=True),
                          feat_df.reset_index(drop=True)], axis=1)
        for col in ["theme", "stance"]:
            if col in source_df.columns:
                full[col] = source_df[col].values
        return full

    meta_full = build_full(meta_lex, meta_feat, meta)
    bsky_full = build_full(bsky_lex, bsky_feat, bsky)

    # -------------------------------------------------- statistics
    print("=== Running statistical comparisons ===")
    feat_cols = [c for c in meta_full.columns if c not in ("theme","stance")]
    stats_df  = compare_platforms(meta_full, bsky_full, feat_cols)

    stats_df.to_csv(out / "platform_comparison.csv", index=False)
    meta_full.to_csv(out / "meta_scored.csv",          index=False)
    bsky_full.to_csv(out / "bluesky_scored.csv",       index=False)
    print(f"  CSVs saved to {out_dir}/")

    print("\n--- Top 20 features by |Cohen's d| ---")
    print(stats_df[["feature","meta_mean","bsky_mean","cohens_d","sig_bonferroni"]]
          .head(20).to_string(index=False, float_format="{:.3f}".format))

    # ------------------------------------------------------- plots
    print("\n=== Generating plots ===")

    plot_top_features(stats_df, out_path=str(out / "fig_top_features.png"))

    # Build lexicon groups for grouped bar chart
    groups = {}
    nrc_f  = [c for c in stats_df["feature"] if c.startswith("nrc_")]
    mfd2_f = [c for c in stats_df["feature"] if c.startswith("mfd2_")]
    vad_f  = [c for c in stats_df["feature"] if c.startswith("vader_")]
    comp_f = [c for c in stats_df["feature"]
              if not any(c.startswith(p) for p in ("nrc_","mfd2_","vader_","liwc_"))]
    liwc_f = [c for c in stats_df["feature"] if c.startswith("liwc_")]
    if nrc_f:  groups["NRC Emotions"]          = nrc_f
    if mfd2_f: groups["MFD2 Moral Foundations"] = mfd2_f
    if vad_f:  groups["VADER Sentiment"]        = vad_f
    if comp_f: groups["Computational"]          = comp_f[:12]
    if liwc_f: groups["LIWC"]                   = liwc_f[:20]
    if groups:
        plot_grouped_bars(stats_df, groups, out_path=str(out / "fig_grouped_bars.png"))

    plot_radar(stats_df, out_path=str(out / "fig_radar.png"))

    # Stance heatmaps
    stance_feats = (
        [c for c in ["nrc_positive","nrc_negative","nrc_fear","nrc_anger",
                      "nrc_joy","nrc_trust","nrc_sadness","nrc_disgust",
                      "nrc_anticipation","nrc_surprise"]
         if c in meta_full.columns] +
        [c for c in mfd2_f if c in meta_full.columns] +
        [c for c in ["vader_compound","vader_pos","vader_neg",
                      "modal_rate","imperative_rate","ttr","fk_grade"]
         if c in meta_full.columns]
    )
    plot_stance_heatmap(meta_full, stance_feats,
                         str(out / "fig_stance_heatmap_meta.png"),
                         "Meta Ads: Feature Means by Stance")
    plot_stance_heatmap(bsky_full, stance_feats,
                         str(out / "fig_stance_heatmap_bluesky.png"),
                         "Bluesky Posts: Feature Means by Stance")

    # Theme box plot for most discriminating feature
    if len(stats_df):
        top_feat = stats_df.iloc[0]["feature"]
        safe_name = re.sub(r"[^\w]", "_", top_feat)
        plot_theme_boxplot(meta_full, bsky_full, top_feat,
                            str(out / f"fig_theme_{safe_name}.png"))

    print(f"\n✓  All outputs saved to: {out_dir}/")
    return stats_df, meta_full, bsky_full


# ============================================================================
# 6.  Demo mode  (synthetic data, no CSVs needed)
# ============================================================================

def _demo_data(n: int = 300):
    import random
    random.seed(42)
    meta_t = [
        "Join us to support clean energy for a better future! Act now.",
        "Together we can build a sustainable tomorrow. Invest in solar today.",
        "Our community deserves clean air and clean water. Support this bill.",
        "Help us fight climate change. Donate and make a difference today.",
        "Great news: renewable energy is booming. We're winning the clean future.",
        "Sign the petition to protect our oceans from fossil fuel pollution.",
        "We have a plan to cut carbon emissions by 50% before 2030.",
        "Call your senator today. We must pass the Clean Energy Act now.",
    ]
    bsky_t = [
        "Scientists warn the Arctic is melting faster than predicted. We're doomed.",
        "I'm so worried about my kids growing up on a burning planet.",
        "Trump's EPA just gutted methane regulations. This is infuriating.",
        "Another wildfire. Another flood. Another year of climate inaction.",
        "The fossil fuel industry knew about climate change for decades and lied.",
        "Science under attack again. They're defunding climate research.",
        "Can't believe some people still deny climate change in 2024.",
        "Climate rollbacks are happening faster than I can keep track of.",
    ]
    meta_s  = ["Pro-Climate"]*7 + ["Neutral"]
    bsky_s  = ["Pro-Climate"]*7 + ["Neutral"]
    meta_th = ["clean energy advocacy","solar energy advocacy","clean water advocacy",
               "climate crisis advocacy","clean energy boom","ocean conservation",
               "pro-climate candidate","energy cost alarm"]
    bsky_th = ["science under attack","climate unease","climate rollbacks",
               "climate-fueled wildfires","fossil fuel corruption","science under attack",
               "climate denial backlash","climate rollbacks"]

    def rows(templates, stances, themes, n):
        data = []
        for _ in range(n):
            i = random.randrange(len(templates))
            data.append({"text":   templates[i] + " " + random.choice(templates),
                         "stance": stances[i],
                         "theme":  themes[i]})
        return pd.DataFrame(data)

    return rows(meta_t, meta_s, meta_th, n), rows(bsky_t, bsky_s, bsky_th, n)


# ============================================================================
# 7.  CLI
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Psycholinguistic analysis of climate discourse "
                    "(NRC EmoLex + MFD2 + VADER + computational features)")
    ap.add_argument("--meta",     help="Path to Meta ads CSV")
    ap.add_argument("--bsky",     help="Path to Bluesky posts CSV")
    ap.add_argument("--text_col", default="text",
                    help="Name of text column in both CSVs (default: text)")
    ap.add_argument("--liwc_dic", default=None,
                    help="Optional path to LIWC 2015/2022 .dic file")
    ap.add_argument("--out_dir",  default="liwc_results",
                    help="Output directory (default: liwc_results)")
    ap.add_argument("--sample",   type=int, default=None,
                    help="Subsample N rows per platform (default: all)")
    ap.add_argument("--demo",     action="store_true",
                    help="Run on synthetic data — no CSV files required")
    args = ap.parse_args()

    if args.demo or (args.meta is None and args.bsky is None):
        print("Running DEMO mode on synthetic data …")
        import tempfile
        mdf, bdf = _demo_data(300)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            mdf.to_csv(f, index=False); mt = f.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            bdf.to_csv(f, index=False); bt = f.name
        run_analysis(mt, bt, out_dir=args.out_dir,
                     liwc_path=args.liwc_dic, sample_n=args.sample)
        os.unlink(mt); os.unlink(bt)
    else:
        if not args.meta or not args.bsky:
            ap.error("Provide both --meta and --bsky, or use --demo.")
        run_analysis(args.meta, args.bsky,
                     text_col=args.text_col,
                     out_dir=args.out_dir,
                     liwc_path=args.liwc_dic,
                     sample_n=args.sample)