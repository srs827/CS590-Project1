#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


HUMAN_FIELDS = [
    "claim_valid_human",
    "ground_valid_human",
    "warrant_valid_human",
]

CONTEXT_COLS = [
    "ad_archive_id",
    "ad_text",
    "gpt_toulmin_json",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default="")
    ap.add_argument("--output-csv", default="")
    ap.add_argument("--key", default="pred_toulmin_json")
    ap.add_argument("--format", choices=["yesno", "bool"], default="yesno")
    ap.add_argument("--only-unlabeled", action="store_true")
    args, _ = ap.parse_known_args()
    return args


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def load_df(input_path: Path, output_path: Optional[Path]) -> pd.DataFrame:
    if output_path and output_path.exists():
        df = pd.read_csv(output_path)
    else:
        df = pd.read_csv(input_path)
    return ensure_columns(df, HUMAN_FIELDS)


def get_work_indices(df: pd.DataFrame, only_unlabeled: bool) -> List[int]:
    if not only_unlabeled:
        return list(df.index)
    mask = pd.Series(False, index=df.index)
    for c in HUMAN_FIELDS:
        mask = mask | df[c].isna()
    return list(df[mask].index)


def parse_toulmin_json(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def value_to_choice(val: Any, fmt: str) -> str:
    if pd.isna(val):
        return "Skip"
    if fmt == "bool":
        if val is True:
            return "Yes"
        if val is False:
            return "No"
    low = str(val).strip().lower()
    if low in {"yes", "true", "1"}:
        return "Yes"
    if low in {"no", "false", "0"}:
        return "No"
    return "Skip"


def choice_to_value(choice: str, fmt: str) -> Any:
    if choice == "Yes":
        return True if fmt == "bool" else "Yes"
    if choice == "No":
        return False if fmt == "bool" else "No"
    return pd.NA


def set_defaults_for_row(row: pd.Series, fmt: str) -> None:
    for field in HUMAN_FIELDS:
        st.session_state[f"{field}_choice"] = value_to_choice(row.get(field, pd.NA), fmt)


def save_df(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Claim Validation Annotation", layout="wide")

    st.title("Claim Validation Annotation")

    with st.sidebar:
        input_csv = st.text_input("Input CSV", value=args.input_csv)
        output_csv = st.text_input("Output CSV (optional)", value=args.output_csv)
        fmt = st.selectbox("Label format", options=["yesno", "bool"], index=0 if args.format == "yesno" else 1)
        only_unlabeled = st.checkbox("Only unlabeled", value=args.only_unlabeled)
        reload_clicked = st.button("Reload CSV")

    if not input_csv:
        st.info("Provide an input CSV path in the sidebar.")
        return

    input_path = Path(input_csv).expanduser()
    output_path = Path(output_csv).expanduser() if output_csv else input_path.with_name(input_path.stem + ".human.csv")

    if "df" not in st.session_state or reload_clicked:
        st.session_state["df"] = load_df(input_path, output_path)
        st.session_state["work_indices"] = get_work_indices(st.session_state["df"], only_unlabeled)
        st.session_state["pos"] = 0
        st.session_state["last_row_id"] = None

    df = st.session_state["df"]
    work_indices = get_work_indices(df, only_unlabeled)
    if not work_indices:
        st.success("No rows to annotate.")
        return

    pos = min(st.session_state.get("pos", 0), len(work_indices) - 1)
    row_id = work_indices[pos]
    row = df.loc[row_id]

    if st.session_state.get("last_row_id") != row_id:
        set_defaults_for_row(row, fmt)
        st.session_state["last_row_id"] = row_id

    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"Row {pos + 1} / {len(work_indices)} (index {row_id})")
        for col in CONTEXT_COLS:
            if col in row.index and pd.notna(row[col]):
                st.markdown(f"**{col}**")
                st.write(row[col])

        toulmin = parse_toulmin_json(row.get("pred_toulmin_json", ""))
        if toulmin:
            st.markdown("**Parsed gpt_toulmin_json**")
            st.json(toulmin)

    with right:
        st.subheader("Annotations")
        claim_choice = st.radio(
            "Claim valid?",
            ["Yes", "No", "Skip"],
            key="claim_valid_human_choice",
        )
        ground_choice = st.radio(
            "Ground valid?",
            ["Yes", "No", "Skip"],
            key="ground_valid_human_choice",
        )
        warrant_choice = st.radio(
            "Warrant valid?",
            ["Yes", "No", "Skip"],
            key="warrant_valid_human_choice",
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Prev") and pos > 0:
                st.session_state["pos"] = pos - 1
                st.rerun()
        with col_b:
            if st.button("Save"):
                df.at[row_id, "claim_valid_human"] = choice_to_value(claim_choice, fmt)
                df.at[row_id, "ground_valid_human"] = choice_to_value(ground_choice, fmt)
                df.at[row_id, "warrant_valid_human"] = choice_to_value(warrant_choice, fmt)
                save_df(df, output_path)
                st.toast(f"Saved: {output_path}")
        with col_c:
            if st.button("Next") and pos < len(work_indices) - 1:
                df.at[row_id, "claim_valid_human"] = choice_to_value(claim_choice, fmt)
                df.at[row_id, "ground_valid_human"] = choice_to_value(ground_choice, fmt)
                df.at[row_id, "warrant_valid_human"] = choice_to_value(warrant_choice, fmt)
                save_df(df, output_path)
                st.session_state["pos"] = pos + 1
                st.rerun()

        jump = st.number_input("Jump to row #", min_value=1, max_value=len(work_indices), value=pos + 1, step=1)
        if st.button("Go"):
            st.session_state["pos"] = int(jump) - 1
            st.rerun()

    st.caption(f"Output: {output_path}")


if __name__ == "__main__":
    main()
