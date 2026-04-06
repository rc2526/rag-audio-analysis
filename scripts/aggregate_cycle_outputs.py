#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR


SUMMARY_DIR = CYCLE_ANALYSIS_DIR / "summary"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_cycle_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("PMHCycle")])


def load_cycle_csvs(filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for cycle_dir in list_cycle_dirs(CYCLE_ANALYSIS_DIR):
        path = cycle_dir / filename
        if path.exists():
            frames.append(pd.read_csv(path, keep_default_na=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pct(series: pd.Series) -> float:
    if len(series.index) == 0:
        return 0.0
    return float(series.mean()) * 100.0


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def build_fidelity_by_cycle(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    if fidelity_df.empty:
        return pd.DataFrame()
    df = fidelity_df.copy()
    numeric_cols = [
        "adherence_score",
        "manual_unit_coverage",
        "subsection_coverage",
        "retrieved_evidence_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        labels = group["adherence_label"].astype(str)
        rows.append(
            {
                "cycle_id": cycle_id,
                "session_topic_rows": len(group.index),
                "mean_adherence_score": round(group["adherence_score"].mean(), 4),
                "median_adherence_score": round(group["adherence_score"].median(), 4),
                "mean_manual_unit_coverage": round(group["manual_unit_coverage"].mean(), 4),
                "mean_subsection_coverage": round(group["subsection_coverage"].mean(), 4),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
                "pct_high_adherence": round(100.0 * (labels == "high").mean(), 2),
                "pct_moderate_adherence": round(100.0 * (labels == "moderate").mean(), 2),
                "pct_low_adherence": round(100.0 * (labels == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def build_fidelity_by_topic(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    if fidelity_df.empty:
        return pd.DataFrame()
    df = fidelity_df.copy()
    for col in ["adherence_score", "manual_unit_coverage", "subsection_coverage", "retrieved_evidence_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby(["topic_id", "topic_label"], dropna=False)
    rows = []
    for (topic_id, topic_label), group in grouped:
        rows.append(
            {
                "topic_id": topic_id,
                "topic_label": topic_label,
                "session_topic_rows": len(group.index),
                "mean_adherence_score": round(group["adherence_score"].mean(), 4),
                "median_adherence_score": round(group["adherence_score"].median(), 4),
                "mean_manual_unit_coverage": round(group["manual_unit_coverage"].mean(), 4),
                "mean_subsection_coverage": round(group["subsection_coverage"].mean(), 4),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_adherence_score", "session_topic_rows"], ascending=[False, False])


def build_pi_by_cycle(pi_df: pd.DataFrame) -> pd.DataFrame:
    if pi_df.empty:
        return pd.DataFrame()
    df = pi_df.copy()
    df["retrieved_evidence_count"] = pd.to_numeric(df["retrieved_evidence_count"], errors="coerce")
    df["has_answer"] = df["answer_summary"].astype(str).str.strip().ne("")
    df["has_evidence_ref"] = df["evidence_refs"].astype(str).str.strip().ne("")
    df["no_evidence_found"] = df["has_evidence_ref"].eq(False)
    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        confidence = group["confidence"].astype(str)
        rows.append(
            {
                "cycle_id": cycle_id,
                "question_rows": len(group.index),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
                "pct_rows_with_answer": round(pct(group["has_answer"]), 2),
                "pct_rows_with_no_evidence_refs": round(pct(group["no_evidence_found"]), 2),
                "pct_confidence_high": round(100.0 * (confidence == "high").mean(), 2),
                "pct_confidence_medium": round(100.0 * (confidence == "medium").mean(), 2),
                "pct_confidence_low": round(100.0 * (confidence == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def build_pi_by_question(pi_df: pd.DataFrame) -> pd.DataFrame:
    if pi_df.empty:
        return pd.DataFrame()
    df = pi_df.copy()
    df["retrieved_evidence_count"] = pd.to_numeric(df["retrieved_evidence_count"], errors="coerce")
    df["has_answer"] = df["answer_summary"].astype(str).str.strip().ne("")
    df["has_evidence_ref"] = df["evidence_refs"].astype(str).str.strip().ne("")
    grouped = df.groupby(["question_id", "question_label"], dropna=False)
    rows = []
    for (question_id, question_label), group in grouped:
        confidence = group["confidence"].astype(str)
        rows.append(
            {
                "question_id": question_id,
                "question_label": question_label,
                "question_rows": len(group.index),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
                "pct_rows_with_answer": round(pct(group["has_answer"]), 2),
                "pct_rows_with_evidence_refs": round(pct(group["has_evidence_ref"]), 2),
                "pct_confidence_high": round(100.0 * (confidence == "high").mean(), 2),
                "pct_confidence_medium": round(100.0 * (confidence == "medium").mean(), 2),
                "pct_confidence_low": round(100.0 * (confidence == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("question_id")


def build_pi_by_cycle_and_question(pi_df: pd.DataFrame) -> pd.DataFrame:
    if pi_df.empty:
        return pd.DataFrame()
    df = pi_df.copy()
    df["retrieved_evidence_count"] = pd.to_numeric(df["retrieved_evidence_count"], errors="coerce")
    df["has_answer"] = df["answer_summary"].astype(str).str.strip().ne("")
    df["has_evidence_ref"] = df["evidence_refs"].astype(str).str.strip().ne("")
    grouped = df.groupby(["cycle_id", "question_id", "question_label"], dropna=False)
    rows = []
    for (cycle_id, question_id, question_label), group in grouped:
        rows.append(
            {
                "cycle_id": cycle_id,
                "question_id": question_id,
                "question_label": question_label,
                "question_rows": len(group.index),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
                "pct_rows_with_answer": round(pct(group["has_answer"]), 2),
                "pct_rows_with_evidence_refs": round(pct(group["has_evidence_ref"]), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["cycle_id", "question_id"])


def build_evidence_by_cycle(evidence_df: pd.DataFrame) -> pd.DataFrame:
    if evidence_df.empty:
        return pd.DataFrame()
    df = evidence_df.copy()
    for col in ["score_combined", "score_doc", "score_topic", "manual_unit_match_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        rows.append(
            {
                "cycle_id": cycle_id,
                "evidence_rows": len(group.index),
                "mean_combined_score": round(group["score_combined"].mean(), 4) if "score_combined" in group else "",
                "mean_doc_score": round(group["score_doc"].mean(), 4) if "score_doc" in group else "",
                "mean_topic_score": round(group["score_topic"].mean(), 4) if "score_topic" in group else "",
                "mean_manual_unit_match_score": round(group["manual_unit_match_score"].mean(), 4) if "manual_unit_match_score" in group else "",
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def main() -> None:
    ensure_dir(SUMMARY_DIR)

    fidelity_df = load_cycle_csvs("fidelity_summary.csv")
    pi_df = load_cycle_csvs("pi_question_answers.csv")
    evidence_df = load_cycle_csvs("topic_evidence.csv")

    write_csv(fidelity_df, SUMMARY_DIR / "table_session_topic_fidelity.csv")
    write_csv(pi_df, SUMMARY_DIR / "table_pi_question_answers.csv")
    write_csv(evidence_df, SUMMARY_DIR / "table_topic_evidence.csv")

    write_csv(build_fidelity_by_cycle(fidelity_df), SUMMARY_DIR / "summary_fidelity_by_cycle.csv")
    write_csv(build_fidelity_by_topic(fidelity_df), SUMMARY_DIR / "summary_fidelity_by_topic.csv")
    write_csv(build_pi_by_cycle(pi_df), SUMMARY_DIR / "summary_pi_questions_by_cycle.csv")
    write_csv(build_pi_by_question(pi_df), SUMMARY_DIR / "summary_pi_questions_by_type.csv")
    write_csv(build_pi_by_cycle_and_question(pi_df), SUMMARY_DIR / "summary_pi_questions_by_cycle_and_type.csv")
    write_csv(build_evidence_by_cycle(evidence_df), SUMMARY_DIR / "summary_evidence_by_cycle.csv")

    print(f"Wrote aggregate summary tables to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
