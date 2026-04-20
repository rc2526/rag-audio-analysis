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


def build_session_fidelity_by_cycle(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    if fidelity_df.empty:
        return pd.DataFrame()
    df = fidelity_df.copy()
    numeric_cols = [
        "adherence_score",
        "manual_unit_coverage",
        "subsection_coverage",
        "evidence_density",
        "retrieved_evidence_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        labels = group["adherence_label"].astype(str)
        rows.append(
            {
                "cycle_id": cycle_id,
                "manual_session_rows": len(group.index),
                "mean_adherence_score": round(group["adherence_score"].mean(), 4),
                "median_adherence_score": round(group["adherence_score"].median(), 4),
                "mean_manual_unit_coverage": round(group["manual_unit_coverage"].mean(), 4),
                "mean_subsection_coverage": round(group["subsection_coverage"].mean(), 4),
                "mean_evidence_density": round(group["evidence_density"].mean(), 4),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
                "pct_high_adherence": round(100.0 * (labels == "high").mean(), 2),
                "pct_moderate_adherence": round(100.0 * (labels == "moderate").mean(), 2),
                "pct_low_adherence": round(100.0 * (labels == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def build_session_fidelity_by_manual_session(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    if fidelity_df.empty:
        return pd.DataFrame()
    df = fidelity_df.copy()
    for col in ["adherence_score", "manual_unit_coverage", "subsection_coverage", "evidence_density", "retrieved_evidence_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby(["manual_session_num", "manual_session_label"], dropna=False)
    rows = []
    for (manual_session_num, manual_session_label), group in grouped:
        rows.append(
            {
                "manual_session_num": manual_session_num,
                "manual_session_label": manual_session_label,
                "cycle_rows": len(group.index),
                "mean_adherence_score": round(group["adherence_score"].mean(), 4),
                "median_adherence_score": round(group["adherence_score"].median(), 4),
                "mean_manual_unit_coverage": round(group["manual_unit_coverage"].mean(), 4),
                "mean_subsection_coverage": round(group["subsection_coverage"].mean(), 4),
                "mean_evidence_density": round(group["evidence_density"].mean(), 4),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["manual_session_num"])


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


def build_pi_by_topic(pi_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PI answers by topic (topic_id, topic_label).

    Produces per-topic counts, mean retrieved evidence, percent rows with answers,
    percent rows with evidence references, and confidence breakdown when available.
    """
    if pi_df.empty:
        return pd.DataFrame()
    df = pi_df.copy()
    # normalize numeric and presence columns
    if "retrieved_evidence_count" in df.columns:
        df["retrieved_evidence_count"] = pd.to_numeric(df["retrieved_evidence_count"], errors="coerce")
    df["has_answer"] = df.get("answer_summary", "").astype(str).str.strip().ne("")
    df["has_evidence_ref"] = df.get("evidence_refs", "").astype(str).str.strip().ne("")

    grouped = df.groupby(["topic_id", "topic_label"], dropna=False)
    rows = []
    for (topic_id, topic_label), group in grouped:
        # collect unique session identifiers contributing to this topic
        session_nums = sorted({str(x) for x in group.get("session_num", []) if str(x).strip()})
        session_labels = sorted({str(x) for x in group.get("session_label", []) if str(x).strip()})
        # confidence column may or may not exist in PI outputs
        conf_col = "confidence" in group.columns
        confidence = group["confidence"].astype(str) if conf_col else None
        rows.append(
            {
                "topic_id": topic_id,
                "topic_label": topic_label,
                "session_nums": ";".join(session_nums) if session_nums else "",
                "session_labels": ";".join(session_labels) if session_labels else "",
                "pi_rows": len(group.index),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2) if "retrieved_evidence_count" in group else "",
                "pct_rows_with_answer": round(pct(group["has_answer"]), 2),
                "pct_rows_with_evidence_refs": round(pct(group["has_evidence_ref"]), 2),
                "pct_confidence_high": round(100.0 * (confidence == "high").mean(), 2) if conf_col else "",
                "pct_confidence_medium": round(100.0 * (confidence == "medium").mean(), 2) if conf_col else "",
                "pct_confidence_low": round(100.0 * (confidence == "low").mean(), 2) if conf_col else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["topic_label"])


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


def build_pi_by_cycle_and_topic(pi_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PI answers by (cycle_id, topic_id, topic_label).

    Produces per-cycle-per-topic counts and same summary metrics as other PI tables.
    """
    if pi_df.empty:
        return pd.DataFrame()
    df = pi_df.copy()
    if "retrieved_evidence_count" in df.columns:
        df["retrieved_evidence_count"] = pd.to_numeric(df["retrieved_evidence_count"], errors="coerce")
    df["has_answer"] = df.get("answer_summary", "").astype(str).str.strip().ne("")
    df["has_evidence_ref"] = df.get("evidence_refs", "").astype(str).str.strip().ne("")

    grouped = df.groupby(["cycle_id", "topic_id", "topic_label"], dropna=False)
    rows = []
    for (cycle_id, topic_id, topic_label), group in grouped:
        # collect unique session identifiers for this (cycle, topic)
        session_nums = sorted({str(x) for x in group.get("session_num", []) if str(x).strip()})
        session_labels = sorted({str(x) for x in group.get("session_label", []) if str(x).strip()})
        confidence = group["confidence"].astype(str) if "confidence" in group.columns else None
        rows.append(
            {
                "cycle_id": cycle_id,
                "topic_id": topic_id,
                "topic_label": topic_label,
                "session_nums": ";".join(session_nums) if session_nums else "",
                "session_labels": ";".join(session_labels) if session_labels else "",
                "pi_rows": len(group.index),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2) if "retrieved_evidence_count" in group else "",
                "pct_rows_with_answer": round(pct(group["has_answer"]), 2),
                "pct_rows_with_evidence_refs": round(pct(group["has_evidence_ref"]), 2),
                "pct_confidence_high": round(100.0 * (confidence == "high").mean(), 2) if confidence is not None else "",
                "pct_confidence_medium": round(100.0 * (confidence == "medium").mean(), 2) if confidence is not None else "",
                "pct_confidence_low": round(100.0 * (confidence == "low").mean(), 2) if confidence is not None else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["cycle_id", "topic_label"])


def build_pi_by_cycle_question_topic(pi_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PI answers by (cycle_id, question_id, topic_id).

    Produces per-(cycle,question,topic) counts and summary metrics similar to
    other PI tables so you can inspect how each question performs within a
    topic across cycles.
    """
    if pi_df.empty:
        return pd.DataFrame()
    df = pi_df.copy()
    if "retrieved_evidence_count" in df.columns:
        df["retrieved_evidence_count"] = pd.to_numeric(df["retrieved_evidence_count"], errors="coerce")
    df["has_answer"] = df.get("answer_summary", "").astype(str).str.strip().ne("")
    df["has_evidence_ref"] = df.get("evidence_refs", "").astype(str).str.strip().ne("")

    grouped = df.groupby(["cycle_id", "question_id", "question_label", "topic_id", "topic_label"], dropna=False)
    rows = []
    for (cycle_id, question_id, question_label, topic_id, topic_label), group in grouped:
        confidence = group["confidence"].astype(str) if "confidence" in group.columns else None
        rows.append(
            {
                "cycle_id": cycle_id,
                "question_id": question_id,
                "question_label": question_label,
                "topic_id": topic_id,
                "topic_label": topic_label,
                "rows": len(group.index),
                "mean_retrieved_evidence_count": round(group["retrieved_evidence_count"].mean(), 2) if "retrieved_evidence_count" in group else "",
                "pct_rows_with_answer": round(pct(group["has_answer"]), 2),
                "pct_rows_with_evidence_refs": round(pct(group["has_evidence_ref"]), 2),
                "pct_confidence_high": round(100.0 * (confidence == "high").mean(), 2) if confidence is not None else "",
                "pct_confidence_medium": round(100.0 * (confidence == "medium").mean(), 2) if confidence is not None else "",
                "pct_confidence_low": round(100.0 * (confidence == "low").mean(), 2) if confidence is not None else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["cycle_id", "question_id", "topic_label"])


def build_adjudication_by_cycle(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjudication labels (generation grade) by cycle.

    Produces percent of rows with adjudication_label == high/moderate/low per cycle.
    """
    if fidelity_df.empty or "adjudication_label" not in fidelity_df.columns:
        return pd.DataFrame()
    df = fidelity_df.copy()
    labels = df["adjudication_label"].astype(str)
    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        lbl = group["adjudication_label"].astype(str)
        rows.append(
            {
                "cycle_id": cycle_id,
                "rows": len(group.index),
                "pct_adjud_high": round(100.0 * (lbl == "high").mean(), 2),
                "pct_adjud_moderate": round(100.0 * (lbl == "moderate").mean(), 2),
                "pct_adjud_low": round(100.0 * (lbl == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def build_adjudication_confidence_by_cycle(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjudication confidence distribution (low/medium/high) by cycle."""
    if fidelity_df.empty or "adjudication_confidence" not in fidelity_df.columns:
        return pd.DataFrame()
    df = fidelity_df.copy()
    conf = df["adjudication_confidence"].astype(str)
    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        c = group["adjudication_confidence"].astype(str)
        rows.append(
            {
                "cycle_id": cycle_id,
                "rows": len(group.index),
                "pct_conf_high": round(100.0 * (c.str.lower() == "high").mean(), 2),
                "pct_conf_medium": round(100.0 * (c.str.lower() == "medium").mean(), 2),
                "pct_conf_low": round(100.0 * (c.str.lower() == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def build_adjudication_by_manual_session(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjudication labels (generation grade) by manual session.

    Produces percent of rows with adjudication_label == high/moderate/low per
    manual_session_num/manual_session_label.
    """
    if fidelity_df.empty or "adjudication_label" not in fidelity_df.columns:
        return pd.DataFrame()
    df = fidelity_df.copy()
    grouped = df.groupby(["manual_session_num", "manual_session_label"], dropna=False)
    rows = []
    for (manual_session_num, manual_session_label), group in grouped:
        lbl = group["adjudication_label"].astype(str)
        rows.append(
            {
                "manual_session_num": manual_session_num,
                "manual_session_label": manual_session_label,
                "rows": len(group.index),
                "pct_adjud_high": round(100.0 * (lbl == "high").mean(), 2),
                "pct_adjud_moderate": round(100.0 * (lbl == "moderate").mean(), 2),
                "pct_adjud_low": round(100.0 * (lbl == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["manual_session_num"])


def build_adjudication_confidence_by_manual_session(fidelity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjudication confidence distribution (low/medium/high) by manual session."""
    if fidelity_df.empty or "adjudication_confidence" not in fidelity_df.columns:
        return pd.DataFrame()
    df = fidelity_df.copy()
    grouped = df.groupby(["manual_session_num", "manual_session_label"], dropna=False)
    rows = []
    for (manual_session_num, manual_session_label), group in grouped:
        c = group["adjudication_confidence"].astype(str).str.lower()
        rows.append(
            {
                "manual_session_num": manual_session_num,
                "manual_session_label": manual_session_label,
                "rows": len(group.index),
                "pct_conf_high": round(100.0 * (c == "high").mean(), 2),
                "pct_conf_medium": round(100.0 * (c == "medium").mean(), 2),
                "pct_conf_low": round(100.0 * (c == "low").mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["manual_session_num"])


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


def build_evidence_by_cycle_split(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-cycle evidence counts split by analysis_mode.

    Produces columns: cycle_id, evidence_rows, evidence_rows_pi_question,
    evidence_rows_fidelity and preserves mean_combined_score computed over all rows
    to remain backward compatible.
    """
    if evidence_df.empty:
        return pd.DataFrame()
    df = evidence_df.copy()
    # Ensure score columns are numeric for aggregation
    for col in ["score_combined", "score_doc", "score_topic", "manual_unit_match_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = df.groupby("cycle_id", dropna=False)
    rows = []
    for cycle_id, group in grouped:
        # counts by mode
        mode_counts = group["analysis_mode"].astype(str).fillna("").value_counts()
        pi_count = int(mode_counts.get("pi_question", 0))
        fid_count = int(mode_counts.get("fidelity", 0))
        rows.append(
            {
                "cycle_id": cycle_id,
                "evidence_rows": len(group.index),
                "evidence_rows_pi_question": pi_count,
                "evidence_rows_fidelity": fid_count,
                "mean_combined_score": round(group["score_combined"].mean(), 4) if "score_combined" in group else "",
                "mean_doc_score": round(group["score_doc"].mean(), 4) if "score_doc" in group else "",
                "mean_topic_score": round(group["score_topic"].mean(), 4) if "score_topic" in group else "",
                "mean_manual_unit_match_score": round(group["manual_unit_match_score"].mean(), 4) if "manual_unit_match_score" in group else "",
            }
        )
    return pd.DataFrame(rows).sort_values("cycle_id")


def main() -> None:
    ensure_dir(SUMMARY_DIR)

    # create group subdirectories for clearer organization
    FIDELITY_SUMMARY_DIR = SUMMARY_DIR / "fidelity"
    PI_SUMMARY_DIR = SUMMARY_DIR / "pi_questions"
    ensure_dir(FIDELITY_SUMMARY_DIR)
    ensure_dir(PI_SUMMARY_DIR)

    fidelity_df = load_cycle_csvs("fidelity_summary.csv")
    session_fidelity_df = load_cycle_csvs("session_fidelity_summary.csv")
    pi_df = load_cycle_csvs("pi_question_answers.csv")
    evidence_df = load_cycle_csvs("topic_evidence.csv")
    session_evidence_df = load_cycle_csvs("session_fidelity_evidence.csv")

    # table-level outputs (also duplicate into group folders for UI grouping)

    write_csv(session_fidelity_df, SUMMARY_DIR / "table_cycle_manual_session_fidelity.csv")
    write_csv(session_fidelity_df, FIDELITY_SUMMARY_DIR / "table_cycle_manual_session_fidelity.csv")

    write_csv(pi_df, SUMMARY_DIR / "table_pi_question_answers.csv")
    write_csv(pi_df, PI_SUMMARY_DIR / "table_pi_question_answers.csv")

    #write_csv(evidence_df, SUMMARY_DIR / "table_topic_evidence.csv")
    #write_csv(evidence_df, FIDELITY_SUMMARY_DIR / "table_topic_evidence.csv")

    write_csv(session_evidence_df, SUMMARY_DIR / "table_cycle_manual_session_evidence.csv")
    write_csv(session_evidence_df, FIDELITY_SUMMARY_DIR / "table_cycle_manual_session_evidence.csv")

    # Fidelity summaries (written to both root and grouped fidelity folder)
    write_csv(build_session_fidelity_by_cycle(session_fidelity_df), SUMMARY_DIR / "summary_session_fidelity_by_cycle.csv")
    write_csv(build_session_fidelity_by_cycle(session_fidelity_df), FIDELITY_SUMMARY_DIR / "summary_session_fidelity_by_cycle.csv")
    write_csv(build_session_fidelity_by_manual_session(session_fidelity_df), SUMMARY_DIR / "summary_session_fidelity_by_manual_session.csv")
    write_csv(build_session_fidelity_by_manual_session(session_fidelity_df), FIDELITY_SUMMARY_DIR / "summary_session_fidelity_by_manual_session.csv")

    # PI summaries (written to both root and grouped pi_questions folder)
    write_csv(build_pi_by_cycle(pi_df), SUMMARY_DIR / "summary_pi_questions_by_cycle.csv")
    write_csv(build_pi_by_cycle(pi_df), PI_SUMMARY_DIR / "summary_pi_questions_by_cycle.csv")
    write_csv(build_pi_by_question(pi_df), SUMMARY_DIR / "summary_pi_questions_by_type.csv")
    write_csv(build_pi_by_question(pi_df), PI_SUMMARY_DIR / "summary_pi_questions_by_type.csv")
    write_csv(build_pi_by_cycle_and_question(pi_df), SUMMARY_DIR / "summary_pi_questions_by_cycle_and_type.csv")
    write_csv(build_pi_by_cycle_and_question(pi_df), PI_SUMMARY_DIR / "summary_pi_questions_by_cycle_and_type.csv")
    write_csv(build_pi_by_topic(pi_df), SUMMARY_DIR / "summary_pi_questions_by_topic.csv")
    write_csv(build_pi_by_topic(pi_df), PI_SUMMARY_DIR / "summary_pi_questions_by_topic.csv")
    write_csv(build_pi_by_cycle_and_topic(pi_df), SUMMARY_DIR / "summary_pi_questions_by_cycle_and_topic.csv")
    write_csv(build_pi_by_cycle_and_topic(pi_df), PI_SUMMARY_DIR / "summary_pi_questions_by_cycle_and_topic.csv")
    # PI by (cycle, question, topic) for fine-grained inspection
    write_csv(build_pi_by_cycle_question_topic(pi_df), SUMMARY_DIR / "summary_pi_questions_by_cycle_question_topic.csv")
    write_csv(build_pi_by_cycle_question_topic(pi_df), PI_SUMMARY_DIR / "summary_pi_questions_by_cycle_question_topic.csv")

    # Evidence and other fidelity-adjacent summaries
    # Write the legacy combined evidence summary (backwards compatible)
    #write_csv(build_evidence_by_cycle(evidence_df), SUMMARY_DIR / "summary_evidence_by_cycle.csv")
    #write_csv(build_evidence_by_cycle(evidence_df), FIDELITY_SUMMARY_DIR / "summary_evidence_by_cycle.csv")
    # Also write a split-by-mode evidence summary and a PI-only concatenated table
    #write_csv(build_evidence_by_cycle_split(evidence_df), SUMMARY_DIR / "summary_evidence_by_cycle_by_mode.csv")
    #write_csv(build_evidence_by_cycle_split(evidence_df), FIDELITY_SUMMARY_DIR / "summary_evidence_by_cycle_by_mode.csv")
    # PI-only table for downstream consumers that expect only PI-question evidence
    pi_only = evidence_df[evidence_df.get("analysis_mode","")=="pi_question"].copy() if not evidence_df.empty else evidence_df
    write_csv(pi_only, SUMMARY_DIR / "table_topic_evidence_pionly.csv")
    write_csv(pi_only, PI_SUMMARY_DIR / "table_topic_evidence_pionly.csv")
    # Also write a PI-only per-cycle evidence summary (ignore fidelity rows)
    write_csv(build_evidence_by_cycle(pi_only), SUMMARY_DIR / "summary_evidence_by_cycle_pionly.csv")
    write_csv(build_evidence_by_cycle(pi_only), PI_SUMMARY_DIR / "summary_evidence_by_cycle_pionly.csv")

    # New: adjudication-level aggregates (generation grade + confidence) by cycle
    #write_csv(build_adjudication_by_cycle(fidelity_df), SUMMARY_DIR / "summary_adjudication_by_cycle.csv")
    #write_csv(build_adjudication_by_cycle(fidelity_df), FIDELITY_SUMMARY_DIR / "summary_adjudication_by_cycle.csv")
    #write_csv(build_adjudication_confidence_by_cycle(fidelity_df), SUMMARY_DIR / "summary_adjudication_confidence_by_cycle.csv")
    #write_csv(build_adjudication_confidence_by_cycle(fidelity_df), FIDELITY_SUMMARY_DIR / "summary_adjudication_confidence_by_cycle.csv")

    # Also produce adjudication summaries computed from the session-level fidelity table
    # (session_fidelity_summary.csv -> session-level adjudication rollups)
    write_csv(build_adjudication_by_cycle(session_fidelity_df), SUMMARY_DIR / "summary_adjudication_by_cycle_session.csv")
    write_csv(build_adjudication_confidence_by_cycle(session_fidelity_df), SUMMARY_DIR / "summary_adjudication_confidence_by_cycle_session.csv")
    write_csv(build_adjudication_by_cycle(session_fidelity_df), FIDELITY_SUMMARY_DIR / "summary_adjudication_by_cycle_session.csv")
    write_csv(build_adjudication_confidence_by_cycle(session_fidelity_df), FIDELITY_SUMMARY_DIR / "summary_adjudication_confidence_by_cycle_session.csv")

    # Also produce adjudication summaries rolled up by manual session (across cycles)
    write_csv(build_adjudication_by_manual_session(session_fidelity_df), SUMMARY_DIR / "summary_adjudication_by_manual_session.csv")
    write_csv(build_adjudication_confidence_by_manual_session(session_fidelity_df), SUMMARY_DIR / "summary_adjudication_confidence_by_manual_session.csv")
    write_csv(build_adjudication_by_manual_session(session_fidelity_df), FIDELITY_SUMMARY_DIR / "summary_adjudication_by_manual_session.csv")
    write_csv(build_adjudication_confidence_by_manual_session(session_fidelity_df), FIDELITY_SUMMARY_DIR / "summary_adjudication_confidence_by_manual_session.csv")

    print(f"Wrote aggregate summary tables to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
