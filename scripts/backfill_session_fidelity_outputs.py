#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Any

import pandas as pd

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR, TOPIC_CATALOG_CSV
from rag_audio_analysis.settings import get_float, get_int, get_str
from rag_audio_analysis.source_bridge import (
    build_doc_index_by_path,
    expand_transcript_context,
    get_manual_units_for_session,
    get_rag_index_rows,
    get_session_summary,
    infer_manual_unit_for_text,
    infer_session_id,
    query_evidence,
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def list_cycle_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted([path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("PMHCycle")])


def build_session_fidelity_for_cycle(
    cycle_dir: Path,
    topic_catalog: pd.DataFrame,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_rows: list[dict[str, str]] = []
    evidence_rows: list[dict[str, str]] = []

    manual_weight = get_float("fidelity", "manual_coverage_weight", 0.6)
    subsection_weight = get_float("fidelity", "subsection_coverage_weight", 0.4)
    high_cutoff = get_float("fidelity", "adherence_high_cutoff", 0.66)
    moderate_cutoff = get_float("fidelity", "adherence_moderate_cutoff", 0.33)
    fidelity_topk = get_int("cycle_analysis", "fidelity_topk", 12)
    fidelity_weight_doc = get_float("cycle_analysis", "fidelity_weight_doc", 0.5)
    fidelity_weight_topic = get_float("cycle_analysis", "fidelity_weight_topic", 0.5)
    context_window = get_int("transcript_export", "context_window", 2)
    meta_rows = get_rag_index_rows()
    path_lookup = build_doc_index_by_path(meta_rows)

    session_numbers = sorted(
        x
        for x in topic_catalog["session_num"].astype(str).unique()
        if x and x.isdigit()
    )
    for session_num in session_numbers:
        session_manual_units = get_manual_units_for_session(session_num)
        session_topics = topic_catalog[topic_catalog["session_num"].astype(str) == session_num].copy()
        session_summary_row = get_session_summary(session_num)
        session_summary = session_summary_row.get("session_summary", "")
        topic_ids = [x for x in session_topics["topic_id"].astype(str).tolist() if x]
        topic_labels = [x for x in session_topics["topic_label"].astype(str).tolist() if x]
        fidelity_query = session_summary or get_str(
            "fidelity",
            "fidelity_query_template",
            "Session {session_num} {topic_label}",
        ).format(session_num=session_num, topic_label=" ; ".join(topic_labels)).strip()

        raw_results = query_evidence(
            fidelity_query,
            topk=fidelity_topk,
            weight_doc=fidelity_weight_doc,
            weight_topic=fidelity_weight_topic,
            cycle_id=cycle_dir.name,
            transcript_only=True,
        )

        windows: list[dict[str, Any]] = []
        seen_doc_indices: set[int] = set()
        for result in raw_results:
            doc_index = int(result.get("doc_index", -1))
            if doc_index < 0 or doc_index in seen_doc_indices:
                continue
            seen_doc_indices.add(doc_index)
            context = expand_transcript_context(
                doc_index,
                meta_rows=meta_rows,
                path_lookup=path_lookup,
                window=context_window,
            )
            text = str(context.get("text", "") or "").strip()
            if not text:
                continue
            manual_match = infer_manual_unit_for_text(
                text,
                topic_id="",
                manual_units=session_manual_units,
            )
            windows.append(
                {
                    "retrieval_rank": str(result.get("rank", "") or ""),
                    "session_id": infer_session_id(str(context.get("path", "") or result.get("file", ""))),
                    "speaker": str(context.get("speaker", "") or result.get("speaker", "") or ""),
                    "score_combined": str(result.get("score_combined", "") or ""),
                    "score_doc": str(result.get("score_doc", "") or ""),
                    "score_topic": str(result.get("score_topic", "") or ""),
                    "manual_unit_id_best_match": str(manual_match.get("manual_unit_id", "") or ""),
                    "manual_unit_match_score": str(manual_match.get("score", "") or ""),
                    "text": text,
                    "excerpt": text.replace("\n", " ").strip(),
                }
            )

        session_evidence = pd.DataFrame(windows)
        if session_evidence.empty:
            continue

        expected_ids = {str(unit.get("manual_unit_id", "")) for unit in session_manual_units if str(unit.get("manual_unit_id", "")).strip()}
        observed_ids = {
            x
            for x in session_evidence["manual_unit_id_best_match"].astype(str)
            if x and x in expected_ids
        }
        expected_subsections = {
            str(unit.get("manual_subsection", "")).strip()
            for unit in session_manual_units
            if str(unit.get("manual_subsection", "")).strip()
        }
        observed_subsections = {
            str(unit.get("manual_subsection", "")).strip()
            for unit in session_manual_units
            if str(unit.get("manual_unit_id", "")).strip() in observed_ids and str(unit.get("manual_subsection", "")).strip()
        }

        manual_cov = (len(observed_ids) / len(expected_ids)) if expected_ids else 0.0
        subsection_cov = (len(observed_subsections) / len(expected_subsections)) if expected_subsections else 0.0
        evidence_density = (len(session_evidence.index) / len(expected_ids)) if expected_ids else 0.0
        adherence_score = manual_weight * manual_cov + subsection_weight * subsection_cov
        if adherence_score >= high_cutoff:
            adherence = "high"
        elif adherence_score >= moderate_cutoff:
            adherence = "moderate"
        else:
            adherence = "low"

        summary_rows.append(
            {
                "cycle_id": cycle_dir.name,
                "manual_session_num": session_num,
                "manual_session_label": session_summary_row.get("session_label", f"Session {session_num}"),
                "fidelity_query": fidelity_query,
                "session_summary": session_summary,
                "session_topic_ids": ";".join(topic_ids),
                "session_topic_labels": ";".join(topic_labels),
                "retrieved_evidence_count": str(len(session_evidence.index)),
                "expected_manual_unit_count": str(len(expected_ids)),
                "matched_manual_unit_count": str(len(observed_ids)),
                "manual_unit_coverage": f"{manual_cov:.3f}",
                "expected_subsection_count": str(len(expected_subsections)),
                "matched_subsection_count": str(len(observed_subsections)),
                "subsection_coverage": f"{subsection_cov:.3f}",
                "evidence_density": f"{evidence_density:.3f}",
                "adherence_score": f"{adherence_score:.3f}",
                "adherence_label": adherence,
                "matched_manual_unit_ids": ";".join(sorted(observed_ids)),
                "matched_subsections": ";".join(sorted(observed_subsections)),
                "sample_session_ids": ";".join(sorted(x for x in session_evidence["session_id"].astype(str).unique() if x)),
            }
        )

        for _, row in session_evidence.iterrows():
            text = str(row.get("text", "") or "")
            excerpt = str(row.get("excerpt", "") or "")
            evidence_rows.append(
                {
                    "cycle_id": cycle_dir.name,
                    "manual_session_num": session_num,
                    "manual_session_label": f"Session {session_num}",
                    "analysis_mode": "session_fidelity",
                    "query_text": fidelity_query,
                    "source_topic_id": "",
                    "source_topic_label": "",
                    "retrieval_rank": str(row.get("retrieval_rank", "") or ""),
                    "session_id": str(row.get("session_id", "") or ""),
                    "speaker": str(row.get("speaker", "") or ""),
                    "score_combined": str(row.get("score_combined", "") or ""),
                    "score_doc": str(row.get("score_doc", "") or ""),
                    "score_topic": str(row.get("score_topic", "") or ""),
                    "manual_unit_id_best_match": str(row.get("manual_unit_id_best_match", "") or ""),
                    "manual_unit_match_score": str(row.get("manual_unit_match_score", "") or ""),
                    "text": text,
                    "excerpt": excerpt,
                }
            )

    return summary_rows, evidence_rows


def main() -> None:
    topic_catalog = load_csv(TOPIC_CATALOG_CSV)
    if topic_catalog.empty:
        raise SystemExit("topic_catalog.csv is required.")

    summary_fields = [
        "cycle_id",
        "manual_session_num",
        "manual_session_label",
        "fidelity_query",
        "session_summary",
        "session_topic_ids",
        "session_topic_labels",
        "retrieved_evidence_count",
        "expected_manual_unit_count",
        "matched_manual_unit_count",
        "manual_unit_coverage",
        "expected_subsection_count",
        "matched_subsection_count",
        "subsection_coverage",
        "evidence_density",
        "adherence_score",
        "adherence_label",
        "matched_manual_unit_ids",
        "matched_subsections",
        "sample_session_ids",
    ]
    evidence_fields = [
        "cycle_id",
        "manual_session_num",
        "manual_session_label",
        "analysis_mode",
        "query_text",
        "source_topic_id",
        "source_topic_label",
        "retrieval_rank",
        "session_id",
        "speaker",
        "score_combined",
        "score_doc",
        "score_topic",
        "manual_unit_id_best_match",
        "manual_unit_match_score",
        "text",
        "excerpt",
    ]

    for cycle_dir in list_cycle_dirs(CYCLE_ANALYSIS_DIR):
        summary_rows, evidence_rows = build_session_fidelity_for_cycle(cycle_dir, topic_catalog)
        write_csv(cycle_dir / "session_fidelity_summary.csv", summary_fields, summary_rows)
        write_csv(cycle_dir / "session_fidelity_evidence.csv", evidence_fields, evidence_rows)
        print(f"Wrote session-level fidelity outputs to {cycle_dir}")


if __name__ == "__main__":
    main()
