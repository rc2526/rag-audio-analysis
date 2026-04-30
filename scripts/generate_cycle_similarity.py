#!/usr/bin/env python3
"""Generate similarity-first session evidence and coverage CSVs for a cycle.

This script runs the vectorized similarity helper for every manual session and
appends results into the cycle directory CSVs without overwriting.
"""
from pathlib import Path
import csv
import sys
from typing import Any

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR
from rag_audio_analysis.source_bridge import (
    load_topic_entries,
    get_manual_units_for_session,
    get_rag_index_rows,
    build_doc_index_by_path,
    query_evidence_by_manual_unit_similarity,
)
from rag_audio_analysis.settings import get_int

SIMILARITY_EVIDENCE_HEADER = [
    "cycle_id",
    "manual_session_num",
    "query_manual_unit_id",
    "query_manual_unit_subsection",
    "query_text_excerpt",
    "doc_index",
    "retrieval_rank",
    "global_rank",
    "session_id",
    "speaker",
    "mapped_manual_unit_match_score",
    "mapped_manual_unit_accepted",
    "text_excerpt",
    "text_length",
    "doc_path",
    "note",
]

SIMILARITY_COVERAGE_HEADER = [
    "manual_session_num",
    "query_manual_unit_id",
    "matched",
    "max_similarity",
    "num_matches",
    "avg_similarity",
    "best_doc_index",
    "best_doc_excerpt",
    "query_manual_unit_subsection",
]


def format_excerpt(text: str, limit: int | None = None) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").strip()
    if limit is None:
        limit = get_int("prompting", "evidence_excerpt_chars", 400)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main(cycle_id: str = "PMHCycle1") -> None:
    cycle_dir = Path(CYCLE_ANALYSIS_DIR) / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = get_rag_index_rows()
    path_lookup = build_doc_index_by_path(meta_rows)

    topics = load_topic_entries()
    sessions = sorted({str(t.get("session_num", "")).strip() for t in topics if str(t.get("session_num", "")).strip()})

    total_sim_rows = 0
    total_cov_rows = 0

    for s in sessions:
        units = get_manual_units_for_session(s)
        if not units:
            continue
        rows = query_evidence_by_manual_unit_similarity(
            session_manual_units=units,
            cycle_id=cycle_id,
            meta_rows=meta_rows,
            path_lookup=path_lookup,
            window=int(get_int("transcript_export", "context_window", 2)),
            topk_per_unit=0,
        )
        if not rows:
            continue

        sim_rows = []
        for w in rows:
            text = str(w.get("text", "") or "")
            sim_rows.append(
                {
                    "cycle_id": cycle_id,
                    "manual_session_num": s,
                    "query_manual_unit_id": w.get("query_manual_unit_id", ""),
                    "query_manual_unit_subsection": w.get("query_manual_unit_subsection", ""),
                    "query_text_excerpt": format_excerpt(w.get("query_text", ""), get_int("prompting", "manual_excerpt_chars", 220)),
                    "doc_index": w.get("doc_index", ""),
                    "retrieval_rank": w.get("retrieval_rank", ""),
                    "global_rank": "",
                    "session_id": w.get("session_id", ""),
                    "speaker": w.get("speaker", ""),
                    "mapped_manual_unit_match_score": w.get("mapped_manual_unit_match_score", w.get("manual_unit_match_score", "")),
                    "mapped_manual_unit_accepted": w.get("mapped_manual_unit_accepted", False),
                    "text_excerpt": format_excerpt(text, get_int("prompting", "evidence_excerpt_chars", 400)),
                    "text_length": len(text),
                    "doc_path": (meta_rows[int(w.get("doc_index"))].get("path") if w.get("doc_index") is not None and int(w.get("doc_index")) < len(meta_rows) else ""),
                    "note": "",
                }
            )
        # sort by similarity and set global rank per-session
        sim_rows.sort(key=lambda r: float(r.get("mapped_manual_unit_match_score", 0.0)), reverse=True)
        for i, r in enumerate(sim_rows, start=1):
            r["global_rank"] = i

        # write/append sim rows
        sim_path = cycle_dir / "session_manual_similarity_evidence.csv"
        append_csv_rows(sim_path, SIMILARITY_EVIDENCE_HEADER, sim_rows)
        total_sim_rows += len(sim_rows)

        # build coverage rows per session and append
        coverage_map: dict[str, dict[str, Any]] = {}
        for r in sim_rows:
            uid = str(r.get("query_manual_unit_id", ""))
            score = float(r.get("mapped_manual_unit_match_score", 0.0) or 0.0)
            doc_index = r.get("doc_index", "")
            if uid not in coverage_map:
                coverage_map[uid] = {
                    "manual_session_num": s,
                    "query_manual_unit_id": uid,
                    "matched": False,
                    "max_similarity": 0.0,
                    "num_matches": 0,
                    "sum_similarity": 0.0,
                    "best_doc_index": "",
                    "best_doc_excerpt": "",
                    "query_manual_unit_subsection": "",
                }
            entry = coverage_map[uid]
            entry["num_matches"] += 1
            entry["sum_similarity"] += score
            if score > float(entry.get("max_similarity", 0.0) or 0.0):
                entry["max_similarity"] = score
                entry["best_doc_index"] = doc_index
                entry["best_doc_excerpt"] = r.get("text_excerpt", "")
                entry["query_manual_unit_subsection"] = r.get("query_manual_unit_subsection", "")
            entry["matched"] = entry["matched"] or bool(r.get("mapped_manual_unit_accepted"))

        coverage_rows = []
        for uid, entry in coverage_map.items():
            num = int(entry.get("num_matches", 0) or 0)
            avg = (float(entry.get("sum_similarity", 0.0) or 0.0) / num) if num else 0.0
            coverage_rows.append(
                {
                    "manual_session_num": entry.get("manual_session_num", ""),
                    "query_manual_unit_id": entry.get("query_manual_unit_id", ""),
                    "matched": str(bool(entry.get("matched", False))),
                    "max_similarity": f"{float(entry.get('max_similarity', 0.0)):.6f}",
                    "num_matches": str(num),
                    "avg_similarity": f"{avg:.6f}",
                    "best_doc_index": entry.get("best_doc_index", ""),
                    "best_doc_excerpt": entry.get("best_doc_excerpt", ""),
                    "query_manual_unit_subsection": entry.get("query_manual_unit_subsection", ""),
                }
            )
        cov_path = cycle_dir / "manual_unit_coverage_summary.csv"
        append_csv_rows(cov_path, SIMILARITY_COVERAGE_HEADER, coverage_rows)
        total_cov_rows += len(coverage_rows)

    print(f"Wrote {total_sim_rows} similarity rows and {total_cov_rows} coverage rows to {cycle_dir}")


if __name__ == "__main__":
    cycle = sys.argv[1] if len(sys.argv) > 1 else "PMHCycle1"
    main(cycle)
