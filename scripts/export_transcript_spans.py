#!/usr/bin/env python3
import csv

from rag_audio_analysis.config import TRANSCRIPT_SPANS_CSV
from rag_audio_analysis.coding_rules import infer_contains_skill_language, infer_review_priority
from rag_audio_analysis.settings import get_float, get_int
from rag_audio_analysis.source_bridge import (
    build_doc_index_by_path,
    expand_transcript_context,
    get_rag_index_rows,
    infer_cycle_id,
    infer_manual_unit_for_text,
    infer_session_id,
    infer_speaker_role,
    is_manual_row,
    load_speaker_role_map,
    load_topic_entries,
    query_topic_evidence,
)

FIELDNAMES = [
    "span_id",
    "session_id",
    "cycle_id",
    "week_num",
    "transcript_file",
    "source_type",
    "speaker_role",
    "speaker_label",
    "start_time_s",
    "end_time_s",
    "turn_count",
    "span_text",
    "span_text_blinded",
    "topic_id_primary",
    "topic_id_secondary",
    "topic_confidence",
    "manual_week_expected",
    "manual_unit_id_best_match",
    "manual_unit_match_score",
    "manual_topic_id_best_match",
    "contains_topic_reference",
    "contains_skill_language",
    "contains_home_language",
    "contains_child_language",
    "review_priority",
    "priority_reason",
    "rag_run_id",
    "selected_by_rag",
    "human_review_status",
    "notes",
    "source_doc_index",
    "retrieval_rank",
    "score_combined",
    "score_doc",
    "score_topic",
]


def confidence_from_rank(rank: int, score_combined: float) -> str:
    high_rank = get_int("transcript_export", "confidence_rank_high", 10)
    medium_rank = get_int("transcript_export", "confidence_rank_medium", 30)
    high_score = get_float("transcript_export", "confidence_score_high", 1.5)
    medium_score = get_float("transcript_export", "confidence_score_medium", 0.5)
    if rank <= high_rank or score_combined >= high_score:
        return "high"
    if rank <= medium_rank or score_combined >= medium_score:
        return "medium"
    return "low"


def main() -> None:
    topics = load_topic_entries()
    meta_rows = get_rag_index_rows()
    path_lookup = build_doc_index_by_path(meta_rows)
    speaker_role_map = load_speaker_role_map()
    seen: set[tuple[str, int]] = set()
    query_topk = get_int("transcript_export", "topic_query_topk", 120)
    query_weight_doc = get_float("transcript_export", "topic_query_weight_doc", 0.3)
    query_weight_topic = get_float("transcript_export", "topic_query_weight_topic", 0.7)
    context_window = get_int("transcript_export", "context_window", 2)

    with open(TRANSCRIPT_SPANS_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        span_num = 0

        for topic in topics:
            topic_id = str(topic.get("id", "") or "")
            topic_label = str(topic.get("label", "") or "")
            if not topic_id or not topic_label:
                continue

            results = query_topic_evidence(
                topic_label,
                topk=query_topk,
                weight_doc=query_weight_doc,
                weight_topic=query_weight_topic,
            )
            transcript_results = [row for row in results if not is_manual_row(meta_rows[int(row.get("doc_index", -1))])]

            for rank, result in enumerate(transcript_results, start=1):
                doc_index = int(result.get("doc_index", -1))
                if doc_index < 0 or doc_index >= len(meta_rows):
                    continue
                dedupe_key = (topic_id, doc_index)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                center_row = meta_rows[doc_index]
                context = expand_transcript_context(
                    doc_index,
                    meta_rows=meta_rows,
                    path_lookup=path_lookup,
                    window=context_window,
                )
                text = str(context.get("text", "") or "").strip()
                if not text:
                    continue

                transcript_path = str(context.get("path", "") or center_row.get("path", "") or "")
                session_id = infer_session_id(transcript_path)
                speaker_label = str(center_row.get("speaker", "") or "")
                mapped_role = speaker_role_map.get((session_id, speaker_label), infer_speaker_role(speaker_label))
                lowered = text.lower()
                manual_match = infer_manual_unit_for_text(text, topic_id=topic_id)
                score_combined = float(result.get("score_combined") or 0.0)
                topic_confidence = confidence_from_rank(rank, score_combined)

                span_num += 1
                row = {
                    "span_id": f"SPAN_{span_num:06d}",
                    "session_id": session_id,
                    "cycle_id": infer_cycle_id(transcript_path),
                    "week_num": "",
                    "transcript_file": transcript_path,
                    "source_type": "retrieved_transcript_window",
                    "speaker_role": mapped_role,
                    "speaker_label": speaker_label,
                    "start_time_s": "",
                    "end_time_s": "",
                    "turn_count": context.get("turn_count", 1),
                    "span_text": text,
                    "span_text_blinded": text,
                    "topic_id_primary": topic_id,
                    "topic_id_secondary": "",
                    "topic_confidence": topic_confidence,
                    "manual_week_expected": manual_match["manual_week"],
                    "manual_unit_id_best_match": manual_match["manual_unit_id"],
                    "manual_unit_match_score": manual_match.get("score", ""),
                    "manual_topic_id_best_match": manual_match["topic_id"],
                    "contains_topic_reference": "1",
                    "contains_skill_language": "1" if infer_contains_skill_language(text) else "0",
                    "contains_home_language": "1" if "home" in lowered else "0",
                    "contains_child_language": "1" if "child" in lowered else "0",
                    "review_priority": "",
                    "priority_reason": "",
                    "rag_run_id": f"topic_query::{topic_id}",
                    "selected_by_rag": "1",
                    "human_review_status": "unreviewed",
                    "notes": "",
                    "source_doc_index": str(doc_index),
                    "retrieval_rank": str(rank),
                    "score_combined": f"{score_combined:.4f}",
                    "score_doc": f"{float(result.get('score_doc') or 0.0):.4f}",
                    "score_topic": f"{float(result.get('score_topic') or 0.0):.4f}",
                }
                priority, reason = infer_review_priority(row)
                row["review_priority"] = priority
                row["priority_reason"] = reason
                writer.writerow(row)
    print(f"Wrote {TRANSCRIPT_SPANS_CSV}")


if __name__ == "__main__":
    main()
