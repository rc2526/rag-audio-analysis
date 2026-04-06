#!/usr/bin/env python3
import csv
from pathlib import Path

from rag_audio_analysis.coding_rules import (
    BARRIER_KEYWORDS,
    CONFUSION_KEYWORDS,
    FAMILY_KEYWORDS,
    HELP_KEYWORDS,
    SUCCESS_KEYWORDS,
    has_any,
    infer_contains_skill_language,
    infer_demo_type,
    infer_practice_valence,
    infer_question_domain,
)
from rag_audio_analysis.config import CODED_EVIDENCE_CSV, TRANSCRIPT_SPANS_CSV
from rag_audio_analysis.source_bridge import build_manual_unit_index, infer_manual_unit_for_text

FIELDNAMES = [
    "coding_id",
    "span_id",
    "topic_id",
    "question_domain",
    "is_relevant",
    "speaker_role",
    "facilitator_refers_topic",
    "facilitator_demonstrates_skill",
    "facilitator_demo_type",
    "participant_practices_in_session",
    "participant_practices_home_individual",
    "participant_practices_home_child",
    "participant_practices_home_family",
    "participant_reports_barrier",
    "participant_reports_success",
    "participant_reports_confusion",
    "participant_requests_help",
    "practice_valence",
    "manual_alignment",
    "manual_alignment_type",
    "manual_unit_id_best_match",
    "evidence_strength",
    "coder_id",
    "coding_round",
    "adjudication_status",
    "comments",
]

PRESERVE_IF_EXISTS = {
    "manual_alignment",
    "manual_alignment_type",
    "manual_unit_id_best_match",
    "coder_id",
    "coding_round",
    "adjudication_status",
    "comments",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strength_from_confidence(confidence: str) -> str:
    return {"high": "strong", "medium": "moderate", "low": "weak"}.get(confidence, "weak")


def main() -> None:
    spans = read_csv(TRANSCRIPT_SPANS_CSV)
    existing_rows = {row["coding_id"]: row for row in read_csv(CODED_EVIDENCE_CSV) if row.get("coding_id")}
    manual_units = build_manual_unit_index()

    generated_rows: list[dict[str, str]] = []
    for span in spans:
        text = span.get("span_text", "")
        topic_id = span.get("topic_id_primary", "")
        role = span.get("speaker_role", "")
        question_domain = infer_question_domain(span)
        manual_match = infer_manual_unit_for_text(text, topic_id=topic_id, manual_units=manual_units)
        facilitator_demonstrates = "1" if role == "facilitator" and infer_contains_skill_language(text) and question_domain == "facilitator_demonstration" else "0"
        participant_home = "1" if role == "participant" and span.get("contains_home_language") == "1" and span.get("contains_child_language") != "1" else "0"
        participant_child = "1" if role == "participant" and span.get("contains_child_language") == "1" else "0"
        participant_family = "1" if role == "participant" and has_any(text, FAMILY_KEYWORDS) else "0"
        participant_in_session = "1" if role == "participant" and participant_home == "0" and participant_child == "0" and infer_contains_skill_language(text) else "0"
        relevant = "1" if topic_id or infer_contains_skill_language(text) or span.get("contains_home_language") == "1" or span.get("contains_child_language") == "1" else "0"
        generated = {
            "coding_id": f"COD_{span['span_id']}",
            "span_id": span["span_id"],
            "topic_id": topic_id,
            "question_domain": question_domain,
            "is_relevant": relevant,
            "speaker_role": role,
            "facilitator_refers_topic": "1" if role == "facilitator" and topic_id else "0",
            "facilitator_demonstrates_skill": facilitator_demonstrates,
            "facilitator_demo_type": infer_demo_type(text) if facilitator_demonstrates == "1" else "",
            "participant_practices_in_session": participant_in_session,
            "participant_practices_home_individual": participant_home,
            "participant_practices_home_child": participant_child,
            "participant_practices_home_family": participant_family,
            "participant_reports_barrier": "1" if role == "participant" and has_any(text, BARRIER_KEYWORDS) else "0",
            "participant_reports_success": "1" if role == "participant" and has_any(text, SUCCESS_KEYWORDS) else "0",
            "participant_reports_confusion": "1" if role == "participant" and has_any(text, CONFUSION_KEYWORDS) else "0",
            "participant_requests_help": "1" if role == "participant" and has_any(text, HELP_KEYWORDS) else "0",
            "practice_valence": infer_practice_valence(text) if role == "participant" else "not_applicable",
            "manual_alignment": "",
            "manual_alignment_type": "unclear" if topic_id else "",
            "manual_unit_id_best_match": manual_match.get("manual_unit_id", ""),
            "evidence_strength": strength_from_confidence(span.get("topic_confidence", "")),
            "coder_id": "",
            "coding_round": "prefill",
            "adjudication_status": "pending" if topic_id else "not_needed",
            "comments": "auto-prefilled",
        }

        existing = existing_rows.get(generated["coding_id"])
        if existing:
            existing_comments = (existing.get("comments") or "").strip()
            is_human_curated = any(
                existing.get(field)
                for field in ("coder_id", "comments")
            ) or existing.get("coding_round") not in ("", "prefill")
            if existing_comments in ("", "auto-prefilled"):
                is_human_curated = bool(existing.get("coder_id")) or existing.get("coding_round") not in ("", "prefill")

            if is_human_curated:
                for field in (
                    "topic_id",
                    "question_domain",
                    "is_relevant",
                    "facilitator_refers_topic",
                    "facilitator_demonstrates_skill",
                    "facilitator_demo_type",
                    "participant_practices_in_session",
                    "participant_practices_home_individual",
                    "participant_practices_home_child",
                    "participant_practices_home_family",
                    "participant_reports_barrier",
                    "participant_reports_success",
                    "participant_reports_confusion",
                    "participant_requests_help",
                    "practice_valence",
                    "evidence_strength",
                ):
                    if existing.get(field):
                        generated[field] = existing[field]

            for field in PRESERVE_IF_EXISTS:
                if existing.get(field):
                    generated[field] = existing[field]
        generated_rows.append(generated)

    with open(CODED_EVIDENCE_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(generated_rows)
    print(f"Wrote {CODED_EVIDENCE_CSV}")


if __name__ == "__main__":
    main()
