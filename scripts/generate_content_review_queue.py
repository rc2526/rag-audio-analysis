#!/usr/bin/env python3
import csv
from pathlib import Path

from rag_audio_analysis.config import CONTENT_REVIEW_QUEUE_CSV, TRANSCRIPT_SPANS_CSV

FIELDNAMES = [
    "review_id",
    "span_id",
    "session_id",
    "cycle_id",
    "topic_id_candidate",
    "topic_confidence",
    "manual_unit_id_best_match",
    "manual_topic_id_best_match",
    "contains_home_language",
    "contains_child_language",
    "review_priority",
    "priority_reason",
    "content_review_status",
    "manual_alignment_status",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    spans = read_csv(TRANSCRIPT_SPANS_CSV)
    existing = {row["review_id"]: row for row in read_csv(CONTENT_REVIEW_QUEUE_CSV) if row.get("review_id")}
    queue_rows: list[dict[str, str]] = []
    for span in spans:
        include = any(
            [
                span.get("topic_id_primary"),
                span.get("manual_unit_id_best_match"),
                span.get("contains_home_language") == "1",
                span.get("contains_child_language") == "1",
                span.get("review_priority") in ("high", "medium"),
            ]
        )
        if not include:
            continue

        review_id = f"REV_{span['span_id']}"
        generated = {
            "review_id": review_id,
            "span_id": span["span_id"],
            "session_id": span.get("session_id", ""),
            "cycle_id": span.get("cycle_id", ""),
            "topic_id_candidate": span.get("topic_id_primary", ""),
            "topic_confidence": span.get("topic_confidence", ""),
            "manual_unit_id_best_match": span.get("manual_unit_id_best_match", ""),
            "manual_topic_id_best_match": span.get("manual_topic_id_best_match", ""),
            "contains_home_language": span.get("contains_home_language", "0"),
            "contains_child_language": span.get("contains_child_language", "0"),
            "review_priority": span.get("review_priority", ""),
            "priority_reason": span.get("priority_reason", ""),
            "content_review_status": "unreviewed",
            "manual_alignment_status": "candidate" if span.get("manual_unit_id_best_match") else "",
            "notes": "",
        }
        prev = existing.get(review_id)
        if prev:
            for field in ("topic_id_candidate", "manual_unit_id_best_match", "manual_topic_id_best_match", "content_review_status", "manual_alignment_status", "notes"):
                if prev.get(field):
                    generated[field] = prev[field]
        queue_rows.append(generated)

    with open(CONTENT_REVIEW_QUEUE_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(queue_rows)
    print(f"Wrote {CONTENT_REVIEW_QUEUE_CSV}")


if __name__ == "__main__":
    main()
