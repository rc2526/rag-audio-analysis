#!/usr/bin/env python3
import csv

from rag_audio_analysis.config import MANUAL_UNITS_CSV
from rag_audio_analysis.source_bridge import (
    bool_to_str,
    get_structured_manual_units,
)

FIELDNAMES = [
    "manual_unit_id",
    "manual_chunk_index",
    "source_file",
    "topic_id",
    "topic_match_score",
    "manual_week",
    "manual_section",
    "manual_subsection",
    "manual_text",
    "manual_text_short",
    "manual_home_practice",
    "manual_child_context",
    "manual_facilitator_instruction",
    "manual_participant_activity",
    "manual_quote_candidate",
]


def main() -> None:
    manual_chunks = get_structured_manual_units()

    with open(MANUAL_UNITS_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, chunk in enumerate(manual_chunks, start=1):
            text = chunk.get("text", "")
            lowered = text.lower()
            writer.writerow(
                {
                    "manual_unit_id": f"MAN_{i:04d}",
                    "manual_chunk_index": i - 1,
                    "source_file": chunk.get("source", "manual.txt"),
                    "topic_id": chunk.get("topic_id", ""),
                    "topic_match_score": chunk.get("topic_match_score", ""),
                    "manual_week": chunk.get("manual_week", ""),
                    "manual_section": chunk.get("manual_section", ""),
                    "manual_subsection": chunk.get("manual_subsection", ""),
                    "manual_text": text,
                    "manual_text_short": text[:180].replace("\n", " "),
                    "manual_home_practice": bool_to_str("homework" in lowered or "at home" in lowered),
                    "manual_child_context": bool_to_str("child" in lowered),
                    "manual_facilitator_instruction": bool_to_str(
                        any(token in lowered for token in ("ask participants", "explain", "introduce", "pass around", "discuss"))
                    ),
                    "manual_participant_activity": bool_to_str(
                        any(token in lowered for token in ("practice", "write", "log", "journal", "activity"))
                    ),
                    "manual_quote_candidate": bool_to_str(len(text) > 80),
                }
            )
    print(f"Wrote {MANUAL_UNITS_CSV}")


if __name__ == "__main__":
    main()
