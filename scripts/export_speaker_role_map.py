#!/usr/bin/env python3
import csv
from collections import Counter, defaultdict

from rag_audio_analysis.config import SPEAKER_ROLE_MAP_CSV, TRANSCRIPT_SPANS_CSV
from rag_audio_analysis.source_bridge import get_transcript_turns, infer_cycle_id, infer_session_id, infer_speaker_role

FIELDNAMES = [
    "session_id",
    "cycle_id",
    "speaker_label",
    "assigned_role",
    "assignment_confidence",
    "turn_count",
    "sample_text",
    "notes",
]


def read_existing() -> dict[tuple[str, str], dict[str, str]]:
    if not SPEAKER_ROLE_MAP_CSV.exists():
        return {}
    with open(SPEAKER_ROLE_MAP_CSV, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {(row["session_id"], row["speaker_label"]): row for row in rows}


def main() -> None:
    turns = get_transcript_turns()
    existing = read_existing()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for turn in turns:
        session_id = infer_session_id(turn.get("path", ""))
        speaker_label = turn.get("speaker", "")
        grouped[(session_id, speaker_label)].append(turn)

    with open(SPEAKER_ROLE_MAP_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for (session_id, speaker_label), items in sorted(grouped.items()):
            existing_row = existing.get((session_id, speaker_label), {})
            default_role = infer_speaker_role(speaker_label)
            writer.writerow(
                {
                    "session_id": session_id,
                    "cycle_id": infer_cycle_id(items[0].get("path", "")),
                    "speaker_label": speaker_label,
                    "assigned_role": existing_row.get("assigned_role", default_role),
                    "assignment_confidence": existing_row.get("assignment_confidence", "low" if default_role == "unknown" else "medium"),
                    "turn_count": len(items),
                    "sample_text": (items[0].get("text", "") or "")[:180],
                    "notes": existing_row.get("notes", ""),
                }
            )
    print(f"Wrote {SPEAKER_ROLE_MAP_CSV}")


if __name__ == "__main__":
    main()
