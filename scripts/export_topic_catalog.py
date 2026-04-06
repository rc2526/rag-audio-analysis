#!/usr/bin/env python3
import csv

from rag_audio_analysis.config import TOPIC_CATALOG_CSV
from rag_audio_analysis.source_bridge import (
    get_session_summary,
    get_topic_definition,
    infer_week_num,
    load_topic_entries,
)

FIELDNAMES = [
    "topic_id",
    "topic_label",
    "session_num",
    "session_label",
    "manual_week",
    "manual_session_title",
    "topic_definition",
    "primary_skill",
    "secondary_skill",
    "manual_priority",
    "notes",
]


def main() -> None:
    topics = load_topic_entries()
    TOPIC_CATALOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TOPIC_CATALOG_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for topic in topics:
            label = topic["label"]
            session_num = str(topic.get("session_num", "") or "")
            session_summary = get_session_summary(session_num).get("session_summary", "")
            writer.writerow(
                {
                    "topic_id": topic["id"],
                    "topic_label": label,
                    "session_num": session_num,
                    "session_label": topic.get("session_label", ""),
                    "manual_week": session_num or infer_week_num(label),
                    "manual_session_title": topic.get("session_label", ""),
                    "topic_definition": get_topic_definition(topic["id"], label, session_summary),
                    "primary_skill": "",
                    "secondary_skill": "",
                    "manual_priority": "core",
                    "notes": "",
                }
            )
    print(f"Wrote {TOPIC_CATALOG_CSV}")


if __name__ == "__main__":
    main()
