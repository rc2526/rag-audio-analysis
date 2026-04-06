#!/usr/bin/env python3
import csv

from rag_audio_analysis.config import DATA_DIR
from rag_audio_analysis.source_bridge import get_session_summary, get_topic_definition, load_topic_entries


OUT_PATH = DATA_DIR / "derived" / "topic_definition_preview_v3.csv"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for topic in load_topic_entries():
        session_num = str(topic.get("session_num", "")).strip()
        session_label = str(topic.get("session_label", "")).strip()
        session_summary = get_session_summary(session_num).get("session_summary", "")
        rows.append(
            {
                "session_num": session_num,
                "session_label": session_label,
                "topic_id": topic.get("id", ""),
                "topic_label": topic.get("label", ""),
                "topic_definition_preview": get_topic_definition(
                    topic.get("id", ""),
                    topic.get("label", ""),
                    session_summary,
                ),
            }
        )

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "session_num",
                "session_label",
                "topic_id",
                "topic_label",
                "topic_definition_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(OUT_PATH)
    print(f"rows {len(rows)}")
    for row in rows[:20]:
        print(f"{row['session_num']} | {row['topic_label']} | {row['topic_definition_preview']}")


if __name__ == "__main__":
    main()
