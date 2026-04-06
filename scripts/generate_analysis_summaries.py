#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

from rag_audio_analysis.config import (
    MANUAL_FIDELITY_SUMMARY_CSV,
    MANUAL_UNITS_CSV,
    TOPIC_CONTENT_SUMMARY_CSV,
    TOPIC_SESSION_SUMMARY_CSV,
    TRANSCRIPT_SPANS_CSV,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    spans = read_csv(TRANSCRIPT_SPANS_CSV)
    manual_units = read_csv(MANUAL_UNITS_CSV)

    topic_summary: dict[str, dict[str, object]] = {}
    topic_session_summary: dict[tuple[str, str], dict[str, object]] = {}
    manual_summary: dict[str, dict[str, object]] = {}

    for unit in manual_units:
        manual_unit_id = unit.get("manual_unit_id", "")
        manual_summary[manual_unit_id] = {
            "manual_unit_id": manual_unit_id,
            "topic_id": unit.get("topic_id", ""),
            "manual_week": unit.get("manual_week", ""),
            "manual_function": unit.get("manual_function", ""),
            "manual_text_short": unit.get("manual_text_short", ""),
            "matched_span_count": 0,
            "matched_session_count": set(),
            "home_span_count": 0,
            "child_span_count": 0,
        }

    for span in spans:
        topic_id = span.get("topic_id_primary", "")
        session_id = span.get("session_id", "")
        if topic_id:
            topic_entry = topic_summary.setdefault(
                topic_id,
                {
                    "topic_id": topic_id,
                    "span_count": 0,
                    "session_ids": set(),
                    "high_conf_count": 0,
                    "medium_conf_count": 0,
                    "low_conf_count": 0,
                    "home_span_count": 0,
                    "child_span_count": 0,
                    "manual_match_span_count": 0,
                },
            )
            topic_entry["span_count"] += 1
            topic_entry["session_ids"].add(session_id)
            conf = span.get("topic_confidence", "")
            if conf == "high":
                topic_entry["high_conf_count"] += 1
            elif conf == "medium":
                topic_entry["medium_conf_count"] += 1
            elif conf == "low":
                topic_entry["low_conf_count"] += 1
            if span.get("contains_home_language") == "1":
                topic_entry["home_span_count"] += 1
            if span.get("contains_child_language") == "1":
                topic_entry["child_span_count"] += 1
            if span.get("manual_unit_id_best_match"):
                topic_entry["manual_match_span_count"] += 1

            ts_key = (topic_id, session_id)
            ts_entry = topic_session_summary.setdefault(
                ts_key,
                {
                    "topic_id": topic_id,
                    "session_id": session_id,
                    "cycle_id": span.get("cycle_id", ""),
                    "span_count": 0,
                    "home_span_count": 0,
                    "child_span_count": 0,
                    "manual_match_span_count": 0,
                },
            )
            ts_entry["span_count"] += 1
            if span.get("contains_home_language") == "1":
                ts_entry["home_span_count"] += 1
            if span.get("contains_child_language") == "1":
                ts_entry["child_span_count"] += 1
            if span.get("manual_unit_id_best_match"):
                ts_entry["manual_match_span_count"] += 1

        manual_unit_id = span.get("manual_unit_id_best_match", "")
        if manual_unit_id and manual_unit_id in manual_summary:
            ms = manual_summary[manual_unit_id]
            ms["matched_span_count"] += 1
            ms["matched_session_count"].add(session_id)
            if span.get("contains_home_language") == "1":
                ms["home_span_count"] += 1
            if span.get("contains_child_language") == "1":
                ms["child_span_count"] += 1

    topic_rows = []
    for entry in topic_summary.values():
        topic_rows.append(
            {
                "topic_id": entry["topic_id"],
                "span_count": str(entry["span_count"]),
                "session_count": str(len(entry["session_ids"])),
                "high_conf_count": str(entry["high_conf_count"]),
                "medium_conf_count": str(entry["medium_conf_count"]),
                "low_conf_count": str(entry["low_conf_count"]),
                "home_span_count": str(entry["home_span_count"]),
                "child_span_count": str(entry["child_span_count"]),
                "manual_match_span_count": str(entry["manual_match_span_count"]),
            }
        )
    topic_rows.sort(key=lambda r: int(r["span_count"]), reverse=True)
    write_csv(
        TOPIC_CONTENT_SUMMARY_CSV,
        [
            "topic_id",
            "span_count",
            "session_count",
            "high_conf_count",
            "medium_conf_count",
            "low_conf_count",
            "home_span_count",
            "child_span_count",
            "manual_match_span_count",
        ],
        topic_rows,
    )

    session_rows = []
    for entry in topic_session_summary.values():
        session_rows.append({k: str(v) for k, v in entry.items()})
    session_rows.sort(key=lambda r: (r["topic_id"], -int(r["span_count"])))
    write_csv(
        TOPIC_SESSION_SUMMARY_CSV,
        [
            "topic_id",
            "session_id",
            "cycle_id",
            "span_count",
            "home_span_count",
            "child_span_count",
            "manual_match_span_count",
        ],
        session_rows,
    )

    manual_rows = []
    for entry in manual_summary.values():
        matched_span_count = int(entry["matched_span_count"])
        if matched_span_count == 0:
            fidelity_status = "not_observed"
        elif matched_span_count < 3:
            fidelity_status = "underrepresented"
        else:
            fidelity_status = "observed"
        manual_rows.append(
            {
                "manual_unit_id": entry["manual_unit_id"],
                "topic_id": entry["topic_id"],
                "manual_week": entry["manual_week"],
                "manual_function": entry["manual_function"],
                "manual_text_short": entry["manual_text_short"],
                "matched_span_count": str(matched_span_count),
                "matched_session_count": str(len(entry["matched_session_count"])),
                "home_span_count": str(entry["home_span_count"]),
                "child_span_count": str(entry["child_span_count"]),
                "fidelity_status": fidelity_status,
            }
        )
    manual_rows.sort(key=lambda r: (r["fidelity_status"], -int(r["matched_span_count"])))
    write_csv(
        MANUAL_FIDELITY_SUMMARY_CSV,
        [
            "manual_unit_id",
            "topic_id",
            "manual_week",
            "manual_function",
            "manual_text_short",
            "matched_span_count",
            "matched_session_count",
            "home_span_count",
            "child_span_count",
            "fidelity_status",
        ],
        manual_rows,
    )

    print(f"Wrote {TOPIC_CONTENT_SUMMARY_CSV}")
    print(f"Wrote {TOPIC_SESSION_SUMMARY_CSV}")
    print(f"Wrote {MANUAL_FIDELITY_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
