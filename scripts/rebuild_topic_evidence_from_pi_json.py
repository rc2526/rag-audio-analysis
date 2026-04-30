#!/usr/bin/env python3
"""
scripts/rebuild_topic_evidence_from_pi_json.py

Rebuild per-cycle topic_evidence.csv rows from pi_question_answers.json entries.
Supports dry-run and safe backups.
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

OUT_FIELDS = [
    "cycle_id",
    "session_num",
    "topic_id",
    "topic_label",
    "analysis_mode",
    "question_id",
    "query_text",
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
    "topk_mode",
    "topk_value",
]

def excerpt(text: str, limit: int = 500) -> str:
    if not text:
        return ""
    t = text.replace("\n", " ").strip()
    return t if len(t) <= limit else t[:limit].rstrip() + "..."

def load_existing(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in OUT_FIELDS})

def dedupe_keep_last(rows: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
    seen = {}
    order = []
    for row in rows:
        key = tuple(row.get(k, "") for k in key_fields)
        seen[key] = row
        if key not in order:
            order.append(key)
    return [seen[k] for k in order]

def make_rows_from_json_entry(ent: Dict[str, Any]) -> List[Dict[str, Any]]:
    cycle_id = ent.get("cycle_id", "")
    session_num = str(ent.get("session_num", ""))
    topic_id = ent.get("topic_id", "")
    topic_label = ent.get("topic_label", "")
    question_id = ent.get("question_id", "")
    query_text = ent.get("query_text", "")
    # Evidence is expected to be a list of window dicts
    out: List[Dict[str, Any]] = []
    for i, w in enumerate(ent.get("evidence", []) or [], start=1):
        retrieval_rank = w.get("retrieval_rank") or w.get("rank") or str(i)
        r = {
            "cycle_id": cycle_id,
            "session_num": session_num,
            "topic_id": topic_id,
            "topic_label": topic_label,
            "analysis_mode": "pi_question",
            "question_id": question_id,
            "query_text": query_text,
            "retrieval_rank": str(retrieval_rank),
            "session_id": w.get("session_id", w.get("session_id_in_meta", "")),
            "speaker": w.get("speaker", ""),
            "score_combined": w.get("score_combined", "") or w.get("score", "") or "",
            "score_doc": w.get("score_doc", ""),
            "score_topic": w.get("score_topic", ""),
            "manual_unit_id_best_match": w.get("manual_unit_id_best_match") or w.get("manual_unit_id") or "",
            "manual_unit_match_score": w.get("manual_unit_match_score") or w.get("manual_unit_score") or "",
            "text": w.get("text", "") or "",
            "excerpt": excerpt(w.get("text", "")),
            "topk_mode": w.get("topk_mode", "") or "",
            "topk_value": w.get("topk_value", "") or "",
        }
        out.append(r)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="+", default=None, help="Cycle ids (e.g. PMHCycle1) or numbers (1 2 3) to process. Default: all PMHCycle* under data/derived/cycle_analysis")
    parser.add_argument("--apply", action="store_true", help="Write files (default is dry-run)")
    parser.add_argument("--overwrite", action="store_true", help="When writing, overwrite topic_evidence.csv instead of merging")
    args = parser.parse_args()

    base = Path("data/derived/cycle_analysis")
    if args.cycles:
        cycle_dirs = []
        for c in args.cycles:
            if c.startswith("PMHCycle"):
                cycle_dirs.append(base / c)
            else:
                cycle_dirs.append(base / f"PMHCycle{c}")
    else:
        cycle_dirs = sorted([p for p in base.glob("PMHCycle*") if p.is_dir()])

    for cycle_dir in cycle_dirs:
        json_path = cycle_dir / "pi_question_answers.json"
        if not json_path.exists():
            print(f"skip {cycle_dir.name}: no {json_path.name}")
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        new_rows = []
        for ent in data:
            new_rows.extend(make_rows_from_json_entry(ent))
        print(f"{cycle_dir.name}: loaded {len(data)} pi entries -> {len(new_rows)} evidence windows")

        out_path = cycle_dir / "topic_evidence.csv"
        existing = load_existing(out_path) if out_path.exists() else []
        if args.apply:
            # backup
            if out_path.exists():
                bak = cycle_dir / f"topic_evidence.csv.bak"
                bak.write_bytes(out_path.read_bytes())
                print(f"backed up {out_path.name} -> {bak.name}")
            if args.overwrite:
                final_rows = new_rows
            else:
                # merge: remove any existing rows that share (cycle_id, session_num, topic_id, retrieval_rank) with new rows
                # safer: only drop existing rows that are pi_question rows with matching keys
                remove_keys = {(r["cycle_id"], r["session_num"], r["topic_id"], r["retrieval_rank"]) for r in new_rows}
                kept = []
                for r in existing:
                    key = (r.get("cycle_id",""), r.get("session_num",""), r.get("topic_id",""), r.get("retrieval_rank",""))
                    # preserve any fidelity rows even if key collides
                    if key in remove_keys and r.get("analysis_mode","") != "pi_question":
                        kept.append(r)
                    elif key not in remove_keys:
                        kept.append(r)
                final_rows = kept + new_rows
            # dedupe by (cycle_id, session_num, topic_id, question_id, retrieval_rank) keeping last-seen
            # include question_id so windows from different PI questions (same topic/session/rank)
            # are not collapsed together
            final_rows = dedupe_keep_last(final_rows, ["cycle_id", "session_num", "topic_id", "question_id", "retrieval_rank"])
            write_csv(out_path, final_rows)
            print(f"WROTE {out_path} ({len(final_rows)} rows)")
        else:
            print(f"DRY-RUN: would write {len(new_rows)} rows to {cycle_dir.name}/topic_evidence.csv (existing {len(existing)} rows).")
            if new_rows:
                sample = new_rows[:2]
                for s in sample:
                    print(" SAMPLE:", {k: s.get(k, "") for k in ('cycle_id','session_num','topic_id','question_id','retrieval_rank','session_id','excerpt')})
            print("Use --apply to write files (creates .bak backups).")

if __name__ == "__main__":
    main()