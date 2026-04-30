#!/usr/bin/env python3
"""Precompute manual_unit_counts.json for one or all cycles.

Writes a `manual_unit_counts.json` file into each cycle directory under
`CYCLE_ANALYSIS_DIR`. By default processes all cycle folders; pass specific
cycle names with `--cycles` to limit work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR
from rag_audio_analysis.source_bridge import build_manual_unit_index


def compute_counts_for_cycle(cycle_dir: Path) -> dict[str, int]:
    # Build the manual unit index once and tally by manual_week/session
    all_units = build_manual_unit_index()
    counts: dict[str, int] = {}
    for u in all_units:
        sess = str(u.get("manual_week", "") or "").strip()
        if not sess:
            continue
        counts[sess] = counts.get(sess, 0) + 1
    return counts


def main(cycles: Iterable[str] | None = None) -> None:
    base = Path(CYCLE_ANALYSIS_DIR)
    if not base.exists():
        print(f"CYCLE_ANALYSIS_DIR not found: {base}")
        return

    if cycles:
        targets = [base / c for c in cycles]
    else:
        targets = sorted([p for p in base.iterdir() if p.is_dir()])

    for target in targets:
        cycle_name = target.name if isinstance(target, Path) else str(target)
        cycle_dir = base / cycle_name
        if not cycle_dir.exists():
            print(f"Skipping missing cycle dir: {cycle_dir}")
            continue
        try:
            counts = compute_counts_for_cycle(cycle_dir)
            out_path = cycle_dir / "manual_unit_counts.json"
            out_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
            print(f"Wrote {out_path} ({len(counts)} sessions)")
        except Exception as e:
            print(f"Failed to compute counts for {cycle_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="*", help="Cycle names to process (default: all)")
    args = parser.parse_args()
    main(args.cycles if args.cycles else None)
