#!/usr/bin/env python3
"""Build Topic × Transcript top-k matches and persist parquet + diagnostics.

Usage examples:
  python scripts/build_topic_window_heatmaps.py --topic-id T1 --topk 20
  python scripts/build_topic_window_heatmaps.py --all --topk 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

from rag_audio_analysis.config import TOPIC_CATALOG_CSV
from rag_audio_analysis.window_manual_heatmap import (
    HeatmapConfig,
    select_window_centers,
    build_windows_from_centers,
    embed_texts_with_cache,
    topk_sparse_matches,
    save_matches,
)
from rag_audio_analysis.source_bridge import encode_texts


def build_for_topic(topic_row: dict, outdir: Path, topk: int = 20, context_window: int = 2) -> None:
    topic_id = str(topic_row.get("topic_id", "") or "").strip()
    topic_label = str(topic_row.get("topic_label", "") or "").strip()
    # Prefer explicit definition fields commonly used; fall back to any textual column
    topic_text = (
        str(topic_row.get("topic_definition", "") or "")
        or str(topic_row.get("definition", "") or "")
        or str(topic_row.get("topic_text", "") or "")
        or ""
    )

    if not topic_text:
        print(f"Skipping topic {topic_id} (no text definition found).")
        return

    cfg = HeatmapConfig(
        selection_mode="cycle_only",
        cycle_id="",
        session_num="",
        topic_id=topic_id,
        context_window=context_window,
        max_windows=0,  # disable cap (no limit)
        random_sample=False,
        sample_seed=0,
        max_window_chars=0,
        topk=topk,
        manual_scope="global",
    )

    print(f"Building Topic×Transcript for topic {topic_id} ({topic_label}) ...")
    start = time.time()

    # Gather centers (all cycles, no filtering)
    centers = select_window_centers(selection_mode=cfg.selection_mode, cycle_id=cfg.cycle_id, session_num=cfg.session_num, max_windows=cfg.max_windows, random_sample=cfg.random_sample, sample_seed=cfg.sample_seed)

    windows = build_windows_from_centers(centers=centers, context_window=cfg.context_window)
    window_texts = [str(w.get("window_text", "") or "") for w in windows]
    window_emb = embed_texts_with_cache(window_texts)

    # Create a single "manual unit" record for the topic and embed it
    manual_units = [
        {
            "manual_unit_id": f"TOPIC_{topic_id}",
            "text": topic_text,
            "matching_text": topic_text,
            "manual_week": "",
            "manual_section": "",
            "manual_subsection": "",
            "manual_category": "topic",
        }
    ]
    # Encode topic text (use project's encoder for consistency)
    try:
        manual_emb = embed_texts_with_cache([topic_text])
    except Exception:
        manual_emb = encode_texts([topic_text])

    matches = topk_sparse_matches(windows=windows, window_embeddings=window_emb, manual_units=manual_units, manual_embeddings=manual_emb, topk=cfg.topk)

    diagnostics = {
        "topic_id": topic_id,
        "topic_label": topic_label,
        "candidate_centers_count": len(centers),
        "candidate_windows_count": len(windows),
        "topk": int(topk),
        "elapsed_seconds": round(time.time() - start, 2),
    }

    if matches is None or matches.empty:
        print(f"No matches produced for topic {topic_id}. diagnostics={diagnostics}")
        return

    mode = f"topic_only_{topic_id}_full"
    outpath = save_matches(matches, diagnostics, mode, outdir=outdir)
    print(f"Saved {outpath} | rows={len(matches)} windows={matches['window_id'].nunique()} elapsed={diagnostics['elapsed_seconds']}s")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Topic × Transcript top-k matches")
    parser.add_argument("--topic-id", type=str, help="Topic ID to build (default: first row in catalog)")
    parser.add_argument("--all", action="store_true", help="Build for all topics in the catalog")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="data/derived/topic_window_heatmaps")
    parser.add_argument("--context-window", type=int, default=2)
    args = parser.parse_args(argv or sys.argv[1:])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load topic catalog
    try:
        topic_df = pd.read_csv(Path(TOPIC_CATALOG_CSV))
    except Exception as exc:
        print(f"Failed to load topic catalog at {TOPIC_CATALOG_CSV}: {exc}")
        return 2

    if args.all:
        rows = topic_df.to_dict(orient="records")
    else:
        if args.topic_id:
            rows = [r for r in topic_df.to_dict(orient="records") if str(r.get("topic_id", "")).strip() == args.topic_id]
            if not rows:
                print(f"Topic id {args.topic_id} not found in catalog; defaulting to first row")
        if not args.topic_id or not rows:
            rows = topic_df.head(1).to_dict(orient="records")

    for row in rows:
        build_for_topic(row, outdir=outdir, topk=args.topk, context_window=args.context_window)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
