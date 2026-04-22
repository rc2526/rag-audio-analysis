#!/usr/bin/env python3
"""Build prebuilt top-k window x manual heatmaps for the configured modes.

Usage:
    python scripts/build_prebuilt_heatmaps.py [--force]

This will create files under data/derived/topk_window_manual_heatmaps/{mode}.parquet or .csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from rag_audio_analysis.window_manual_heatmap import (
    HeatmapConfig,
    build_topk_window_manual_heatmap,
    save_matches,
)
from rag_audio_analysis.source_bridge import get_rag_index_rows, infer_cycle_id, infer_session_id
from rag_audio_analysis.settings import get_int, get_str


MODES: List[str] = [
    "cycle_only",
]


def _default_config_for_mode(mode: str) -> HeatmapConfig:
    # Use settings defaults where appropriate.
    context_window = int(get_int("topk_window_manual_heatmap", "context_window", get_int("transcript_export", "context_window", 2)))
    max_windows = int(get_int("topk_window_manual_heatmap", "max_windows", 200))
    topk = int(get_int("topk_window_manual_heatmap", "topk", 20))
    max_window_chars = int(get_int("topk_window_manual_heatmap", "max_window_chars", 0))

    cfg = HeatmapConfig(
        selection_mode=mode,  # type: ignore[arg-type]
        cycle_id=str(get_str("topk_window_manual_heatmap", "default_cycle_id", "").strip() or ""),
        session_num=str(get_str("topk_window_manual_heatmap", "default_session_num", "").strip() or ""),
        topic_id=str(get_str("topk_window_manual_heatmap", "default_topic_id", "").strip() or ""),
        context_window=context_window,
        max_windows=max_windows,
        random_sample=(get_str("topk_window_manual_heatmap", "random_sample", "false").strip().lower() == "true"),
        sample_seed=int(get_int("topk_window_manual_heatmap", "sample_seed", 0)),
        max_window_chars=max_window_chars,
        topk=topk,
        manual_scope=str(get_str("topk_window_manual_heatmap", "manual_scope", "session") or "session"),
        manual_unit_prefix_template=str(get_str("topk_window_manual_heatmap", "manual_unit_prefix_template", "Session {session_num} ") or "Session {session_num} "),
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/derived/topk_window_manual_heatmaps")
    parser.add_argument("--force", action="store_true", help="Rebuild even if files exist")
    parser.add_argument("--cycles", default="", help="Comma-separated cycle ids to build (overrides discovery)")
    parser.add_argument("--sessions", default="", help="Comma-separated session ids to build (overrides discovery)")
    parser.add_argument("--dryrun", action="store_true", help="Print planned outputs without building")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Only cycle_only is supported in this simplified build script
    modes_to_build = ["cycle_only"]

    # Determine cycles to target for cycle-scoped modes.
    cycles: list[str] = []
    if args.cycles:
        cycles = [c.strip() for c in args.cycles.split(",") if c.strip()]
        print(f"Using cycles from CLI: {cycles}")
    else:
        # Infer from meta rows
        print("Discovering cycles from RAG index...")
        rows = get_rag_index_rows()
        seen = set()
        for r in rows:
            path = str(r.get("path", "") or "")
            c = infer_cycle_id(path)
            if c:
                seen.add(c)
        cycles = sorted(seen, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0) if any(ch.isdigit() for ch in x) else x)
        print(f"Discovered cycles: {cycles}")

    # Determine sessions to target (optional). If provided, we'll produce per-session outputs.
    sessions: list[str] = []
    if args.sessions:
        sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
        print(f"Using sessions from CLI: {sessions}")
    else:
        # do not auto-discover sessions unless explicitly requested; leave sessions empty by default
        sessions = []

    summary_created = []
    for mode in modes_to_build:
        # For cycle-scoped mode, produce per-cycle files; if sessions provided, produce per-cycle x per-session files
        if cycles:
            for cycle in cycles:
                safe_cycle = cycle.replace(' ', '_')
                # If sessions specified, produce per-session files for this cycle
                if sessions:
                    for session in sessions:
                        safe_session = str(session).replace(' ', '_')
                        name = f"{mode}_{safe_cycle}_session{safe_session}"
                        p = outdir / f"{name}.parquet"
                        p_csv = outdir / f"{name}.csv"
                        if (p.exists() or p_csv.exists()) and not args.force:
                            print(f"Skipping {name}: already exists (use --force to rebuild)")
                            continue
                        if args.dryrun:
                            print(f"[dryrun] Would build {name}")
                            continue

                        cfg = _default_config_for_mode(mode)
                        cfg = HeatmapConfig(
                            selection_mode=cfg.selection_mode,
                            cycle_id=cycle,
                            session_num=str(session),
                            topic_id=cfg.topic_id,
                            context_window=cfg.context_window,
                            max_windows=cfg.max_windows,
                            random_sample=cfg.random_sample,
                            sample_seed=cfg.sample_seed,
                            max_window_chars=cfg.max_window_chars,
                            topk=cfg.topk,
                            manual_scope=cfg.manual_scope,
                            manual_unit_prefix_template=cfg.manual_unit_prefix_template,
                        )
                        print(f"Building heatmap for mode={mode} cycle={cycle} session={session} ...")
                        matches, diag = build_topk_window_manual_heatmap(cfg)
                        if matches is None or matches.empty:
                            print(f"No matches produced for {name}; skipping save")
                            continue
                        path = save_matches(matches, diag, name, outdir=outdir)
                        print(f"Saved {name} to {path}")
                        summary_created.append(str(path))
                else:
                    name = f"{mode}_{safe_cycle}"
                    p = outdir / f"{name}.parquet"
                    p_csv = outdir / f"{name}.csv"
                    if (p.exists() or p_csv.exists()) and not args.force:
                        print(f"Skipping {name}: already exists (use --force to rebuild)")
                        continue
                    if args.dryrun:
                        print(f"[dryrun] Would build {name}")
                        continue

                    cfg = _default_config_for_mode(mode)
                    cfg = HeatmapConfig(
                        selection_mode=cfg.selection_mode,
                        cycle_id=cycle,
                        session_num=cfg.session_num,
                        topic_id=cfg.topic_id,
                        context_window=cfg.context_window,
                        max_windows=cfg.max_windows,
                        random_sample=cfg.random_sample,
                        sample_seed=cfg.sample_seed,
                        max_window_chars=cfg.max_window_chars,
                        topk=cfg.topk,
                        manual_scope=cfg.manual_scope,
                        manual_unit_prefix_template=cfg.manual_unit_prefix_template,
                    )
                    print(f"Building heatmap for mode={mode} cycle={cycle} ...")
                    matches, diag = build_topk_window_manual_heatmap(cfg)
                    if matches is None or matches.empty:
                        print(f"No matches produced for {name}; skipping save")
                        continue
                    path = save_matches(matches, diag, name, outdir=outdir)
                    print(f"Saved {name} to {path}")
                    summary_created.append(str(path))
        else:
            # No cycles specified: if sessions provided, produce session-only prebuilt files
            if sessions:
                for session in sessions:
                    safe_session = str(session).replace(' ', '_')
                    name = f"{mode}_session{safe_session}"
                    p = outdir / f"{name}.parquet"
                    p_csv = outdir / f"{name}.csv"
                    if (p.exists() or p_csv.exists()) and not args.force:
                        print(f"Skipping {name}: already exists (use --force to rebuild)")
                        continue
                    if args.dryrun:
                        print(f"[dryrun] Would build {name}")
                        continue

                    cfg = _default_config_for_mode(mode)
                    cfg = HeatmapConfig(
                        selection_mode=cfg.selection_mode,
                        cycle_id="",
                        session_num=str(session),
                        topic_id=cfg.topic_id,
                        context_window=cfg.context_window,
                        max_windows=cfg.max_windows,
                        random_sample=cfg.random_sample,
                        sample_seed=cfg.sample_seed,
                        max_window_chars=cfg.max_window_chars,
                        topk=cfg.topk,
                        manual_scope=cfg.manual_scope,
                        manual_unit_prefix_template=cfg.manual_unit_prefix_template,
                    )
                    print(f"Building heatmap for mode={mode} session={session} ...")
                    matches, diag = build_topk_window_manual_heatmap(cfg)
                    if matches is None or matches.empty:
                        print(f"No matches produced for {name}; skipping save")
                        continue
                    path = save_matches(matches, diag, name, outdir=outdir)
                    print(f"Saved {name} to {path}")
                    summary_created.append(str(path))

    print("Build complete. Created files:")
    for p in summary_created:
        print(" -", p)


if __name__ == "__main__":
    main()
