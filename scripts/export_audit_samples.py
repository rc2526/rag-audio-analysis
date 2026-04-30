#!/usr/bin/env python3
"""Export stratified audit samples for manual-unit ↔ transcript-window evidence.

Creates CSV files under data/derived/audit_samples/<cycle>_audit_sample.csv for human
labeling. Defaults min_similarity to 0.65 and includes bins centered around the
threshold so you can evaluate calibration.

Usage examples:
  # all cycles, 30 rows per stratum
  python3 scripts/export_audit_samples.py --n-per-stratum 30

  # specific cycles
  python3 scripts/export_audit_samples.py --cycles PMHCycle1 PMHCycle2 --n-per-stratum 40

Outputs: CSV with columns useful for labeling and analysis (includes full manual text
and expanded window text where available).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR, MANUAL_UNITS_CSV
from rag_audio_analysis.source_bridge import (
    get_rag_index_rows,
    build_doc_index_by_path,
    expand_transcript_context,
    build_manual_unit_index,
)


DEFAULT_MIN_SIM = 0.65
OUT_BASE = Path("data/derived/audit_samples")
OUT_BASE.mkdir(parents=True, exist_ok=True)

SCORE_BINS = [0.0, 0.55, 0.65, 0.75, 1.0]
SCORE_BIN_LABELS = ["<0.55", "0.55-0.65", "0.65-0.75", ">=0.75"]
RANK_BUCKETS = ["top1", "top2-5", ">5"]


def rank_bucket_from_rank(r: Any) -> str:
    try:
        iv = int(r)
    except Exception:
        return ">5"
    if iv <= 1:
        return "top1"
    if 2 <= iv <= 5:
        return "top2-5"
    return ">5"


def score_bin_label(score: float) -> str:
    for i in range(len(SCORE_BINS) - 1):
        lo = SCORE_BINS[i]
        hi = SCORE_BINS[i + 1]
        if score >= lo and score < hi:
            return SCORE_BIN_LABELS[i]
    return SCORE_BIN_LABELS[-1]


def load_manual_units_index() -> dict[str, dict[str, Any]]:
    # key by manual_unit_id for quick lookup
    units = build_manual_unit_index()
    by_id: dict[str, dict[str, Any]] = {}
    for u in units:
        uid = str(u.get("manual_unit_id") or u.get("id") or "").strip()
        if not uid:
            continue
        by_id[uid] = u
    return by_id


def expand_window_text_safe(doc_index: Any, meta_rows: list[dict[str, Any]], path_lookup: dict[str, list[int]]):
    try:
        if doc_index is None or str(doc_index).strip() == "":
            return ""
        return expand_transcript_context(int(doc_index), meta_rows=meta_rows, path_lookup=path_lookup, window=2).get("text", "")
    except Exception:
        return ""


def export_for_cycle(cycle: str, n_per_stratum: int = 30, min_similarity: float = DEFAULT_MIN_SIM):
    cycle_dir = Path(CYCLE_ANALYSIS_DIR) / cycle
    sim_path = cycle_dir / "session_manual_similarity_evidence.csv"
    if not sim_path.exists():
        print(f"Skipping {cycle}: no evidence CSV at {sim_path}")
        return

    df = pd.read_csv(sim_path)
    score_col_candidates = ["mapped_manual_unit_match_score", "manual_unit_match_score", "mapped_manual_unit_match_score_combined", "score"]
    score_col = next((c for c in score_col_candidates if c in df.columns), None)
    if score_col is None:
        # try to coerce a numeric column named 'score'
        score_col = "mapped_manual_unit_match_score"
        df[score_col] = 0.0
    else:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

    # add acceptance flag by threshold
    df["accepted_by_min_similarity"] = df[score_col] >= float(min_similarity)

    # normalize retrieval_rank maybe string
    if "retrieval_rank" in df.columns:
        df["retrieval_rank_int"] = pd.to_numeric(df["retrieval_rank"], errors="coerce").fillna(9999).astype(int)
    else:
        df["retrieval_rank_int"] = 9999

    # attach bin labels
    df["score_bin"] = df[score_col].map(lambda v: score_bin_label(float(v)))
    df["rank_bucket"] = df["retrieval_rank_int"].map(lambda r: rank_bucket_from_rank(int(r)))

    # load rag index rows for expanding full window text
    meta_rows = get_rag_index_rows()
    path_lookup = build_doc_index_by_path(meta_rows)

    # load manual units by id
    manual_by_id = load_manual_units_index()

    strata = []
    for sb in SCORE_BIN_LABELS:
        for rb in RANK_BUCKETS:
            strata.append((sb, rb))

    sampled_rows = []
    rng = random.Random(1)

    for sb, rb in strata:
        subset = df[(df["score_bin"] == sb) & (df["rank_bucket"] == rb)]
        cnt = len(subset)
        want = n_per_stratum
        if cnt == 0:
            continue
        if cnt <= want:
            chosen = subset.index.tolist()
        else:
            chosen = rng.sample(list(subset.index), want)
        for idx in chosen:
            row = df.loc[idx].to_dict()
            uid = str(row.get("query_manual_unit_id", "") or "").strip()
            manual_unit = manual_by_id.get(uid, {})
            full_manual_text = manual_unit.get("text") or manual_unit.get("matching_text") or ""
            doc_index = row.get("doc_index", "")
            full_window = expand_window_text_safe(doc_index, meta_rows, path_lookup)
            out_row = {
                "cycle": cycle,
                "manual_session_num": row.get("manual_session_num", ""),
                "query_manual_unit_id": uid,
                "query_manual_unit_subsection": row.get("query_manual_unit_subsection", ""),
                "query_text_excerpt": row.get("query_text_excerpt", ""),
                "full_manual_unit_text": full_manual_text,
                "doc_index": doc_index,
                "doc_path": row.get("doc_path", ""),
                "session_id": row.get("session_id", ""),
                "speaker": row.get("speaker", ""),
                "text_excerpt": row.get("text_excerpt", ""),
                "full_window_text": full_window,
                "mapped_manual_unit_match_score": float(row.get(score_col) or 0.0),
                "retrieval_rank": row.get("retrieval_rank", ""),
                "mapped_manual_unit_accepted": bool(row.get("accepted_by_min_similarity", False)),
                "score_bin": row.get("score_bin", ""),
                "rank_bucket": row.get("rank_bucket", ""),
            }
            sampled_rows.append(out_row)

    # also include a small set of low-score negative controls (score < min_similarity)
    neg_subset = df[df[score_col] < float(min_similarity)]
    neg_sample_size = min(n_per_stratum, max(10, int(0.05 * len(df))))
    if len(neg_subset) > 0:
        neg_chosen = rng.sample(list(neg_subset.index), min(len(neg_subset), neg_sample_size))
        for idx in neg_chosen:
            row = df.loc[idx].to_dict()
            uid = str(row.get("query_manual_unit_id", "") or "").strip()
            manual_unit = manual_by_id.get(uid, {})
            full_manual_text = manual_unit.get("text") or manual_unit.get("matching_text") or ""
            doc_index = row.get("doc_index", "")
            full_window = expand_window_text_safe(doc_index, meta_rows, path_lookup)
            sampled_rows.append(
                {
                    "cycle": cycle,
                    "manual_session_num": row.get("manual_session_num", ""),
                    "query_manual_unit_id": uid,
                    "query_manual_unit_subsection": row.get("query_manual_unit_subsection", ""),
                    "query_text_excerpt": row.get("query_text_excerpt", ""),
                    "full_manual_unit_text": full_manual_text,
                    "doc_index": doc_index,
                    "doc_path": row.get("doc_path", ""),
                    "session_id": row.get("session_id", ""),
                    "speaker": row.get("speaker", ""),
                    "text_excerpt": row.get("text_excerpt", ""),
                    "full_window_text": full_window,
                    "mapped_manual_unit_match_score": float(row.get(score_col) or 0.0),
                    "retrieval_rank": row.get("retrieval_rank", ""),
                    "mapped_manual_unit_accepted": bool(row.get("accepted_by_min_similarity", False)),
                    "score_bin": row.get("score_bin", ""),
                    "rank_bucket": row.get("rank_bucket", ""),
                }
            )

    # write output
    out_path = OUT_BASE / f"{cycle}_audit_sample.csv"
    out_df = pd.DataFrame(sampled_rows)
    if out_df.empty:
        print(f"No samples generated for {cycle}")
        return
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} sample rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="*", help="Cycle names to process (default: discover from CYCLE_ANALYSIS_DIR)")
    parser.add_argument("--n-per-stratum", type=int, default=30, help="Samples per (score_bin,rank_bucket) stratum")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIM, help="Min similarity threshold used for marking accepted rows")
    args = parser.parse_args()

    base = Path(CYCLE_ANALYSIS_DIR)
    if args.cycles:
        cycles = args.cycles
    else:
        cycles = sorted([p.name for p in base.iterdir() if p.is_dir()])

    for c in cycles:
        export_for_cycle(c, n_per_stratum=args.n_per_stratum, min_similarity=args.min_similarity)


if __name__ == "__main__":
    main()
