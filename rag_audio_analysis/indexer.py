"""Indexing utilities for transcript-window embeddings per cycle.

This module creates and loads cached window embeddings and metadata for a cycle.
It reuses encode_texts() and get_rag_index_rows() from source_bridge.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import json
from typing import Any, Tuple

from rag_audio_analysis.source_bridge import get_rag_index_rows, expand_transcript_context, encode_texts
from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR

INDEX_BASE = Path("data/derived/cycle_analysis/indexes")
INDEX_BASE.mkdir(parents=True, exist_ok=True)


def build_cycle_window_index(cycle: str, window: int = 2, force: bool = False) -> Tuple[Path, Path]:
    """Builds and caches window embeddings and metadata for a cycle.

    Returns (emb_path, meta_path)
    """
    out_dir = INDEX_BASE / cycle
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "windows.npy"
    meta_path = out_dir / "windows_meta.json"

    if emb_path.exists() and meta_path.exists() and not force:
        return emb_path, meta_path

    meta_rows = get_rag_index_rows()
    # collect texts for windows belonging to the cycle
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    path_lookup = {}  # not used here but required by expand
    for idx, row in enumerate(meta_rows):
        path = str(row.get("path") or row.get("file") or "")
        if not path:
            continue
        # naive cycle membership test: check cycle substring in path
        if cycle not in path:
            continue
        ctx = expand_transcript_context(idx, meta_rows=meta_rows, path_lookup=path_lookup, window=window)
        text = str(ctx.get("text", "") or "").strip()
        if not text:
            continue
        texts.append(text)
        metas.append({"doc_index": idx, "path": path, "speaker": ctx.get("speaker", "")})

    if not texts:
        # write empty placeholders
        np.save(emb_path, np.zeros((0, 0), dtype=np.float32))
        meta_path.write_text(json.dumps([]), encoding="utf-8")
        return emb_path, meta_path

    embs = encode_texts(texts)
    # persist as float32
    np.save(emb_path, embs.astype("float32"))
    meta_path.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    return emb_path, meta_path


def load_cycle_window_index(cycle: str) -> Tuple[np.ndarray, list[dict[str, Any]]]:
    out_dir = INDEX_BASE / cycle
    emb_path = out_dir / "windows.npy"
    meta_path = out_dir / "windows_meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index for cycle {cycle} not found; run build_cycle_window_index first")
    embs = np.load(emb_path)
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    return embs, metas
