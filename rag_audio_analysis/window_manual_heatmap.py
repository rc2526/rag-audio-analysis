from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
import json

from rag_audio_analysis.source_bridge import (
    build_doc_index_by_path,
    build_manual_unit_index,
    encode_texts,
    expand_transcript_context,
    get_manual_unit_embedding_index,
    get_manual_units_for_session,
    get_rag_index_rows,
    infer_cycle_id,
    infer_session_id,
    is_manual_row,
)


SelectionMode = Literal[
    "cycle_only",
]


@dataclass(frozen=True)
class HeatmapConfig:
    selection_mode: SelectionMode = "cycle_only"
    cycle_id: str = ""
    session_num: str = ""
    topic_id: str = ""
    context_window: int = 2
    max_windows: int = 200
    random_sample: bool = False
    sample_seed: int = 0
    max_window_chars: int = 0
    topk: int = 20
    manual_scope: Literal["session", "global"] = "session"
    manual_unit_prefix_template: str = "Session {session_num} "


_EMBED_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def _stable_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _infer_model_id() -> str:
    # The embed model comes from the upstream rag-audio module; treat its identity
    # as best-effort (this is only used for caching).
    try:
        from rag_audio_analysis.source_bridge import get_embedding_model

        model = get_embedding_model()
        for attr in ("model_name", "model_name_or_path", "name_or_path"):
            value = getattr(model, attr, None)
            if value:
                return str(value)
        return model.__class__.__name__
    except Exception:
        return "unknown_model"


def select_window_centers(
    *,
    meta_rows: Optional[List[Dict[str, Any]]] = None,
    selection_mode: SelectionMode,
    cycle_id: str = "",
    session_num: str = "",
    max_windows: int = 200,
    random_sample: bool = False,
    sample_seed: int = 0,
) -> List[int]:
    rows = meta_rows or get_rag_index_rows()
    cycle_id = str(cycle_id or "").strip()
    session_num = str(session_num or "").strip()

    centers: List[int] = []
    for idx, row in enumerate(rows):
        if is_manual_row(row):
            continue
        path = str(row.get("path", "") or "")
        # For the single supported mode (cycle_only), apply cycle filter when provided.
        if cycle_id and infer_cycle_id(path) != cycle_id:
            continue
        centers.append(idx)

    if max_windows and len(centers) > max_windows:
        if random_sample:
            rng = random.Random(int(sample_seed))
            centers = rng.sample(centers, k=int(max_windows))
        else:
            centers = centers[: int(max_windows)]
    return centers


def _window_doc_indices(
    *,
    doc_index: int,
    meta_rows: List[Dict[str, Any]],
    path_lookup: Dict[str, List[int]],
    context_window: int,
) -> List[int]:
    if not (0 <= doc_index < len(meta_rows)):
        return []
    row = meta_rows[doc_index]
    path = str(row.get("path", "") or "")
    if not path:
        return [doc_index]
    indices = path_lookup.get(path, [])
    if doc_index not in indices:
        return [doc_index]
    loc = indices.index(doc_index)
    start = max(loc - context_window, 0)
    end = min(loc + context_window + 1, len(indices))
    selected = [i for i in indices[start:end] if not is_manual_row(meta_rows[i])]
    return selected


def build_windows_from_centers(
    *,
    centers: Sequence[int],
    meta_rows: Optional[List[Dict[str, Any]]] = None,
    context_window: int = 2,
    max_window_chars: int = 0,
) -> List[Dict[str, Any]]:
    rows = meta_rows or get_rag_index_rows()
    path_lookup = build_doc_index_by_path(rows)
    context_window = int(context_window)
    max_window_chars = int(max_window_chars or 0)

    windows: List[Dict[str, Any]] = []
    for center in centers:
        expanded = expand_transcript_context(
            int(center),
            meta_rows=rows,
            path_lookup=path_lookup,
            window=context_window,
        )
        text = str(expanded.get("text", "") or "")
        if max_window_chars and len(text) > max_window_chars:
            text = text[:max_window_chars]
        path = str(expanded.get("path", "") or "")
        windows.append(
            {
                "window_id": f"WIN_{int(center)}",
                "center_doc_index": int(center),
                "path": path,
                "cycle_id": infer_cycle_id(path),
                "session_num": str(infer_session_id(path) or ""),
                "turn_count": int(expanded.get("turn_count", 0) or 0),
                "window_text": text,
                "included_doc_indices": _window_doc_indices(
                    doc_index=int(center),
                    meta_rows=rows,
                    path_lookup=path_lookup,
                    context_window=context_window,
                ),
            }
        )
    return windows


def embed_texts_with_cache(texts: Sequence[str]) -> np.ndarray:
    model_id = _infer_model_id()
    vectors: List[np.ndarray] = []
    missing_texts: List[str] = []
    missing_keys: List[Tuple[str, str]] = []

    for text in texts:
        text_str = str(text or "")
        key = (model_id, _stable_text_hash(text_str))
        if key in _EMBED_CACHE:
            vectors.append(_EMBED_CACHE[key])
        else:
            vectors.append(None)  # type: ignore[arg-type]
            missing_texts.append(text_str)
            missing_keys.append(key)

    if missing_texts:
        encoded = encode_texts(list(missing_texts))
        for i, key in enumerate(missing_keys):
            vec = encoded[i]
            _EMBED_CACHE[key] = vec

        # Fill any None placeholders in order.
        it = iter(encoded)
        filled: List[np.ndarray] = []
        for vec in vectors:
            if vec is None:
                filled.append(next(it))
            else:
                filled.append(vec)
        vectors = filled

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def _manual_units_and_embeddings(
    *,
    manual_scope: Literal["session", "global"],
    session_num: str = "",
    topic_id: str = "",
    manual_unit_prefix_template: str = "",
    selection_mode: SelectionMode = "cycle_only",
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    session_num = str(session_num or "").strip()
    topic_id = str(topic_id or "").strip()

    if manual_scope == "session" and session_num:
        units = get_manual_units_for_session(session_num, topic_id=topic_id)
    else:
        units = build_manual_unit_index()

    # No prefix-bias mode in the simplified flow; fall through to using cached global embeddings

    # Prefer the cached global manual-unit embeddings when possible.
    global_units, global_emb = get_manual_unit_embedding_index()
    if not global_units or global_emb is None or global_emb.size == 0:
        texts = [str(u.get("matching_text") or u.get("text") or "").strip() for u in units]
        return units, encode_texts(texts)

    emb_by_id = {str(u.get("manual_unit_id", "")): global_emb[i] for i, u in enumerate(global_units)}
    embeddings_list: List[np.ndarray] = []
    missing = False
    for u in units:
        key = str(u.get("manual_unit_id", ""))
        if key in emb_by_id:
            embeddings_list.append(emb_by_id[key])
        else:
            missing = True
            break
    if missing:
        texts = [str(u.get("matching_text") or u.get("text") or "").strip() for u in units]
        return units, encode_texts(texts)
    return units, np.vstack(embeddings_list).astype(np.float32)


def topk_sparse_matches(
    *,
    windows: Sequence[Dict[str, Any]],
    window_embeddings: np.ndarray,
    manual_units: Sequence[Dict[str, Any]],
    manual_embeddings: np.ndarray,
    topk: int = 20,
) -> pd.DataFrame:
    if not windows or not manual_units:
        return pd.DataFrame()
    if window_embeddings.size == 0 or manual_embeddings.size == 0:
        return pd.DataFrame()

    w = window_embeddings.astype(np.float32)
    m = manual_embeddings.astype(np.float32)
    sims = w.dot(m.T)  # (n_windows, n_units)

    k = int(topk or 0)
    if k <= 0:
        k = min(20, sims.shape[0])
    k = min(k, sims.shape[0])

    rows: List[Dict[str, Any]] = []
    for j, unit in enumerate(manual_units):
        col = sims[:, j]
        if k == len(col):
            idxs = np.argsort(-col)
        else:
            idxs = np.argpartition(-col, kth=k - 1)[:k]
            idxs = idxs[np.argsort(-col[idxs])]
        for rank, i in enumerate(idxs, start=1):
            win = windows[int(i)]
            score = float(col[int(i)])
            rows.append(
                {
                    "manual_unit_id": unit.get("manual_unit_id", ""),
                    "manual_week": unit.get("manual_week", ""),
                    "manual_section": unit.get("manual_section", ""),
                    "manual_subsection": unit.get("manual_subsection", ""),
                    "manual_category": unit.get("manual_category", ""),
                    "window_id": win.get("window_id", ""),
                    "center_doc_index": win.get("center_doc_index", ""),
                    "path": win.get("path", ""),
                    "cycle_id": win.get("cycle_id", ""),
                    "session_num": win.get("session_num", ""),
                    "turn_count": win.get("turn_count", ""),
                    "score": score,
                    "rank": int(rank),
                    "manual_unit_text": unit.get("text", ""),
                    "manual_unit_matching_text": unit.get("matching_text", ""),
                    "window_text": win.get("window_text", ""),
                    "included_doc_indices": win.get("included_doc_indices", []),
                }
            )
    return pd.DataFrame(rows)


def build_topk_window_manual_heatmap(
    config: HeatmapConfig,
    *,
    meta_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = meta_rows or get_rag_index_rows()

    centers = select_window_centers(
        meta_rows=rows,
        selection_mode=config.selection_mode,
        cycle_id=config.cycle_id,
        session_num=config.session_num,
        max_windows=config.max_windows,
        random_sample=config.random_sample,
        sample_seed=config.sample_seed,
    )

    windows = build_windows_from_centers(
        centers=centers,
        meta_rows=rows,
        context_window=config.context_window,
        max_window_chars=config.max_window_chars,
    )

    window_texts = [str(w.get("window_text", "") or "") for w in windows]
    window_emb = embed_texts_with_cache(window_texts)

    manual_units, manual_emb = _manual_units_and_embeddings(
        manual_scope=config.manual_scope,
        session_num=config.session_num,
        topic_id=config.topic_id,
        manual_unit_prefix_template=config.manual_unit_prefix_template,
        selection_mode=config.selection_mode,
    )

    matches = topk_sparse_matches(
        windows=windows,
        window_embeddings=window_emb,
        manual_units=manual_units,
        manual_embeddings=manual_emb,
        topk=config.topk,
    )

    diagnostics = {
        "candidate_centers_count": len(centers),
        "candidate_windows_count": len(windows),
        "context_window": int(config.context_window),
        "turns_per_window_target": int(2 * int(config.context_window) + 1),
        "max_window_chars": int(config.max_window_chars or 0),
        "selection_mode": config.selection_mode,
        "manual_scope": config.manual_scope,
        "topk": int(config.topk),
        "cycle_id": config.cycle_id,
        "session_num": config.session_num,
        "topic_id": config.topic_id,
        "random_sample": bool(config.random_sample),
        "sample_seed": int(config.sample_seed),
    }

    if windows:
        turn_counts = [int(w.get("turn_count", 0) or 0) for w in windows]
        diag_chars = [len(str(w.get("window_text", "") or "")) for w in windows]
        diagnostics.update(
            {
                "min_turn_count_per_window": int(min(turn_counts)),
                "avg_turn_count_per_window": float(sum(turn_counts) / len(turn_counts)),
                "max_turn_count_per_window": int(max(turn_counts)),
                "avg_chars_per_window": float(sum(diag_chars) / len(diag_chars)),
                "p95_chars_per_window": float(np.percentile(diag_chars, 95)),
                "sample_window": windows[0],
            }
        )

    return matches, diagnostics


def save_matches(
    matches: pd.DataFrame,
    diagnostics: Dict[str, Any],
    mode: str,
    outdir: str | Path = "data/derived/topk_window_manual_heatmaps",
) -> Path:
    """Persist matches and diagnostics for a given mode.

    Tries Parquet first, falls back to CSV if Parquet/pyarrow is not available.
    Returns the path to the primary matches file written.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    p_parquet = out / f"{mode}.parquet"
    p_csv = out / f"{mode}.csv"
    p_diag = out / f"{mode}_diagnostics.json"

    try:
        matches.to_parquet(p_parquet)
        primary = p_parquet
    except Exception:
        matches.to_csv(p_csv, index=False)
        primary = p_csv

    try:
        with p_diag.open("w", encoding="utf-8") as fh:
            json.dump(diagnostics or {}, fh)
    except Exception:
        # best-effort; don't fail the whole save on diagnostics
        pass

    return primary


def load_matches(
    mode: str, outdir: str | Path = "data/derived/topk_window_manual_heatmaps"
) -> Tuple[pd.DataFrame | None, Dict[str, Any] | None]:
    """Load previously saved matches and diagnostics for a given mode.

    Returns (matches_df or None, diagnostics or None).
    """
    out = Path(outdir)
    p_parquet = out / f"{mode}.parquet"
    p_csv = out / f"{mode}.csv"
    p_diag = out / f"{mode}_diagnostics.json"

    df = None
    diag = None
    try:
        if p_parquet.exists():
            df = pd.read_parquet(p_parquet)
        elif p_csv.exists():
            df = pd.read_csv(p_csv)
    except Exception:
        df = None

    try:
        if p_diag.exists():
            with p_diag.open("r", encoding="utf-8") as fh:
                diag = json.load(fh)
    except Exception:
        diag = None

    return df, diag

