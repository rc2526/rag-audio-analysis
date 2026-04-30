"""Lightweight RAG service: retrieval + answer wrappers used by the Streamlit demo.

This is a minimal layer that performs similarity search over cached window embeddings
and builds prompts using chat_runner.build_chat_prompt(). It supports RAG and non-RAG
answering and returns structured payloads suitable for UI display.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List
from pathlib import Path
import json

from rag_audio_analysis.indexer import build_cycle_window_index, load_cycle_window_index
from rag_audio_analysis.source_bridge import (
    normalize_rows,
    build_manual_unit_index,
    get_rag_index_rows,
    expand_transcript_context,
    build_doc_index_by_path,
)
from rag_audio_analysis.chat_runner import build_chat_prompt, call_ollama, _nonrag_prompt
from rag_audio_analysis.settings import get_float, get_int, get_str


def _cosine_search(query_emb: np.ndarray, window_embs: np.ndarray, top_k: int = 8, min_sim: float = 0.0):
    if query_emb is None or query_emb.size == 0:
        return []
    if window_embs is None or window_embs.size == 0:
        return []
    sims = float(query_emb) * window_embs  # handle scalar vs array gracefully
    # if query_emb is 1-D and window_embs is (N,d), compute dot
    try:
        sims = np.matmul(query_emb, window_embs.T)
    except Exception:
        sims = np.dot(window_embs, query_emb)
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for rank, i in enumerate(idxs, start=1):
        score = float(sims[i])
        if score < min_sim:
            continue
        results.append((int(i), float(score), rank))
    return results


def retrieve_for_question(question: str, cycle: str, top_k: int | None = None, min_similarity: float | None = None):
    # build index if missing
    try:
        build_cycle_window_index(cycle, force=False)
    except Exception:
        # best-effort: continue if index exists
        pass
    embs, metas = load_cycle_window_index(cycle)
    # embed the question using chat_runner's model via source_bridge.encode_texts indirectly
    from rag_audio_analysis.source_bridge import encode_texts

    q_emb = encode_texts([question])
    if q_emb.size == 0:
        return {"windows": [], "manuals": []}
    # ensure window embeddings are normalized
    if embs.size == 0:
        window_embs = np.zeros((0, 0), dtype=np.float32)
    else:
        window_embs = normalize_rows(np.asarray(embs, dtype=np.float32))
    # resolve defaults from settings.ini when not provided
    if top_k is None:
        top_k = get_int("chat", "topk", 8)
    if min_similarity is None:
        # prefer topic_high_similarity (0.55) for retrieval threshold
        min_similarity = get_float("topic_matching", "topic_high_similarity", 0.55)

    # compute similarities to window embeddings
    qvec = q_emb[0]
    window_results: List[dict[str, Any]] = []
    if window_embs.size != 0:
        sims_w = window_embs.dot(qvec)
        # collect candidate window hits above threshold
        for idx_w, score in enumerate(sims_w.tolist()):
            if float(score) >= float(min_similarity):
                meta = metas[idx_w] if idx_w < len(metas) else {}
                window_results.append({
                    "source": "window",
                    "doc_index": meta.get("doc_index"),
                    "path": meta.get("path"),
                    "speaker": meta.get("speaker"),
                    "text": "",  # expansion left to UI
                    "score": float(score),
                })

    # compute similarities to manual units (use canonical manual unit texts)
    manual_results: List[dict[str, Any]] = []
    try:
        manual_units = build_manual_unit_index()
        manual_texts = [str(u.get("matching_text") or u.get("text") or "") for u in manual_units]
        if any(manual_texts):
            manual_embs = encode_texts(manual_texts)
            if manual_embs.size != 0:
                manual_embs = normalize_rows(np.asarray(manual_embs, dtype=np.float32))
                sims_m = manual_embs.dot(qvec)
                for idx_m, score in enumerate(sims_m.tolist()):
                    if float(score) >= float(min_similarity):
                        mu = manual_units[idx_m]
                        # include standardized session identifiers so downstream UI can display session labels
                        manu_week = mu.get("manual_week")
                        manual_results.append({
                            "source": "manual",
                            "manual_unit_id": mu.get("manual_unit_id"),
                            "manual_week": manu_week,
                            "session_num": manu_week,
                            "manual_session": (f"Session {manu_week}" if manu_week else ""),
                            "text": mu.get("text") or mu.get("matching_text") or "",
                            "matching_text": mu.get("matching_text") or mu.get("text") or "",
                            "score": float(score),
                        })
    except Exception:
        manual_results = []

    # merge and pick top-k by score
    combined = []
    for w in window_results:
        combined.append({**w})
    for m in manual_results:
        combined.append({**m})
    combined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    combined = combined[:int(top_k)] if top_k and len(combined) > top_k else combined

    # split back into windows and manuals to return structured payload
    windows_out = [c for c in combined if c.get("source") == "window"]
    manuals_out = [c for c in combined if c.get("source") == "manual"]
    return {"windows": windows_out, "manuals": manuals_out}


def answer_rag(
    question: str,
    cycle: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
    model: str | None = None,
    prompt_variant: str = "default",
    history: list[dict[str, str]] | None = None,
    history_turns: int = 2,
    preview_prompt: bool = False,
):
    """Answer a question using retrieval; if `history` is provided (list of turns with 'role' and 'text'),
    include the last `history_turns` turns as context prefixed to the question so the model sees prior messages.
    """

    res = retrieve_for_question(question, cycle, top_k=top_k, min_similarity=min_similarity)
    windows = res.get("windows", [])
    manuals = res.get("manuals", [])

    # Enrich manual hits with canonical session IDs (manual_week) from the built manual index
    try:
        from rag_audio_analysis.source_bridge import build_manual_unit_index

        canonical_units = build_manual_unit_index() or []
        manu_map = {str(u.get("manual_unit_id", "")): str(u.get("manual_week", "")) for u in canonical_units}
        for m in manuals:
            mid = str(m.get("manual_unit_id", "") or "")
            if mid:
                mw = manu_map.get(mid, "")
                if mw and not m.get("session_num"):
                    m["session_num"] = mw
                if mw and not m.get("manual_session"):
                    m["manual_session"] = f"Session {mw}"
                # also populate a session_id field for prompt clarity
                if mw and not m.get("session_id"):
                    m["session_id"] = mw
    except Exception:
        # best-effort enrichment; continue if canonical index unavailable
        pass

    # expand transcript window text so prompts and UI include the full window body
    try:
        meta_rows = get_rag_index_rows()
        path_lookup = build_doc_index_by_path(meta_rows)
        for w in windows:
            di = w.get("doc_index")
            if di is not None:
                try:
                    ctx = expand_transcript_context(int(di), meta_rows=meta_rows, path_lookup=path_lookup, window=2)
                    w["text"] = ctx.get("text", "")
                    # ensure speaker/path are updated from canonical context when available
                    if not w.get("speaker"):
                        w["speaker"] = ctx.get("speaker", "")
                    if not w.get("path"):
                        w["path"] = ctx.get("path", "")
                except Exception:
                    # leave text as-is on any failure
                    pass
    except Exception:
        # if index/meta rows unavailable, continue without expansion
        meta_rows = []

    # prepare evidence_rows combining transcript windows and manual units so the default
    # prompt (which renders evidence_rows) will include manual-unit evidence as well.
    combined_items: list[dict] = []
    for w in windows:
        combined_items.append({
            "source": "window",
            "score": float(w.get("score", 0.0) or 0.0),
            # prefer explicit session/transcript identifiers when available
            "session_label": w.get("session_num") or w.get("transcript_id") or "",
            "session_id": "",
            "manual_session": "",
            "manual_unit_id_best_match": "",
            "manual_unit_match_score": "",
            "text": w.get("text", f"(transcript window {w.get('doc_index')})"),
        })
    for m in manuals:
        combined_items.append({
            "source": "manual",
            "score": float(m.get("score", 0.0) or 0.0),
            # prefer explicit session/transcript identifiers when available on manual units
            "session_label": m.get('session_num') or m.get('transcript_id') or m.get('manual_session') or "",
            "session_id": "",
            "manual_session": f"Manual {m.get('manual_unit_id','')}",
            "manual_unit_id_best_match": m.get("manual_unit_id", ""),
            "manual_unit_match_score": float(m.get("score", 0.0) or 0.0),
            # prefer matching_text (heading + chunk) for prompts; fall back to short text
            "matching_text": m.get("matching_text", m.get("text", "")),
            "text": m.get("matching_text", m.get("text", "")),
        })

    # sort by score so highest-scoring evidence appears first
    combined_items.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    # respect top_k if provided (keep same semantics as retrieve_for_question)
    try:
        if top_k:
            combined_items = combined_items[: int(top_k)]
    except Exception:
        pass

    evidence_rows = []
    for item in combined_items:
        evidence_rows.append({
            "session_label": item.get("session_label", ""),
            "session_id": item.get("session_id", ""),
            "manual_session": item.get("manual_session", ""),
            "manual_unit_id_best_match": item.get("manual_unit_id_best_match", ""),
            "manual_unit_match_score": item.get("manual_unit_match_score", ""),
            "text": item.get("text", ""),
            "matching_text": item.get("matching_text", ""),
        })

    # prepare manual_units payload: pass canonical manual units into the prompt so they appear as standalone evidence
    manual_units_payload = []
    for m in manuals:
        manual_units_payload.append({
            "manual_unit_id": m.get("manual_unit_id"),
            "manual_week": m.get("manual_week"),
            "text": m.get("text", ""),
            "matching_text": m.get("matching_text", ""),
            "manual_subsection": "",
        })

    # If history is provided, prepend a concise transcript of recent turns to the question so
    # the prompt includes conversation context. History entries are expected to be dicts with
    # keys 'role' and 'text'. This keeps backwards compatibility: callers that do not pass
    # history are unaffected.
    effective_question = question
    try:
        if history:
            recent = [h for h in history if h.get("role") and h.get("text")]
            # take last N turns
            recent = recent[-int(history_turns) :]
            prefix_lines = []
            for h in recent:
                r_role = str(h.get("role", "")).capitalize()
                r_text = str(h.get("text", ""))
                prefix_lines.append(f"{r_role}: {r_text}")
            prefix = "\n".join(prefix_lines)
            effective_question = prefix + "\nCurrent question: " + str(question)
    except Exception:
        # on any failure, fall back to original question
        effective_question = question

    prompt = build_chat_prompt(
        effective_question, evidence_rows, variant=prompt_variant, cycle_id=cycle, manual_units=manual_units_payload
    )
    # If requested, prepend strict JSON-output instructions and return the prompt without calling the model
    if preview_prompt:
        json_instructions = (
            "You must first output a single valid JSON object matching this schema:\n"
            "{\n  \"answer_summary\": string,\n  \"evidence_refs\": [string],\n  \"confidence\": \"low\"|\"medium\"|\"high\",\n  \"rationale\": string\n}\n"
            "After the JSON object, on a new line, print one human-friendly sentence summarizing the answer.\n"
            "Do NOT include session identifiers or manual unit ids in the JSON object.\n"
            "The JSON object should be the first thing you output."
        )
        prompt = json_instructions + "\n" + prompt
        return {"prompt": prompt, "question": question, "cycle": cycle, "windows": windows, "manuals": manuals}
    try:
        if model is None:
            model = get_str("ollama", "default_model", "gpt-oss:120b")
        answer_text = call_ollama(prompt, model or get_str("ollama", "default_model", "gpt-oss:120b"))
    except Exception as exc:
        answer_text = json.dumps({"error": str(exc)})
    return {"question": question, "cycle": cycle, "windows": windows, "manuals": manuals, "prompt": prompt, "answer_raw": answer_text}


def answer_non_rag(question: str, model: str | None = None):
    # non-RAG: use a very small, direct prompt
    prompt = _nonrag_prompt(question)
    try:
        if model is None:
            model = get_str("ollama", "default_model", "gpt-oss:120b")
        answer_text = call_ollama(prompt, model or get_str("ollama", "default_model", "gpt-oss:120b"))
    except Exception as exc:
        answer_text = json.dumps({"error": str(exc)})
    return {"question": question, "prompt": prompt, "answer_raw": answer_text}
