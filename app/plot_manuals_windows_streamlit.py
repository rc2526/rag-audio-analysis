"""Streamlit app: scatter of manual units vs transcript windows.

Run with: streamlit run scripts/plot_manuals_windows_streamlit.py

This script:
- Loads manual unit embeddings and metadata (via source_bridge helper or .npy + .json)
- Loads per-cycle window embeddings and metas via load_cycle_window_index(cycle)
- Computes cosine similarities (manuals x windows)
- Builds a matches DataFrame (best manual per window and scores)
- Reduces embeddings (PCA or PCA->UMAP) and renders an interactive Plotly scatter
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

try:
    import umap
except Exception:
    umap = None

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from rag_audio_analysis.indexer import load_cycle_window_index
from rag_audio_analysis.source_bridge import get_manual_unit_embedding_index


ROOT = Path("data/derived")


@st.cache_data(show_spinner=False)
def load_manuals() -> Tuple[pd.DataFrame, np.ndarray]:
    """Return manual rows (DataFrame) and embeddings (np.array).
    Uses repo helper get_manual_unit_embedding_index()."""
    manu_rows, embeddings = get_manual_unit_embedding_index()
    # manu_rows is a list of dicts; normalize to DataFrame
    manu_df = pd.DataFrame(manu_rows)
    em = np.asarray(embeddings, dtype=np.float32)
    return manu_df, em


@st.cache_data(show_spinner=False)
def load_windows_for_cycles(cycles: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    rows = []
    embs_list = []
    for cycle in cycles:
        try:
            embs, metas = load_cycle_window_index(cycle)
        except Exception:
            continue
        if embs is None or len(embs) == 0:
            continue
        embs = np.asarray(embs, dtype=np.float32)
        # metas expected to be list of dicts aligned with embs
        for i, meta in enumerate(metas):
            rows.append(
                {
                    "cycle_id": cycle,
                    "win_idx": int(i),
                    "window_id": meta.get("window_id") or f"{cycle}_w{i}",
                    "center_doc_index": meta.get("doc_index"),
                    "path": meta.get("path"),
                    "session_num": meta.get("session_num"),
                    "window_text": meta.get("text") or "",
                }
            )
            embs_list.append(embs[i])
    if len(embs_list) == 0:
        return pd.DataFrame(columns=["cycle_id", "win_idx", "window_id"]), np.zeros((0, 0))
    W = np.vstack(embs_list)
    df = pd.DataFrame(rows)
    return df, W


def derive_transcript_id(row: pd.Series) -> str:
    try:
        c = str(row.get("cycle_id", "") or "").strip()
        s = str(row.get("session_num", "") or "").strip()
        if c and s:
            return f"{c}_session{s}"
        if c and not s:
            return c
        if s and not c:
            return f"session{s}"
        p = str(row.get("path", "") or "")
        if p:
            return Path(p).stem
        return str(row.get("window_id", "") or "")
    except Exception:
        return str(row.get("window_id", "") or "")


def compute_similarities(manual_embs: np.ndarray, window_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized embeddings and manual->window cosine similarities.

    Returns (S, W_n) where S is (M, N) similarity matrix and W_n is normalized window embs.
    """
    if manual_embs.ndim == 1:
        manual_embs = manual_embs.reshape(1, -1)
    if window_embs.ndim == 1:
        window_embs = window_embs.reshape(1, -1)
    # ensure dims agree
    if manual_embs.shape[1] != window_embs.shape[1]:
        raise ValueError(f"Manual emb dim {manual_embs.shape[1]} != window emb dim {window_embs.shape[1]}")
    M = normalize(manual_embs, axis=1)
    W = normalize(window_embs, axis=1)
    S = M.dot(W.T)
    return S, W


def reduce_embeddings(all_embs: np.ndarray, method: str = "pca+umap") -> np.ndarray:
    """Reduce embeddings to 2D. method in {pca, pca+umap, pca50}.
    Returns (N,2) coords.
    """
    if all_embs.shape[0] == 0:
        return np.zeros((0, 2))
    # PCA to 2 for simple
    if method == "pca":
        p = PCA(n_components=2)
        return p.fit_transform(all_embs)
    # PCA->UMAP default
    if method == "pca+umap":
        n_comp = min(50, all_embs.shape[1])
        p = PCA(n_components=n_comp)
        mid = p.fit_transform(all_embs)
        if umap is None:
            # fallback to PCA 2
            p2 = PCA(n_components=2)
            return p2.fit_transform(all_embs)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
        return reducer.fit_transform(mid)
    # pca50 then pca2 (cheap)
    p = PCA(n_components=min(50, all_embs.shape[1]))
    mid = p.fit_transform(all_embs)
    p2 = PCA(n_components=2)
    return p2.fit_transform(mid)


def main():
    st.set_page_config(layout="wide", page_title="Manuals vs Windows — Embedding Scatter")
    st.title("Manuals vs Transcript Windows — Embedding Scatter")

    # sidebar options
    idx_dir = ROOT / "cycle_analysis" / "indexes"
    cycles_avail = []
    if idx_dir.exists():
        cycles_avail = sorted([p.name for p in idx_dir.iterdir() if p.is_dir()])
    if not cycles_avail:
        cycles_avail = ["PMHCycle1", "PMHCycle2", "PMHCycle3"]

    with st.sidebar:
        st.markdown("## Data & compute options")
        chosen_cycles = st.multiselect("Cycles to include", options=cycles_avail, default=cycles_avail)
        reduction = st.selectbox("Reduction method", options=["pca", "pca+umap", "pca50"], index=1)
        top_k = st.slider("Top-k manuals per window (for table)", min_value=1, max_value=20, value=3)
    # visualization controls
    manual_plot_mode = st.selectbox("Manual plot mode", options=["by_manual_traces", "single_trace"], index=0, help="by_manual_traces shows a legend entry per manual; single_trace draws all manuals as one trace")
    show_lines = st.checkbox("Show window→manual lines", value=True)
    min_line_score = st.slider("Min score to draw a line", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    max_lines = st.number_input("Max lines to draw (0 = no limit)", min_value=0, max_value=10000, value=500)
    recompute = st.button("Compute / Refresh")

    # lazy load
    st.info("Load manual embeddings and window indexes — click Compute / Refresh to run the pipeline.")

    if not recompute:
        st.stop()

    with st.spinner("Loading manuals..."):
        manu_df, manu_embs = load_manuals()
    with st.spinner("Loading windows for cycles: " + ",".join(chosen_cycles)):
        win_df, win_embs = load_windows_for_cycles(chosen_cycles)

    if win_embs.size == 0:
        st.error("No window embeddings found for chosen cycles.")
        return

    try:
        S, Wn = compute_similarities(manu_embs, win_embs)
    except Exception as e:
        st.error(f"Failed to compute similarities: {e}")
        return

    # derive best manual and top-k per window
    best_idx = np.argmax(S, axis=0)
    best_sim = S[best_idx, np.arange(S.shape[1])]
    # top-k ids
    topk_idx = np.argsort(-S, axis=0)[:top_k, :]

    # assemble matches DF
    matches = win_df.copy()
    matches = matches.reset_index(drop=True)
    matches["best_manual_id"] = [manu_df.iloc[i].get("manual_unit_id") if 0 <= i < len(manu_df) else None for i in best_idx]
    matches["best_manual_score"] = best_sim
    # top-k as list of manual ids
    def topk_ids_for_col(col):
        ids = []
        for r in range(col.shape[1]):
            ids.append([manu_df.iloc[int(i)].get("manual_unit_id") for i in col[:, r] if 0 <= int(i) < len(manu_df)])
        return ids

    matches["topk_manual_ids"] = topk_ids_for_col(topk_idx)
    matches["transcript_id"] = matches.apply(derive_transcript_id, axis=1)

    # Combine embeddings for reduction: windows first, then manuals
    all_embs = np.vstack([Wn, normalize(manu_embs, axis=1)])
    coords = reduce_embeddings(all_embs, method=reduction)
    nW = Wn.shape[0]
    win_coords = coords[:nW]
    manu_coords = coords[nW:]

    # attach coords
    matches[["x", "y"]] = pd.DataFrame(win_coords, index=matches.index)
    manu_df = manu_df.reset_index(drop=True)
    manu_df[["x", "y"]] = pd.DataFrame(manu_coords, index=manu_df.index)

    # plotting
    if px is None:
        st.error("plotly is required in this environment (pip install plotly)")
        return

    st.subheader("Embedding scatter")
    fig = px.scatter(
        matches,
        x="x",
        y="y",
        color_discrete_sequence=["lightgray"],
        hover_data={"window_id": True, "best_manual_id": True, "best_manual_score": ":.4f", "transcript_id": True},
        title="Windows (grey) and Manuals (colored)",
    )
    # add nearest-neighbor lines from each window to its best manual (single trace with None separators)
    try:
        if go is not None and manu_df.shape[0] > 0 and nW > 0:
            x_lines = []
            y_lines = []
            for wi in range(nW):
                try:
                    mi = int(best_idx[wi])
                except Exception:
                    continue
                if not (0 <= mi < len(manu_df)):
                    continue
                wx = float(matches.at[wi, "x"])
                wy = float(matches.at[wi, "y"])
                mx = float(manu_df.at[mi, "x"])
                my = float(manu_df.at[mi, "y"])
                # segment from window -> manual, separated by None for multiple segments
                x_lines.extend([wx, mx, None])
                y_lines.extend([wy, my, None])
            if x_lines:
                fig.add_trace(
                    go.Scatter(
                        x=x_lines,
                        y=y_lines,
                        mode="lines",
                        line=dict(color="lightblue", width=1),
                        opacity=0.5,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
    except Exception:
        # drawing lines is optional; ignore failures so rest of plot still renders
        pass
    # style window traces (the ones present before adding manuals): make them small and very transparent
    n_before = len(fig.data)
    for ti in range(0, n_before):
        try:
            fig.data[ti].marker.update(size=5, opacity=0.12)
        except Exception:
            pass

    # optionally add nearest-neighbor lines, filtered by score and capped by max_lines
    if show_lines:
        try:
            if go is not None and manu_df.shape[0] > 0 and nW > 0:
                x_lines = []
                y_lines = []
                drawn = 0
                for wi in range(nW):
                    if max_lines > 0 and drawn >= int(max_lines):
                        break
                    try:
                        mi = int(best_idx[wi])
                        score = float(best_sim[wi])
                    except Exception:
                        continue
                    if not (0 <= mi < len(manu_df)):
                        continue
                    if score < float(min_line_score):
                        continue
                    wx = float(matches.at[wi, "x"])
                    wy = float(matches.at[wi, "y"])
                    mx = float(manu_df.at[mi, "x"])
                    my = float(manu_df.at[mi, "y"])
                    x_lines.extend([wx, mx, None])
                    y_lines.extend([wy, my, None])
                    drawn += 1
                if x_lines:
                    # add lines now (they will appear beneath manuals if manuals are added later)
                    fig.add_trace(
                        go.Scatter(
                            x=x_lines,
                            y=y_lines,
                            mode="lines",
                            line=dict(color="lightblue", width=1),
                            opacity=0.5,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
        except Exception:
            pass

    # plot manuals: either per-manual traces (legend) or single_trace
    try:
        if manual_plot_mode == "by_manual_traces":
            # use px.scatter which creates one trace per category (manual_unit_id)
            manu_scatter = px.scatter(
                manu_df,
                x="x",
                y="y",
                color="manual_unit_id" if "manual_unit_id" in manu_df.columns else None,
                hover_data={"manual_unit_id": True, "text": True},
                labels={"manual_unit_id": "Manual"},
            )
            # add these traces last so manuals render on top
            for trace in manu_scatter.data:
                try:
                    trace.marker.update(size=14, symbol="diamond", line=dict(width=1, color="black"))
                except Exception:
                    pass
                fig.add_trace(trace)
        else:
            # single outlined trace
            if go is not None:
                manu_text = manu_df.get("text") if "text" in manu_df.columns else manu_df.get("manual_unit_text") if "manual_unit_text" in manu_df.columns else manu_df.get("manual_unit_id", None)
                manu_trace = go.Scatter(
                    x=manu_df["x"],
                    y=manu_df["y"],
                    mode="markers",
                    marker=dict(size=14, symbol="diamond", color="red", line=dict(width=1, color="black")),
                    text=manu_text,
                    hoverinfo="text",
                    name="Manuals",
                )
                fig.add_trace(manu_trace)
            else:
                manu_scatter = px.scatter(
                    manu_df,
                    x="x",
                    y="y",
                    hover_data={"manual_unit_id": True, "text": True},
                )
                for trace in manu_scatter.data:
                    fig.add_trace(trace)
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matches table (windows with best manual)")
    st.dataframe(matches[["window_id", "transcript_id", "best_manual_id", "best_manual_score", "topk_manual_ids"]])


if __name__ == "__main__":
    main()
