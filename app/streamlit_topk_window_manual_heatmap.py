from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import numpy as np
import altair as alt

from pathlib import Path
from rag_audio_analysis.settings import get_int, get_str
from rag_audio_analysis.window_manual_heatmap import HeatmapConfig, SelectionMode, build_topk_window_manual_heatmap
from rag_audio_analysis.config import TOPIC_CATALOG_CSV


def _maybe_import_plotly():
    try:
        import plotly.express as px  # type: ignore

        return px, None
    except Exception as exc:
        return None, exc


def _maybe_import_plotly_events():
    try:
        from streamlit_plotly_events import plotly_events  # type: ignore

        return plotly_events, None
    except Exception as exc:
        return None, exc


def _cycles_from_settings() -> List[str]:
    cycles_raw = get_str("cycle_analysis", "cycles", "").strip()
    if not cycles_raw:
        return []
    return [c.strip() for c in cycles_raw.split(",") if c.strip()]


def _default_selection_mode() -> SelectionMode:
    mode = get_str("topk_window_manual_heatmap", "selection_mode", "cycle_only").strip()
    valid: List[str] = [
    "cycle_only",
    ]
    return mode if mode in valid else "cycle_only"  # type: ignore[return-value]


def _default_manual_scope() -> str:
    scope = get_str("topk_window_manual_heatmap", "manual_scope", "session").strip()
    return scope if scope in ("session", "global") else "session"


def _render_diagnostics(diag: Dict[str, Any]) -> None:
    st.subheader("Diagnostics")
    cols = st.columns(4)
    cols[0].metric("Centers", int(diag.get("candidate_centers_count", 0)))
    cols[1].metric("Windows", int(diag.get("candidate_windows_count", 0)))
    cols[2].metric("Context Window", int(diag.get("context_window", 0)))
    cols[3].metric("Top-k", int(diag.get("topk", 0)))

    extras = {
        k: v
        for k, v in diag.items()
        if k
        not in {
            "candidate_centers_count",
            "candidate_windows_count",
            "context_window",
            "topk",
            "sample_window",
        }
    }
    # diagnostics JSON dropdown removed per user preference

    # sample window diagnostic removed per UI preference (no inline explain/expander)


def _render_detail_panel(selected: pd.Series) -> None:
    """Render detail panel for a selected Transcript × Manual match row."""
    st.subheader("Detail")
    left, right = st.columns(2)

    with left:
        st.markdown("**Transcript Window**")
        st.write(
            {
                "window_id": selected.get("window_id", ""),
                "center_doc_index": selected.get("center_doc_index", ""),
                "cycle_id": selected.get("cycle_id", ""),
                "session_num": selected.get("session_num", ""),
                "turn_count": selected.get("turn_count", ""),
                "path": selected.get("path", ""),
                "included_doc_indices": selected.get("included_doc_indices", []),
            }
        )
        with st.expander("Window text", expanded=True):
            st.text(str(selected.get("window_text", "") or ""))

    with right:
        st.markdown("**Manual Unit**")
        st.write(
            {
                "manual_unit_id": selected.get("manual_unit_id", ""),
                "manual_week": selected.get("manual_week", ""),
                "manual_section": selected.get("manual_section", ""),
                "manual_subsection": selected.get("manual_subsection", ""),
                "manual_category": selected.get("manual_category", ""),
                "rank": int(selected.get("rank", 0) or 0),
                "score": float(selected.get("score", 0.0) or 0.0),
            }
        )
        with st.expander("Manual unit text", expanded=True):
            st.text(str(selected.get("manual_unit_text", "") or ""))
        matching = str(selected.get("manual_unit_matching_text", "") or "").strip()
        if matching and matching != str(selected.get("manual_unit_text", "") or "").strip():
            with st.expander("Manual matching_text", expanded=False):
                st.text(matching)


def render_transcript_tab(matches: Optional[pd.DataFrame], diag: Optional[Dict[str, Any]], min_score: float) -> Optional[pd.Series]:
    """Render the Transcript × Manual tab. Returns the selected row (or None)."""
    px, px_err = _maybe_import_plotly()
    if px is None:
        st.error("Plotly is required for this view. Install it in the environment running Streamlit (e.g. `pip install plotly`).")
        st.exception(px_err)
        return None

    _render_diagnostics(diag)

    if matches is None:
        st.info("Choose a prebuilt file in the sidebar to load a heatmap. Use the build script to create prebuilt files if none are available.")
        return None

    if matches is None or matches.empty:
        st.warning("No matches to display (empty candidates or embeddings).")
        return None

    matches_tab = matches.copy()
    matches_tab = matches_tab[matches_tab["score"] >= float(min_score)].reset_index(drop=True)
    if matches_tab.empty:
        st.warning("All matches filtered out by min score.")
        return None

    # Option: aggregated Manual x Transcript (session-level) view
    agg_toggle = st.checkbox("Show aggregated Manual × Transcript (session-level)", value=False)
    if agg_toggle:
        st.markdown("**Aggregated Manual × Transcript**")
        # Aggregation selector
        agg_mode = st.selectbox("Aggregation", options=["Max", "Mean", "Median", "Count"], index=0, key="manual_transcript_agg")

        # Display full aggregated matrix (improved readability)
        st.markdown("_Showing aggregated manual units across transcripts — use the min-score slider to filter weak matches._")

        # derive transcript_id if needed
        if "transcript_id" not in matches_tab.columns:
            def _derive_transcript_id(row: pd.Series) -> str:
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

            matches_tab["transcript_id"] = matches_tab.apply(_derive_transcript_id, axis=1)

        df_filtered = matches_tab.copy()
        try:
            if agg_mode == "Count":
                group = df_filtered.groupby(["manual_unit_id", "transcript_id"]).size().reset_index(name="value")
            else:
                if agg_mode == "Max":
                    aggf = "max"
                elif agg_mode == "Mean":
                    aggf = "mean"
                else:
                    aggf = "median"
                group = df_filtered.groupby(["manual_unit_id", "transcript_id"]).score.agg(aggf).reset_index(name="value")

            pivot = group.pivot(index="manual_unit_id", columns="transcript_id", values="value")

            if pivot.shape[0] == 0 or pivot.shape[1] == 0:
                st.info("No aggregated cells to display after grouping.")
                return None

            # Render full heatmap with improved readability
            pivot_view = pivot
            px, px_err = _maybe_import_plotly()
            if px is None:
                st.error("Plotly is required to render the aggregated manual×transcript heatmap.")
            else:
                # transpose so manual units are on the x-axis (bottom)
                pivot_view = pivot.T
                # manual units are columns now (x-axis); transcripts are rows (y-axis)
                labels_x = [str(c) for c in pivot_view.columns.tolist()]
                labels_y = [str(r) for r in pivot_view.index.tolist()]
                z = pivot_view.values
                # dynamic height: 20 px per row (clamped)
                height = int(max(300, min(1200, 20 * len(labels_y))))
                fig = px.imshow(z, x=labels_x, y=labels_y, color_continuous_scale=get_str("topk_window_manual_heatmap", "color_scale", "Viridis"))
                fig.update_layout(height=height)
                # smaller tick fonts for compactness
                fig.update_xaxes(tickangle=90, tickfont=dict(size=8))
                fig.update_yaxes(tickfont=dict(size=9))
                # nicer hovertemplate (manual on x, transcript on y)
                fig.update_traces(hovertemplate="Manual: %{x}<br>Transcript: %{y}<br>Value: %{z:.4f}")
                st.plotly_chart(fig, use_container_width=True)

                # CSV download (full aggregated matrix) - rows=transcript_id, cols=manual_unit_id
                try:
                    out_full = pivot_view.reset_index().rename(columns={"index": "transcript_id"}).to_csv(index=False).encode("utf-8")
                    st.download_button("Download aggregated CSV", data=out_full, file_name="manual_transcript_aggregated.csv", mime="text/csv")
                except Exception:
                    st.warning("Failed to prepare aggregated CSV download.")

            # Drill-down: show windows for a manual unit in a transcript
            st.markdown("**Drill-down: show windows for a manual unit in a transcript**")
            manual_choices = list(pivot_view.columns.astype(str)) if pivot_view.shape[1] > 0 else []
            transcript_choices = list(pivot_view.index.astype(str)) if pivot_view.shape[0] > 0 else []
            manual_choice = st.selectbox("Manual unit", options=manual_choices) if manual_choices else None
            transcript_choice = st.selectbox("Transcript", options=transcript_choices) if transcript_choices else None
            if manual_choice and transcript_choice:
                subset = matches_tab[(matches_tab["manual_unit_id"].astype(str) == str(manual_choice)) & (matches_tab["transcript_id"].astype(str) == str(transcript_choice))]
                if subset.empty:
                    st.info("No window-level matches for that manual unit in the selected transcript (after filters).")
                else:
                    cols_show = [c for c in ["manual_unit_id", "manual_subsection", "window_id", "center_doc_index", "score", "rank", "cycle_id", "session_num", "turn_count", "path"] if c in subset.columns]
                    st.dataframe(subset[cols_show], height=300)
                    st.download_button("Download drill-down CSV", data=subset.to_csv(index=False).encode("utf-8"), file_name="manual_transcript_drilldown.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to compute aggregated manual×transcript matrix: {e}")
        return None

    st.subheader("Sparse Heatmap")
    manual_order = pd.unique(matches_tab["manual_unit_id"]).tolist()
    matches_tab["manual_unit_id"] = pd.Categorical(matches_tab["manual_unit_id"], categories=manual_order, ordered=True)

    # Create a short preview for hover to avoid huge hover popups; keep full text in detail panel
    HOVER_PREVIEW_CHARS = 55
    def _preview(text: str, n: int = HOVER_PREVIEW_CHARS) -> str:
        s = str(text or "")
        if len(s) <= n:
            return s
        return s[:n].rsplit(" ", 1)[0] + "..."

    matches_tab["window_text_preview"] = matches_tab["window_text"].apply(lambda t: _preview(t, HOVER_PREVIEW_CHARS))

    # Minimal hover: show only WIN id and score (formatted)
    hover_fields = {
        "window_id": True,
        "score": ":.4f",
    }

    fig = px.scatter(
        matches_tab,
        x="manual_unit_id",
        y="center_doc_index",
        color="score",
        color_continuous_scale=get_str("topk_window_manual_heatmap", "color_scale", "Viridis"),
        hover_data=hover_fields,
        height=int(get_int("topk_window_manual_heatmap", "plot_height", 650)),
    )
    fig.update_layout(xaxis_title="Manual unit", yaxis_title="Transcript window center (doc_index)")
    fig.update_traces(marker={"size": int(get_int("topk_window_manual_heatmap", "marker_size", 10)), "opacity": 0.85})

    plotly_events, _ = _maybe_import_plotly_events()
    clicked: Optional[Dict[str, Any]] = None
    if plotly_events is not None:
        st.caption("Tip: click a point to inspect details.")
        selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=int(get_int("topk_window_manual_heatmap", "plot_height", 650)))
        if selected_points:
            clicked = selected_points[0]
    else:
        st.plotly_chart(fig, width='stretch')

    st.subheader("Matches Table")
    st.dataframe(
        matches_tab[["manual_unit_id", "manual_subsection", "window_id", "center_doc_index", "score", "rank", "cycle_id", "session_num", "turn_count", "path"]],
        width='stretch',
        hide_index=True,
        height=260,
    )
    st.download_button(
        "Download matches CSV",
        data=matches_tab.to_csv(index=False).encode("utf-8"),
        file_name="topk_window_manual_matches.csv",
        mime="text/csv",
    )

    st.subheader("Selection")
    selected_row: Optional[pd.Series] = None
    if clicked is not None and "pointIndex" in clicked:
        idx = int(clicked["pointIndex"])
        if 0 <= idx < len(matches_tab):
            selected_row = matches_tab.iloc[idx]

    if selected_row is None:
        manual_order = pd.unique(matches_tab["manual_unit_id"]).tolist()
        manual_choice = st.selectbox("Manual unit", options=manual_order, index=0)
        subset = matches_tab[matches_tab["manual_unit_id"].astype(str) == str(manual_choice)].sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)
        if subset.empty:
            st.info("No matches for the selected manual unit (after filters).")
            return None
        window_choice = st.selectbox("Window", options=subset["window_id"].tolist(), index=0)
        picked = subset[subset["window_id"].astype(str) == str(window_choice)]
        if not picked.empty:
            selected_row = picked.iloc[0]

    if selected_row is not None:
        _render_detail_panel(selected_row)
    return selected_row


def render_manual_tab(min_score_manual: float) -> None:
    """Render the Manual × Manual tab independently."""
    px, px_err = _maybe_import_plotly()
    if px is None:
        st.error("Plotly is required for this view. Install it in the environment running Streamlit (e.g. `pip install plotly`).")
        st.exception(px_err)
        return

    st.subheader("Manual × Manual similarity")
    sim_dir = Path("data/derived/manual_unit_similarities")
    sim_df = None
    # offer precomputed files if available
    if sim_dir.exists():
        files = sorted([p for p in sim_dir.iterdir() if p.suffix in {".parquet", ".csv"}])
    else:
        files = []

    if files:
        sel = st.selectbox("Load precomputed similarity file", options=[p.name for p in files])
        chosen = sim_dir / sel
        try:
            if chosen.suffix == ".parquet":
                sim_df = pd.read_parquet(chosen)
            else:
                sim_df = pd.read_csv(chosen, index_col=0)
            st.success(f"Loaded similarity matrix from {chosen.name} (shape={sim_df.shape})")
            # cache the full similarity matrix in session_state for fast interactions
            st.session_state["manual_sim_full"] = sim_df
            st.session_state["manual_sim_loaded_key"] = str(chosen.name)
        except Exception as e:
            st.error(f"Failed to load {chosen.name}: {e}")

    if sim_df is None:
        st.markdown("No precomputed similarity selected or available — compute on demand.")
        compute = st.button("Compute manual×manual similarity (dense)")
        if compute:
            with st.spinner("Computing manual×manual dense cosine similarity — this may take a moment"):
                try:
                    # lazy import of project helper that returns manual unit rows and embeddings
                    from rag_audio_analysis.source_bridge import get_manual_unit_embedding_index

                    manu_df, embeddings = get_manual_unit_embedding_index()
                    em = np.asarray(embeddings).astype(np.float32)
                    # normalize
                    em = em / (np.linalg.norm(em, axis=1, keepdims=True) + 1e-12)
                    sim = em.dot(em.T)
                    ids = [u.get('manual_unit_id', f'MAN_{i:04d}') for i, u in enumerate(manu_df, start=1)]
                    sim_df = pd.DataFrame(sim, index=ids, columns=ids)
                    # ensure output dir exists
                    sim_dir.mkdir(parents=True, exist_ok=True)
                    out = sim_dir / "manual_sim_autosave.parquet"
                    sim_df.to_parquet(out)
                    # cache in session_state
                    st.session_state["manual_sim_full"] = sim_df
                    st.session_state["manual_sim_loaded_key"] = "manual_sim_autosave.parquet"
                    st.success(f"Computed and saved similarity matrix ({sim_df.shape}) to {out}")
                except Exception as ee:
                    st.error(f"Failed to compute manual similarity: {ee}")

    # prefer cached full sim matrix in session_state
    if st.session_state.get("manual_sim_full") is not None:
        sim_df = st.session_state.get("manual_sim_full")

    if sim_df is not None:
        try:
            px2 = px
            labels = sim_df.index.tolist()
            # apply threshold mask from the manual-specific slider
            try:
                threshold = float(min_score_manual)
            except Exception:
                threshold = float(get_str("topk_window_manual_heatmap", "min_score", "-1.0"))

            # mask values below threshold to NaN (blank cells in Plotly)
            sim_masked = sim_df.where(sim_df >= threshold)

            # option to binarize presence/absence
            binarize = st.checkbox("Binarize (show presence/absence)", value=False)
            if binarize:
                plot_df = sim_masked.notnull().astype(int)
                zmin, zmax = 0, 1
            else:
                plot_df = sim_masked
                zmin, zmax = -1, 1

            if plot_df.isnull().all().all():
                st.info("No pairwise similarities exceed the selected threshold.")
            else:
                labels_short = [str(l) for l in labels]
                fig2 = px2.imshow(plot_df.values, x=labels_short, y=labels_short, color_continuous_scale=get_str("topk_window_manual_heatmap", "color_scale", "Viridis"), zmin=zmin, zmax=zmax)
                fig2.update_layout(height=int(get_int("topk_window_manual_heatmap", "manual_sim_plot_height", 700)), xaxis={'tickangle':45})
                st.plotly_chart(fig2, width='stretch')

            # download the full unmasked matrix if user wants it
            st.download_button("Download manual×manual similarity (csv)", data=sim_df.to_csv(index=True).encode("utf-8"), file_name="manual_sim.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to render manual similarity heatmap: {e}")

        # --- New: show a top-N matches table for manual units ---
        try:
            with st.expander("Manual×Manual top-N matches", expanded=True):
                top_n = st.slider("Top N matches per manual unit", min_value=1, max_value=20, value=5, step=1, key="manual_matches_topn")
                # Build a long-form table: for each manual id, list top-N other manual ids and scores
                rows = []
                try:
                    mat = sim_df.copy()
                    ids = mat.index.tolist()

                    # Load manual unit metadata to enrich the table (best-effort)
                    try:
                        from rag_audio_analysis.source_bridge import get_manual_unit_embedding_index

                        manu_rows, _ = get_manual_unit_embedding_index()
                        meta_by_id = {str(u.get('manual_unit_id', '')): u for u in (manu_rows or [])}
                    except Exception:
                        meta_by_id = {}

                    for i, mid in enumerate(ids):
                        # exclude self by setting to -inf so it won't appear in top-N
                        row_vals = mat.iloc[i].copy()
                        if mid in row_vals.index:
                            row_vals[mid] = -float('inf')
                        top_idx = row_vals.nlargest(top_n).index.tolist()
                        top_vals = row_vals.loc[top_idx].tolist()
                        base_meta = meta_by_id.get(str(mid), {})
                        base_text = str(base_meta.get('text') or base_meta.get('matching_text') or '')
                        base_sub = str(base_meta.get('manual_subsection', '') or base_meta.get('manual_section', '') or '')
                        base_cat = str(base_meta.get('manual_category', ''))
                        for rank, (other, score) in enumerate(zip(top_idx, top_vals), start=1):
                            other_meta = meta_by_id.get(str(other), {})
                            other_text = str(other_meta.get('text') or other_meta.get('matching_text') or '')
                            other_sub = str(other_meta.get('manual_subsection', '') or other_meta.get('manual_section', '') or '')
                            other_cat = str(other_meta.get('manual_category', ''))
                            rows.append({
                                "manual_unit_id": mid,
                                "manual_unit_text": base_text,
                                "manual_unit_subsection": base_sub,
                                "manual_unit_category": base_cat,
                                "match_rank": rank,
                                "match_manual_unit_id": other,
                                "match_manual_unit_text": other_text,
                                "match_manual_unit_subsection": other_sub,
                                "match_manual_unit_category": other_cat,
                                "score": float(score),
                            })
                    if rows:
                        tbl = pd.DataFrame(rows)
                        st.dataframe(tbl.reset_index(drop=True), height=400)
                        st.download_button("Download manual matches (csv)", data=tbl.to_csv(index=False).encode("utf-8"), file_name="manual_matches_topn.csv", mime="text/csv")
                    else:
                        st.info("No matches available to display.")
                except Exception as e:
                    st.error(f"Failed to compute manual matches table: {e}")
        except Exception:
            # non-fatal; leave existing UI intact
            pass


def main() -> None:
    st.set_page_config(page_title="Retrieval Heatmap", layout="wide")

    st.title("PMH Group - Retrieval Heatmap")
    st.caption(get_str("topk_window_manual_heatmap", "caption", "Interactive QA view: top-k matches against transcript windows, manual units, and topics."))

    cycles = _cycles_from_settings()
    default_cycle = get_str("topk_window_manual_heatmap", "default_cycle_id", "").strip()
    if not default_cycle and cycles:
        default_cycle = cycles[0]

    with st.sidebar:
        st.header("Controls")

        # Discover available prebuilt files (only data files, omit diagnostics)
        prebuilt_dir = Path("data/derived/topk_window_manual_heatmaps")
        candidates: List[Dict[str, str]] = []
        if prebuilt_dir.exists():
            for p in sorted(prebuilt_dir.iterdir()):
                if not p.is_file():
                    continue
                if p.suffix not in {".parquet", ".csv"}:
                    continue
                stem = p.stem
                if not stem.startswith("cycle_only"):
                    continue
                rest = stem[len("cycle_only_"):] if stem.startswith("cycle_only_") else ""
                cycle = ""
                session = ""
                if rest:
                    if rest.startswith("session"):
                        session = rest.replace("session", "")
                    elif "_session" in rest:
                        cycle, sess = rest.split("_session", 1)
                        session = sess
                    else:
                        cycle = rest
                candidates.append({"stem": stem, "cycle": cycle, "session": session, "path": str(p)})

        available_cycles = sorted({c.get("cycle") for c in candidates if c.get("cycle")})
        available_sessions = sorted({c.get("session") for c in candidates if c.get("session")})

        cycle_choice = st.selectbox("Cycle", options=["All cycles"] + available_cycles, index=0)
        session_choice = st.selectbox("Session", options=["All sessions"] + available_sessions, index=0)

        # Filter candidates by selected cycle/session and expose the prebuilt file selector
        def _matches_filter(c: Dict[str, str]) -> bool:
            if cycle_choice != "All cycles" and c.get("cycle") != cycle_choice:
                return False
            if session_choice != "All sessions" and c.get("session") != session_choice:
                return False
            return True

        filtered = [c for c in candidates if _matches_filter(c)]
        if not filtered:
            st.warning("No prebuilt files found for the selected cycle/session. Build them with the script if needed.")
            chosen_stem = None
        else:
            # build labels and values
            labels = []
            stems = []
            for c in filtered:
                stem = c.get("stem")
                cycle = c.get("cycle")
                session = c.get("session")
                if cycle and session:
                    label = f"{cycle} (session {session})"
                elif cycle:
                    label = cycle
                elif session:
                    label = f"session {session}"
                else:
                    label = stem
                labels.append(label)
                stems.append(stem)
            sel_idx = st.selectbox("Prebuilt file", options=labels, index=0)
            chosen_stem = stems[labels.index(sel_idx)]

        min_score = st.slider(
            "Min score to show",
            min_value=-1.0,
            max_value=1.0,
            value=float(get_str("topk_window_manual_heatmap", "min_score", "-1.0")),
            step=0.01,
        )
    # Manual × Manual tab is always available; threshold slider is independent
        # separate slider for manual×manual so thresholds are independent
        min_score_manual = st.slider(
            "Min score to show (manual×manual)",
            min_value=-1.0,
            max_value=1.0,
            value=float(get_str("topk_window_manual_heatmap", "min_score", "-1.0")),
            step=0.01,
        )

    # (prebuilt selector and candidate filtering already handled above)

    # Session-state keys used: 'matches', 'diag', 'mode_loaded'
    # Auto-load selected prebuilt file into session state
    chosen_stem = locals().get("chosen_stem", None)
    if chosen_stem:
        loaded_key = f"prebuilt:{chosen_stem}"
        if st.session_state.get("mode_loaded") != loaded_key:
            from rag_audio_analysis.window_manual_heatmap import load_matches

            df, diag = load_matches(chosen_stem)
            if df is None or df.empty:
                st.warning(f"No prebuilt file found for '{chosen_stem}'.")
                st.session_state.pop("matches", None)
                st.session_state.pop("diag", None)
                st.session_state.pop("mode_loaded", None)
            else:
                st.session_state["matches"] = df
                st.session_state["diag"] = diag
                st.session_state["mode_loaded"] = loaded_key

    # No Build-fresh support in the UI; users should create prebuilt files with the build script.

    # Use session_state if available
    matches = st.session_state.get("matches")
    diag = st.session_state.get("diag")

    if matches is None:
        st.info("Choose a prebuilt file in the sidebar to load a heatmap. Use the build script to create prebuilt files if none are available.")
        return

    px, px_err = _maybe_import_plotly()
    if px is None:
        st.error("Plotly is required for this view. Install it in the environment running Streamlit (e.g. `pip install plotly`).")
        st.exception(px_err)
        return

    # Config will be constructed when building fresh inside the spinner block.

    # No work inside spinner here; loading/building is done via session_state above.

    tabs = st.tabs(["Transcript × Manual", "Topic × Transcript", "Transcript × Transcript", "Manual × Manual"])

    # --- Tab 0: Transcript × Manual (existing flow) ---
    with tabs[0]:
        # call the formerly nested helper: render_transcript_tab
        # local import of px ensures plotly availability is checked inside the function
        def _render_transcript_local():
            return render_transcript_tab(matches, diag, min_score)

        selected_row = _render_transcript_local()

    # --- Tab 1: Topic × Transcript (prebuilt) ---
    with tabs[1]:
        st.subheader("Topic × Transcript (prebuilt)")
        prebuilt_dir = Path("data/derived/topic_window_heatmaps")
        prebuilt_files = []
        if prebuilt_dir.exists():
            prebuilt_files = sorted([p.name for p in prebuilt_dir.iterdir() if p.is_file() and p.suffix in (".parquet", ".csv")])

        if not prebuilt_files:
            st.info("No prebuilt Topic×Transcript files found. Run `scripts/build_topic_window_heatmaps.py` to create them.")
        else:
            selected = st.multiselect("Prebuilt files (choose one or more)", options=prebuilt_files, default=None, key="topic_prebuilt_multi")
            if selected:
                # load and concatenate selected files
                dfs = []
                for fname in selected:
                    mode = Path(fname).stem
                    try:
                        from rag_audio_analysis.window_manual_heatmap import load_matches

                        df_tmp, _ = load_matches(mode, outdir=prebuilt_dir)
                        if df_tmp is not None and not df_tmp.empty:
                            # tag with a topic id/label if not present
                            if "topic_id" not in df_tmp.columns:
                                # try to derive from filename
                                tid = ""
                                try:
                                    parts = mode.split("topic_only_")
                                    if len(parts) > 1:
                                        tid = parts[1].replace("_full", "")
                                except Exception:
                                    tid = ""
                                df_tmp["topic_id"] = tid
                            dfs.append(df_tmp)
                    except Exception:
                        continue

                if not dfs:
                    st.warning("No valid prebuilt files were loaded.")
                else:
                    df_all = pd.concat(dfs, ignore_index=True)
                    # show topic definitions for the selected topics (best-effort)
                    # also prepare maps so definitions/labels can be embedded into the displayed table
                    label_map: Dict[str, str] = {}
                    def_map: Dict[str, str] = {}
                    try:
                        if TOPIC_CATALOG_CSV and Path(TOPIC_CATALOG_CSV).exists():
                            tdf = pd.read_csv(Path(TOPIC_CATALOG_CSV))
                            for fname in selected:
                                tid = Path(fname).stem.split("topic_only_")[-1].replace("_full", "")
                                rows = tdf[tdf["topic_id"].astype(str) == str(tid)]
                                label = rows.iloc[0].get("topic_label", "") if not rows.empty else tid
                                defn = rows.iloc[0].get("topic_definition", "") if not rows.empty else ""
                                # populate maps for table injection (do not render inline)
                                label_map[str(tid)] = str(label)
                                def_map[str(tid)] = str(defn)
                    except Exception:
                        # leave maps empty on failure
                        label_map = {}
                        def_map = {}

                    # Prepare display & interactive scatter (topic x window center)
                    df_display = df_all.copy()
                    if "window_text" in df_display.columns:
                        df_display["window_excerpt"] = df_display["window_text"].astype(str).apply(lambda t: (t[:280] + "...") if len(t) > 280 else t)

                    # Ensure topic label column exists for nicer x axis and add definitions into the display frame
                    if "topic_label" not in df_display.columns:
                        # prefer label_map if available
                        if label_map:
                            df_display["topic_label"] = df_display["topic_id"].astype(str).map(label_map).fillna(df_display.get("topic_id", ""))
                        else:
                            df_display["topic_label"] = df_display.get("topic_id", df_display.get("manual_unit_id", ""))
                    # attach topic_definition column where available
                    if def_map:
                        df_display["topic_definition"] = df_display["topic_id"].astype(str).map(def_map).fillna("")
                    else:
                        # ensure column exists for consistent table layout
                        if "topic_definition" not in df_display.columns:
                            df_display["topic_definition"] = ""

                    # Derive a transcript identifier for aggregation (document-level)
                    if "transcript_id" not in df_display.columns:
                        # Prefer explicit cycle_id+session_num, else fallback to path or window_id
                        def _derive_transcript_id(row: pd.Series) -> str:
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

                        df_display["transcript_id"] = df_display.apply(_derive_transcript_id, axis=1)

                    # Aggregation controls (min score slider already in sidebar as `min_score`)
                    st.markdown("**Aggregated Topic×Transcript matrix**")
                    agg_mode = st.selectbox("Aggregation", options=["Max", "Mean", "Median", "Count"], index=0, key="topic_transcript_agg")
                    # Use the global min_score slider from the sidebar to filter rows before aggregation
                    df_filtered = df_display[df_display["score"] >= float(min_score)].copy()
                    if df_filtered.empty:
                        st.warning("All matches filtered out by selected min score for the selected topic files.")
                    else:
                        # Group by topic and transcript and compute aggregation
                        try:
                            group = None
                            if agg_mode == "Count":
                                group = df_filtered.groupby(["topic_id", "transcript_id"]).size().reset_index(name="value")
                            else:
                                if agg_mode == "Max":
                                    aggf = "max"
                                elif agg_mode == "Mean":
                                    aggf = "mean"
                                else:
                                    aggf = "median"
                                group = df_filtered.groupby(["topic_id", "transcript_id"]).score.agg(aggf).reset_index(name="value")

                            # Pivot to topic x transcript matrix
                            pivot = group.pivot(index="topic_id", columns="transcript_id", values="value")
                            # Enrich with topic_label if available
                            if "topic_label" in df_display.columns:
                                # map topic_id -> label
                                lbl_map = df_display.set_index("topic_id")["topic_label"].to_dict()
                                pivot = pivot.reset_index()
                                pivot["topic_label"] = pivot["topic_id"].map(lbl_map)
                                pivot = pivot.set_index("topic_label")

                            # Show a small preview and download
                            st.markdown("Aggregated matrix (topics × transcripts). Missing values are shown as blank.")
                            # Render heatmap using Plotly if available
                            px, px_err = _maybe_import_plotly()
                            if px is None:
                                st.error("Plotly is required to render the aggregated heatmap. Install it in the environment.")
                            else:
                                plot_df = pivot.copy()
                                # Convert to numeric matrix for imshow; keep column/row labels
                                if plot_df.shape[0] == 0 or plot_df.shape[1] == 0:
                                    st.info("No aggregated cells to display after grouping.")
                                else:
                                    labels_x = [str(c) for c in plot_df.columns.tolist()]
                                    labels_y = [str(r) for r in plot_df.index.tolist()]
                                    z = plot_df.values
                                    fig = px.imshow(z, x=labels_x, y=labels_y, color_continuous_scale=get_str("topk_window_manual_heatmap", "color_scale", "Viridis"))
                                    fig.update_layout(height=500, xaxis={'tickangle':45})
                                    st.plotly_chart(fig, use_container_width=True)

                                    # CSV download
                                    try:
                                        out_csv = plot_df.reset_index().rename(columns={"index": "topic_label"}).to_csv(index=False).encode("utf-8")
                                        st.download_button("Download aggregated CSV", data=out_csv, file_name="topic_transcript_aggregated.csv", mime="text/csv")
                                    except Exception:
                                        st.warning("Failed to prepare aggregated CSV download.")
                        except Exception as e:
                            st.error(f"Failed to compute aggregated matrix: {e}")

                    # plotly scatter similar to Transcript×Manual view, but use the global min_score filter
                    px, px_err = _maybe_import_plotly()
                    if px is None:
                        st.error("Plotly is required for the interactive topic×transcript scatter. Install it in the environment (pip install plotly).")
                    else:
                        # Use the filtered dataframe so min_score controls what is shown
                        df_scatter = df_filtered.copy()
                        # Order topics consistently
                        topic_order = pd.unique(df_scatter["topic_label"]).tolist()
                        df_scatter["topic_label"] = pd.Categorical(df_scatter["topic_label"], categories=topic_order, ordered=True)

                        hover_fields = {
                            "window_id": True,
                            "score": ":.4f",
                            "cycle_id": True,
                            "session_num": True,
                        }

                        fig = px.scatter(
                            df_scatter,
                            x="topic_label",
                            y="center_doc_index",
                            color="score",
                            color_continuous_scale=get_str("topk_window_manual_heatmap", "color_scale", "Viridis"),
                            hover_data=hover_fields,
                            height=int(get_int("topk_window_manual_heatmap", "plot_height", 650)),
                        )
                        fig.update_layout(xaxis_title="Topic", yaxis_title="Transcript window center (doc_index)")
                        fig.update_traces(marker={"size": int(get_int("topk_window_manual_heatmap", "marker_size", 6)), "opacity": 0.85})
                        st.plotly_chart(fig, use_container_width=True)

                        # Show matches table and CSV download (combined) based on filtered rows
                        cols_to_show = [c for c in ["topic_label", "topic_definition", "rank", "score", "cycle_id", "session_num", "window_id", "window_excerpt"] if c in df_scatter.columns]
                        st.dataframe(df_scatter[cols_to_show], width='stretch', hide_index=True)
                        try:
                            csv_bytes = df_scatter.to_csv(index=False).encode("utf-8")
                            st.download_button("Download combined CSV", data=csv_bytes, file_name=f"topic_combined_selection.csv", mime="text/csv")
                        except Exception:
                            st.warning("Failed to prepare CSV download.")

    # --- Tab 2: Transcript × Transcript (dense similarity) ---
    with tabs[2]:
        st.subheader("Transcript × Transcript (dense)")
        # uses get_transcript_turns -> group by file/path and embed via encode_texts
        from rag_audio_analysis.source_bridge import get_transcript_turns, encode_texts

        sim_metric = st.selectbox("Similarity metric", options=["Cosine", "Pearson"], index=0, key="transcript_sim_metric")
        max_transcripts = int(st.number_input("Max transcripts to include", min_value=20, max_value=2000, value=200, step=10))
        clustering = st.checkbox("Reorder transcripts by mean similarity (simple)", value=False)

        compute_sim = st.button("Compute transcript×transcript similarity")
        if compute_sim:
            with st.spinner("Loading transcripts and computing embeddings..."):
                try:
                    turns = get_transcript_turns()
                    if not turns:
                        st.info("No transcripts found via configured SOURCE_TRANSCRIPTS_GLOB.")
                    else:
                        df_turns = pd.DataFrame(turns)
                        # group turns into transcripts by path
                        if "path" in df_turns.columns:
                            grp = df_turns.groupby("path")["text"].apply(lambda texts: " ".join([str(t) for t in texts if t])).reset_index()
                            grp["transcript_id"] = grp["path"].apply(lambda p: Path(str(p)).stem)
                        else:
                            grp = df_turns.groupby("source")["text"].apply(lambda texts: " ".join([str(t) for t in texts if t])).reset_index()
                            grp["transcript_id"] = grp["source"].astype(str)

                        if len(grp) > max_transcripts:
                            st.warning(f"Found {len(grp)} transcripts; limiting to first {max_transcripts} to avoid heavy computation.")
                            grp = grp.iloc[:max_transcripts]

                        texts = grp["text"].astype(str).tolist()
                        ids = grp["transcript_id"].astype(str).tolist()

                        emb = encode_texts(texts)
                        if emb.size == 0:
                            st.error("Embedding model returned no vectors; check model/config.")
                        else:
                            if sim_metric == "Cosine":
                                sims = emb.dot(emb.T)
                                sims = np.clip(sims, -1.0, 1.0)
                            else:
                                sims = np.corrcoef(emb)

                            sim_df = pd.DataFrame(sims, index=ids, columns=ids)

                            if clustering:
                                order = sim_df.mean(axis=1).sort_values(ascending=False).index.tolist()
                                sim_df = sim_df.loc[order, order]

                            st.markdown(f"**Transcript similarity matrix (metric={sim_metric}) — shape={sim_df.shape}**")
                            px, px_err = _maybe_import_plotly()
                            if px is None:
                                st.error("Plotly is required to render the transcript similarity heatmap.")
                            else:
                                labels = list(sim_df.index)
                                fig = px.imshow(sim_df.values, x=labels, y=labels, color_continuous_scale=get_str("topk_window_manual_heatmap", "color_scale", "Viridis"))
                                fig.update_layout(height=700, xaxis={'tickangle':45})
                                st.plotly_chart(fig, use_container_width=True)

                                try:
                                    out = sim_df.reset_index().rename(columns={"index": "transcript_id"}).to_csv(index=False).encode("utf-8")
                                    st.download_button("Download transcript×transcript CSV", data=out, file_name="transcript_transcript_similarity.csv", mime="text/csv")
                                except Exception:
                                    st.warning("Failed to prepare transcript×transcript CSV download.")
                except Exception as e:
                    st.error(f"Failed to build transcript similarity: {e}")

        st.write("Tip: reduce max transcripts if computation is slow or memory-heavy.")

    # --- Tab 3: Manual × Manual (independent flow) ---
    with tabs[3]:
        # always render the manual sim tab (no sidebar gating)
        render_manual_tab(min_score_manual)

    # Transcript selection and detail are handled inside render_transcript_tab


if __name__ == "__main__":
    main()

