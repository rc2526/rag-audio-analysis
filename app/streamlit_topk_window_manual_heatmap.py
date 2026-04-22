from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from pathlib import Path
from rag_audio_analysis.settings import get_int, get_str
from rag_audio_analysis.window_manual_heatmap import HeatmapConfig, SelectionMode, build_topk_window_manual_heatmap


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
    if extras:
        st.json(extras)

    sample = diag.get("sample_window")
    if sample:
        with st.expander("Explain Window (sample)", expanded=False):
            st.write(
                {
                    "center_doc_index": sample.get("center_doc_index", ""),
                    "path": sample.get("path", ""),
                    "included_doc_indices": sample.get("included_doc_indices", []),
                    "turn_count": sample.get("turn_count", ""),
                }
            )
            st.text(sample.get("window_text", ""))


def _render_detail_panel(selected: pd.Series) -> None:
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


def main() -> None:
    st.set_page_config(page_title="Top-k Window x Manual Heatmap", layout="wide")

    st.title("Top-k Sparse Heatmap (Transcript Windows x Manual Units)")
    st.caption(get_str("topk_window_manual_heatmap", "caption", "Interactive QA view: manual-unit-driven top-k matches against transcript windows."))

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
        min_score = st.slider(
            "Min score to show",
            min_value=-1.0,
            max_value=1.0,
            value=float(get_str("topk_window_manual_heatmap", "min_score", "-1.0")),
            step=0.01,
        )

        # Filter candidates by selected cycle/session
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

    _render_diagnostics(diag)

    if matches is None or matches.empty:
        st.warning("No matches to display (empty candidates or embeddings).")
        return

    matches = matches.copy()
    matches = matches[matches["score"] >= float(min_score)].reset_index(drop=True)
    if matches.empty:
        st.warning("All matches filtered out by min score.")
        return

    st.subheader("Sparse Heatmap")
    manual_order = pd.unique(matches["manual_unit_id"]).tolist()
    matches["manual_unit_id"] = pd.Categorical(matches["manual_unit_id"], categories=manual_order, ordered=True)

    # Create a short preview for hover to avoid huge hover popups; keep full text in detail panel
    HOVER_PREVIEW_CHARS = 55
    def _preview(text: str, n: int = HOVER_PREVIEW_CHARS) -> str:
        s = str(text or "")
        if len(s) <= n:
            return s
        return s[:n].rsplit(" ", 1)[0] + "..."

    matches["window_text_preview"] = matches["window_text"].apply(lambda t: _preview(t, HOVER_PREVIEW_CHARS))

    # Minimal hover: show only WIN id and score (formatted)
    hover_fields = {
        "window_id": True,
        "score": ":.4f",
    }

    fig = px.scatter(
        matches,
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
        matches[["manual_unit_id", "manual_subsection", "window_id", "center_doc_index", "score", "rank", "cycle_id", "session_num", "turn_count", "path"]],
        width='stretch',
        hide_index=True,
        height=260,
    )
    st.download_button(
        "Download matches CSV",
        data=matches.to_csv(index=False).encode("utf-8"),
        file_name="topk_window_manual_matches.csv",
        mime="text/csv",
    )

    st.subheader("Selection")
    selected_row: Optional[pd.Series] = None
    if clicked is not None and "pointIndex" in clicked:
        idx = int(clicked["pointIndex"])
        if 0 <= idx < len(matches):
            selected_row = matches.iloc[idx]

    if selected_row is None:
        manual_choice = st.selectbox("Manual unit", options=manual_order, index=0)
        subset = matches[matches["manual_unit_id"].astype(str) == str(manual_choice)].sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)
        if subset.empty:
            st.info("No matches for the selected manual unit (after filters).")
            return
        window_choice = st.selectbox("Window", options=subset["window_id"].tolist(), index=0)
        picked = subset[subset["window_id"].astype(str) == str(window_choice)]
        if not picked.empty:
            selected_row = picked.iloc[0]

    if selected_row is not None:
        _render_detail_panel(selected_row)


if __name__ == "__main__":
    main()

