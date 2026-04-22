#!/usr/bin/env python3
import json
from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_audio_analysis.chat_runner import (
    run_chat_query,
    build_chat_prompt,
    call_ollama,
    parse_json_response,
)
from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR, MANUAL_UNITS_CSV, TOPIC_CATALOG_CSV
from rag_audio_analysis.source_bridge import normalize_cycle_frame, get_manual_units_for_session
from rag_audio_analysis.settings import get_float, get_int, get_str


APP_TITLE = get_str("ui", "app_title", "rag-audio-analysis")
APP_CAPTION = get_str(
    "ui",
    "app_caption",
    "Cycle-by-cycle session-topic evidence analysis built from retrieved transcript evidence, manual units, and PI-question summaries.",
)
CYCLE_PREFIX = get_str("ui", "cycle_prefix", "PMHCycle")
UI_EXCERPT_CHARS = get_int("prompting", "ui_excerpt_chars", 280)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_CAPTION)

QUESTION_LABELS = {
    "facilitator_delivery": "Facilitator delivery",
    "facilitator_reference": "Facilitator reference",
    "facilitator_demonstration": "Facilitator demonstration",
    "participant_practice": "Participant practice",
    "participant_child_home": "Participant-child home practice",
}

ANALYSIS_MODE_LABELS = {
    "fidelity": "Fidelity retrieval",
    "session_fidelity": "Cycle-level manual-session fidelity",
    "pi_question": "PI-question retrieval",
}

ADHERENCE_LABELS = {
    "high": "High adherence estimate",
    "moderate": "Moderate adherence estimate",
    "low": "Low adherence estimate",
}

ADJUDICATION_LABELS = {
    "high": "High Generation Grade",
    "moderate": "Moderate Generation Grade",
    "low": "Low Generation Grade",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Avoid Streamlit caching here so external updates to CSVs appear when the
    # app reruns. keep_default_na=False preserves empty strings instead of NaN.
    return pd.read_csv(path, keep_default_na=False)


def reload_flag_set() -> bool:
    # Toggle-able flag stored in session_state to force callers to re-read data.
    return bool(st.session_state.get("reload_data", False))


def get_excerpt(text: str, limit: int = 280) -> str:
    if limit == 280:
        limit = UI_EXCERPT_CHARS
    text = str(text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def human_label(value: str, mapping: dict[str, str]) -> str:
    value = str(value or "")
    return mapping.get(value, value)


def numeric_sort_key(value: str):
    text = str(value or "")
    match = re.search(r"(\d+)", text)
    if match:
        return (0, int(match.group(1)), text.lower())
    return (1, text.lower())


def list_cycle_ids() -> list[str]:
    if not CYCLE_ANALYSIS_DIR.exists():
        return []
    return sorted([path.name for path in CYCLE_ANALYSIS_DIR.iterdir() if path.is_dir() and path.name.startswith(CYCLE_PREFIX)])


def load_cycle_file(cycle_id: str, filename: str) -> pd.DataFrame:
    if not cycle_id:
        return pd.DataFrame()
    return load_csv(CYCLE_ANALYSIS_DIR / cycle_id / filename)


def load_cycle_json(cycle_id: str, filename: str):
    path = CYCLE_ANALYSIS_DIR / cycle_id / filename
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_all_cycle_files(filename: str) -> pd.DataFrame:
    frames = []
    for cycle_id in list_cycle_ids():
        frame = load_cycle_file(cycle_id, filename)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_readable_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()

    if "question_id" in view.columns:
        view["Question"] = view["question_id"].astype(str).map(lambda x: human_label(x, QUESTION_LABELS))
    if "analysis_mode" in view.columns:
        view["Analysis mode"] = view["analysis_mode"].astype(str).map(lambda x: human_label(x, ANALYSIS_MODE_LABELS))
    if "adherence_label" in view.columns:
        view["Adherence"] = view["adherence_label"].astype(str).map(lambda x: human_label(x, ADHERENCE_LABELS))
    if "adjudication_label" in view.columns:
        view["Generation Grade"] = view["adjudication_label"].astype(str).map(lambda x: human_label(x, ADJUDICATION_LABELS))

    rename_map = {
        "cycle_id": "Cycle",
        "manual_session_num": "Session number",
        "manual_session_label": "Session",
        "session_num": "Session number",
        "session_label": "Session",
        "session_topic_ids": "Session topic IDs",
        "session_topic_labels": "Session topic labels",
        "session_summary": "Session summary",
        "topic_id": "Topic ID",
        "topic_label": "Topic label",
        "fidelity_query": "Fidelity query",
        "retrieved_evidence_count": "Retrieved evidence units",
        "expected_manual_unit_count": "Expected manual units",
    "matched_manual_unit_count": "Matched manual units (count)",
    "manual_unit_coverage": "Manual-unit coverage",
        "expected_subsection_count": "Expected subsections",
    "matched_subsection_count": "Matched subsections (count)",
        "subsection_coverage": "Subsection coverage",
        "evidence_density": "Evidence density",
        "adherence_score": "Adherence score",
    "matched_manual_unit_ids": "Matched manual units",
    "matched_subsections": "Matched subsections",
        "sample_session_ids": "Transcript files",
        "adjudication_summary": "Generation Grade summary",
        "adjudication_confidence": "Generation confidence",
        "adjudication_evidence_refs": "Generation evidence refs",
        "adjudication_manual_unit_ids": "Generation manual units",
        "query_text": "Query text",
        "answer_summary": "Automated answer",
    "confidence_explanation": "Confidence explanation",
        "confidence": "Model confidence",
        "evidence_refs": "Evidence refs",
        "manual_unit_ids": "Manual units cited",
        "manual_unit_id_best_match": "Best matching manual unit",
        "manual_unit_match_score": "Manual-unit similarity",
        "source_topic_id": "Source topic ID",
        "source_topic_label": "Source topic label",
        "retrieval_rank": "Retrieval rank",
        "session_id": "Transcript file",
        "speaker": "Speaker label",
        "score_combined": "Combined score",
        "score_doc": "Document score",
        "score_topic": "Topic score",
        "excerpt": "Retrieved excerpt",
        "manual_unit_id": "Manual unit ID",
        "manual_section": "Manual session",
        "manual_subsection": "Manual subsection",
        "topic_match_score": "Topic similarity",
        "manual_text_short": "Manual excerpt",
        "session_num": "Session number",
    }
    # Avoid producing duplicate column names. If the target human-readable name
    # already exists in the frame (for example from a previous transformation),
    # skip that rename so we don't end up with duplicate labels.
    filtered = {k: v for k, v in rename_map.items() if k in view.columns and v not in view.columns}
    display = view.rename(columns=filtered)

    # If renaming created human-friendly columns that collide with original
    # raw column names (or other renamed names), prefer the human-friendly
    # columns and drop the original raw columns to avoid duplicate column
    # labels which raise errors in pandas/Streamlit displays.
    for raw_col, human_name in filtered.items():
        if human_name in display.columns and raw_col in display.columns:
            display = display.drop(columns=[raw_col])

    # As an extra safety, ensure column labels are unique by appending a
    # numeric suffix to any duplicates (should be rare after the above).
    cols = list(display.columns)
    seen = {}
    unique_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            unique_cols.append(c)
        else:
            seen[c] += 1
            unique_cols.append(f"{c}.{seen[c]}")
    display.columns = unique_cols
    return display



# Provide a simple mechanism to force full reloads when the user clicks the
# sidebar "Reload data from disk" button. We don't use @st.cache here to keep
# behavior explicit and simple.
if st.session_state.get("reload_data"):
    # Clear the flag after checking so subsequent interactions don't force a
    # reload unless the button is pressed again.
    st.session_state["reload_data"] = False

manual_units = load_csv(MANUAL_UNITS_CSV)
topic_catalog = load_csv(TOPIC_CATALOG_CSV)
cycle_ids = list_cycle_ids()

all_fidelity = normalize_cycle_frame(load_all_cycle_files("fidelity_summary.csv"))
all_session_fidelity = normalize_cycle_frame(load_all_cycle_files("session_fidelity_summary.csv"))
all_pi_answers = normalize_cycle_frame(load_all_cycle_files("pi_question_answers.csv"))
all_evidence = normalize_cycle_frame(load_all_cycle_files("topic_evidence.csv"))
all_session_evidence = normalize_cycle_frame(load_all_cycle_files("session_fidelity_evidence.csv"))

with st.sidebar:
    st.header("Filters")
    # Reload data from disk: external CSV/JSON updates won't automatically appear
    # in the UI unless the app reruns. This button forces a rerun so files are
    # re-read and the UI shows the latest rows.
    if st.button("Reload data from disk"):
        st.session_state["reload_data"] = True
        # Some Streamlit versions may not expose experimental_rerun; guard use
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            # As a safe fallback, stop execution; the UI will not crash and the
            # reload flag will be processed on the next user interaction.
            st.success("Reload scheduled; please interact with the UI to complete refresh.")
            st.stop()
    cycle_options = ["All cycles"] + cycle_ids
    selected_cycle = st.selectbox("Cycle", cycle_options, index=1 if len(cycle_options) > 1 else 0)

    current_fidelity = all_fidelity if selected_cycle == "All cycles" else load_cycle_file(selected_cycle, "fidelity_summary.csv")
    current_session_fidelity = all_session_fidelity if selected_cycle == "All cycles" else load_cycle_file(selected_cycle, "session_fidelity_summary.csv")
    current_answers = all_pi_answers if selected_cycle == "All cycles" else load_cycle_file(selected_cycle, "pi_question_answers.csv")
    current_evidence = all_evidence if selected_cycle == "All cycles" else load_cycle_file(selected_cycle, "topic_evidence.csv")
    current_session_evidence = all_session_evidence if selected_cycle == "All cycles" else load_cycle_file(selected_cycle, "session_fidelity_evidence.csv")

    session_options = ["All sessions"]
    session_source = current_session_fidelity if not current_session_fidelity.empty else (current_fidelity if not current_fidelity.empty else current_answers)
    session_col = "manual_session_num" if not current_session_fidelity.empty and "manual_session_num" in current_session_fidelity.columns else "session_num"
    if not session_source.empty and session_col in session_source.columns:
        session_options += sorted((x for x in session_source[session_col].astype(str).unique() if x), key=numeric_sort_key)
    selected_session = st.selectbox("Session number", session_options)

    topic_options = ["All topics"]
    topic_source = current_fidelity if not current_fidelity.empty else current_answers
    if not topic_source.empty and "topic_id" in topic_source.columns:
        topic_options += sorted(x for x in topic_source["topic_id"].astype(str).unique() if x)
    selected_topic = st.selectbox("Topic ID", topic_options)


def apply_common_filters(df: pd.DataFrame, include_topic: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()
    if selected_cycle != "All cycles" and "cycle_id" in view.columns:
        view = view[view["cycle_id"].astype(str) == selected_cycle]
    if selected_session != "All sessions":
        # support both session_num and manual_session_num column names
        if "session_num" in view.columns:
            view = view[view["session_num"].astype(str) == selected_session]
        elif "manual_session_num" in view.columns:
            view = view[view["manual_session_num"].astype(str) == selected_session]
    if include_topic and selected_topic != "All topics" and "topic_id" in view.columns:
        view = view[view["topic_id"].astype(str) == selected_topic]
    return view


fidelity_view = apply_common_filters(all_fidelity)
session_fidelity_view = apply_common_filters(all_session_fidelity, include_topic=False)
answers_view = apply_common_filters(all_pi_answers)
topic_evidence_view = apply_common_filters(all_evidence)
session_evidence_view = apply_common_filters(all_session_evidence, include_topic=False)
evidence_frames = [df for df in [topic_evidence_view, session_evidence_view] if not df.empty]
evidence_view = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
manual_view = manual_units.copy()
if selected_session != "All sessions" and "manual_week" in manual_view.columns:
    manual_view = manual_view[manual_view["manual_week"].astype(str) == selected_session]
if selected_topic != "All topics" and "topic_id" in manual_view.columns:
    manual_view = manual_view[manual_view["topic_id"].astype(str) == selected_topic]

def render_key() -> None:
    with st.expander("How to read this app", expanded=False):
        st.markdown("**Fidelity values**")
        st.write("The primary fidelity view is cycle-level alignment to session-structured manual content.")
        st.write("Manual-unit coverage: matched manual units / expected manual units for a manual session within a cycle.")
        st.write("Subsection coverage: matched manual subsections / expected manual subsections for a manual session within a cycle.")
        st.write("Adherence score: `0.6 * manual-unit coverage + 0.4 * subsection coverage`.")
        st.write("Adherence label: `high` if score >= 0.66, `moderate` if >= 0.33, otherwise `low`.")
        st.markdown("**PI-question answers**")
        st.write("These are automated `gpt-oss:120b` summaries constrained to retrieved evidence and matching manual units.")
        st.markdown("**Retrieved evidence**")
        st.write("These are the evidence windows pulled from transcripts for cycle-level fidelity or PI-question mode.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cycle folders", len(cycle_ids))
m2.metric("Manual-session fidelity rows", len(session_fidelity_view.index))
m3.metric("PI-question rows", len(answers_view.index))
m4.metric("Evidence rows", len(evidence_view.index))


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Topic Map", "Fidelity", "PI Questions", "Evidence Browser", "Manual Units", "RAG Chat", "Summaries", "Visuals"]
)

with tab1:
    st.subheader("Topic map by session")
    render_key()

    st.markdown("**Topic map by session**")
    if topic_catalog.empty:
        st.info("No topic catalog found yet.")
    else:
        topic_map = topic_catalog.copy()
        if selected_session != "All sessions" and "session_num" in topic_map.columns:
            topic_map = topic_map[topic_map["session_num"].astype(str) == selected_session]
        st.dataframe(
            add_readable_columns(
                topic_map[
                    [
                        col
                        for col in [
                            "session_num",
                            "session_label",
                            "topic_id",
                            "topic_label",
                            "topic_definition",
                            "primary_skill",
                        ]
                        if col in topic_map.columns
                    ]
                ]
            ),
            width='stretch',
            hide_index=True,
        )

    

with tab2:
    st.subheader("Cycle-level manual-session fidelity")
    render_key()
    st.caption("Transcript evidence is drawn from the full cycle corpus and compared against manual units grouped by manual session.")
    if session_fidelity_view.empty:
        st.info("No fidelity rows are available for the current filters.")
    else:
        fidelity_df = session_fidelity_view.copy()
        fidelity_df["adherence_score"] = pd.to_numeric(fidelity_df["adherence_score"], errors="coerce").fillna(0)
        cols = [
            c
            for c in [
                "cycle_id",
                "manual_session_num",
                "manual_session_label",
                "session_num",
                "retrieved_evidence_count",
                "expected_manual_unit_count",
                "matched_manual_unit_count",
                "manual_unit_coverage",
                "expected_subsection_count",
                "matched_subsection_count",
                "subsection_coverage",
                "evidence_density",
                "adherence_score",
                "adherence_label",
                "adjudication_label",
                "adjudication_confidence",
                "adjudication_summary",
                "topk_mode",
            ]
            if c in fidelity_df.columns
        ]
        sort_cols = [c for c in ["cycle_id", "manual_session_num", "session_num"] if c in fidelity_df.columns]
        if sort_cols:
            display_df = fidelity_df[cols].sort_values(sort_cols)
        else:
            display_df = fidelity_df[cols]

        st.dataframe(
            add_readable_columns(display_df),
            width='stretch',
            hide_index=True,
        )

        row_options = fidelity_df.index.tolist()
        # Choose the best session column name available for display
        session_col_display = "manual_session_num" if "manual_session_num" in fidelity_df.columns else ("session_num" if "session_num" in fidelity_df.columns else None)
        def _format_fidelity_row(idx):
            cycle = fidelity_df.at[idx, 'cycle_id'] if 'cycle_id' in fidelity_df.columns else ''
            sess = fidelity_df.at[idx, session_col_display] if session_col_display else ''
            return f"{cycle} | Session {sess}"

        selected_row_idx = st.selectbox(
            "Select a cycle/manual-session row for detail",
            row_options,
            format_func=_format_fidelity_row,
        )
        row = fidelity_df.loc[selected_row_idx]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Manual-unit coverage", row.get("manual_unit_coverage", ""))
        d2.metric("Subsection coverage", row.get("subsection_coverage", ""))
        d3.metric("Evidence density", row.get("evidence_density", ""))
        d4.metric("Adherence", human_label(str(row.get("adherence_label", "")), ADHERENCE_LABELS))

        generated_label = str(row.get("adjudication_label", "") or "").strip()
        generated_confidence = str(row.get("adjudication_confidence", "") or "").strip()
        if generated_label or generated_confidence:
            g1, g2 = st.columns(2)
            with g1:
                st.metric("Generation Grade", human_label(generated_label, ADJUDICATION_LABELS) if generated_label else "N/A")
            with g2:
                st.metric("Generation confidence", generated_confidence or "N/A")

        st.markdown("**Fidelity query**")
        st.code(str(row.get("fidelity_query", "")))
        st.markdown("**Session summary used for retrieval**")
        st.write(str(row.get("session_summary", "")) or "No session summary is saved yet for this manual session.")

        generated_summary = str(row.get("adjudication_summary", "") or "").strip()
        if generated_summary:
            st.markdown("**Generation Grade summary**")
            st.write(generated_summary)
            st.markdown("**Generation evidence refs**")
            st.write(str(row.get("adjudication_evidence_refs", "")) or "None")
            st.markdown("**Generation manual units cited**")
            st.write(str(row.get("adjudication_manual_unit_ids", "")) or "None")
            with st.expander("Show Generation Grade prompt", expanded=False):
                st.text_area(
                    "Generation Grade prompt",
                    value=str(row.get("adjudication_prompt_text", "")),
                    height=320,
                    disabled=True,
                    label_visibility="collapsed",
                )
            with st.expander("Show Generation Grade raw JSON", expanded=False):
                st.code(str(row.get("adjudication_raw_response", "")), language="json")

        matched_ids = [x for x in str(row.get("matched_manual_unit_ids", "")).split(";") if x]
        expected_units = manual_units.copy()
        # Determine session value from the selected row, supporting both manual_session_num and session_num
        row_session_val = str(row.get("manual_session_num", "") or row.get("session_num", ""))
        if "manual_week" in expected_units.columns and row_session_val:
            expected_units = expected_units[expected_units["manual_week"].astype(str) == row_session_val]
        else:
            expected_units = expected_units.iloc[0:0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Expected manual units**")
            if expected_units.empty:
                st.info("No expected manual units found for this session/topic.")
            else:
                st.dataframe(
                    add_readable_columns(
                        expected_units[
                            [
                                "manual_unit_id",
                                "manual_subsection",
                                "manual_text_short",
                            ]
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )
        with c2:
            st.markdown("**Matched manual units**")
            matched_units = expected_units[expected_units["manual_unit_id"].astype(str).isin(matched_ids)] if not expected_units.empty else pd.DataFrame()
            if matched_units.empty:
                st.info("No matched manual units recorded for this row.")
            else:
                st.dataframe(
                    add_readable_columns(
                        matched_units[
                            [
                                "manual_unit_id",
                                "manual_subsection",
                                "manual_text_short",
                            ]
                        ]
                    ),
                    width='stretch',
                    hide_index=True,
                )

        supporting_evidence = session_evidence_view.copy()
        if not supporting_evidence.empty:
            # choose the correct session column present in supporting_evidence
            session_key = "manual_session_num" if "manual_session_num" in supporting_evidence.columns else ("session_num" if "session_num" in supporting_evidence.columns else None)
            row_session_val = str(row.get("manual_session_num", "") or row.get("session_num", ""))
            # base filter by analysis_mode and cycle_id
            mask = (supporting_evidence["analysis_mode"].astype(str) == "session_fidelity") & (supporting_evidence["cycle_id"].astype(str) == str(row.get("cycle_id", "")))
            if session_key and row_session_val:
                mask = mask & (supporting_evidence[session_key].astype(str) == row_session_val)
            supporting_evidence = supporting_evidence[mask]
        if supporting_evidence.empty:
            st.info("No supporting fidelity evidence rows were found for this cycle/manual-session pair.")
        else:
            sorted_supporting = supporting_evidence.sort_values(
                by=[col for col in ["retrieval_rank"] if col in supporting_evidence.columns] or supporting_evidence.columns.tolist()
            )
            matched_supporting = sorted_supporting[
                sorted_supporting["manual_unit_id_best_match"].astype(str).str.strip().ne("")
            ]
            unmatched_supporting = sorted_supporting[
                sorted_supporting["manual_unit_id_best_match"].astype(str).str.strip().eq("")
            ]

            st.markdown("**Matched evidence snippets**")
            if matched_supporting.empty:
                st.info("No retrieved evidence snippets were matched to a manual unit for this cycle/manual-session row.")
            else:
                for _, evidence_row in matched_supporting.iterrows():
                    full_text = str(evidence_row.get("text", "") or "").strip()
                    excerpt_text = str(evidence_row.get("excerpt", "") or "").strip()
                    visible_excerpt = full_text or excerpt_text
                    st.markdown(
                        f"**Rank {evidence_row.get('retrieval_rank', '')}** | "
                        f"Transcript `{evidence_row.get('session_id', '')}` | "
                        f"Manual `{evidence_row.get('manual_unit_id_best_match', '')}`"
                    )
                    st.write(get_excerpt(visible_excerpt, 700) if visible_excerpt else "No excerpt available.")
                    with st.expander("Show full evidence text", expanded=False):
                        st.write(full_text or excerpt_text or "No full evidence text is available in the current cycle output.")
                    st.divider()

            st.markdown("**Unmatched retrieved snippets**")
            if unmatched_supporting.empty:
                st.info("All retrieved snippets for this row were matched to a manual unit.")
            else:
                for _, evidence_row in unmatched_supporting.iterrows():
                    full_text = str(evidence_row.get("text", "") or "").strip()
                    excerpt_text = str(evidence_row.get("excerpt", "") or "").strip()
                    visible_excerpt = full_text or excerpt_text
                    st.markdown(
                        f"**Rank {evidence_row.get('retrieval_rank', '')}** | "
                        f"Transcript `{evidence_row.get('session_id', '')}` | "
                        "No manual-unit match"
                    )
                    st.write(get_excerpt(visible_excerpt, 700) if visible_excerpt else "No excerpt available.")
                    with st.expander("Show full evidence text", expanded=False):
                        st.write(full_text or excerpt_text or "No full evidence text is available in the current cycle output.")
                    st.divider()

with tab3:
    st.subheader("PI-question answers")
    render_key()
    if answers_view.empty:
        st.info("No PI-question outputs are available for the current filters.")
    else:
        question_options = ["All questions"] + sorted(x for x in answers_view["question_id"].astype(str).unique() if x)
        selected_question = st.selectbox(
            "Question type",
            question_options,
            format_func=lambda x: "All questions" if x == "All questions" else human_label(x, QUESTION_LABELS),
        )
        answer_df = answers_view.copy()
        if selected_question != "All questions":
            answer_df = answer_df[answer_df["question_id"].astype(str) == selected_question]

        if answer_df.empty:
            st.info("No PI-question rows match the current filters.")
        else:
            # show full answer_summary and confidence_explanation in the PI table
            display_cols = [
                "cycle_id",
                "session_num",
                "topic_id",
                "topic_label",
                "question_id",
                "retrieved_evidence_count",
                "confidence",
                "confidence_explanation",
                "answer_summary",
            ]
            available_cols = [c for c in display_cols if c in answer_df.columns]
            st.dataframe(
                add_readable_columns(answer_df[available_cols]),
                width='stretch',
                hide_index=True,
            )

            row_options = answer_df.index.tolist()
            def _format_answer_row(idx):
                cycle = answer_df.at[idx, 'cycle_id'] if 'cycle_id' in answer_df.columns else ''
                sess = answer_df.at[idx, 'session_num'] if 'session_num' in answer_df.columns else ''
                q = human_label(answer_df.at[idx, 'question_id'], QUESTION_LABELS) if 'question_id' in answer_df.columns else ''
                topic = answer_df.at[idx, 'topic_label'] if 'topic_label' in answer_df.columns else ''
                return f"{cycle} | Session {sess} | {q} | {topic}"

            selected_answer_idx = st.selectbox(
                "Select an answer row for detail",
                row_options,
                format_func=_format_answer_row,
            )
            row = answer_df.loc[selected_answer_idx]

            q1, q2, q3 = st.columns(3)
            q1.metric("Retrieved evidence windows", row.get("retrieved_evidence_count", ""))
            q2.metric("Confidence", row.get("confidence", ""))
            q3.metric("Question", human_label(str(row.get("question_id", "")), QUESTION_LABELS))

            st.markdown("**Query text**")
            st.text_area(
                "PI query text",
                value=str(row.get("query_text", "")),
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )
            st.markdown("**Full model prompt**")
            prompt_text = str(row.get("prompt_text", ""))
            if prompt_text:
                st.text_area(
                    "Full model prompt",
                    value=prompt_text,
                    height=320,
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.info("This row was generated before prompt capture was added, so no saved prompt is available yet.")
            st.markdown("**Automated answer**")
            # show full automated answer (with inline evidence refs per prompt)
            st.markdown("**Automated answer (full)**")
            st.write(str(row.get("answer_summary", "")))
            st.markdown("**Confidence explanation**")
            st.write(str(row.get("confidence_explanation", "")) or "None")
            st.markdown("**Evidence refs**")
            st.write(str(row.get("evidence_refs", "")) or "None")
            st.markdown("**Manual units cited**")
            st.write(str(row.get("manual_unit_ids", "")) or "None")

            cycle_json = load_cycle_json(str(row.get("cycle_id", "")), "pi_question_answers.json")
            evidence_payload = None
            for item in cycle_json:
                if (
                    str(item.get("cycle_id", "")) == str(row.get("cycle_id", ""))
                    and str(item.get("session_num", "")) == str(row.get("session_num", ""))
                    and str(item.get("topic_id", "")) == str(row.get("topic_id", ""))
                    and str(item.get("question_id", "")) == str(row.get("question_id", ""))
                ):
                    evidence_payload = item
                    break

            st.markdown("**Evidence refs with retrieved excerpts**")
            refs = [ref for ref in str(row.get("evidence_refs", "")).split(";") if ref]
            evidence_rows = evidence_payload.get("evidence", []) if evidence_payload else []
            if refs and evidence_rows:
                for ref in refs:
                    ref_num = None
                    if ref.startswith("E"):
                        try:
                            ref_num = int(ref[1:])
                        except ValueError:
                            ref_num = None
                    match = evidence_rows[ref_num - 1] if ref_num and 1 <= ref_num <= len(evidence_rows) else None
                    if match:
                        with st.expander(
                            f"{ref} | rank {match.get('retrieval_rank', '')} | manual {match.get('manual_unit_id_best_match', '')}",
                            expanded=False,
                        ):
                            st.write(str(match.get("text", "")))
                    else:
                        st.write(f"{ref}: no matching retrieved excerpt found.")
            elif evidence_rows:
                for idx, match in enumerate(evidence_rows, start=1):
                    with st.expander(
                        f"E{idx} | rank {match.get('retrieval_rank', '')} | manual {match.get('manual_unit_id_best_match', '')}",
                        expanded=False,
                    ):
                        st.write(str(match.get("text", "")))
            else:
                st.info("No saved evidence payload was found for this row.")

            with st.expander("Raw JSON response", expanded=False):
                st.code(str(row.get("raw_response", "")), language="json")

with tab4:
    st.subheader("Retrieved evidence browser")
    render_key()
    if evidence_view.empty:
        st.info("No evidence rows are available for the current filters.")
    else:
        # Coerce to strings and deduplicate before sorting to avoid mixed-type comparison errors
        # Guard access in case some cycle outputs omit the analysis_mode column.
        if "analysis_mode" in evidence_view.columns:
            mode_vals = [str(x) for x in evidence_view["analysis_mode"].fillna("").tolist() if str(x).strip()]
            mode_options = ["All modes"] + sorted(set(mode_vals), key=str)
        else:
            mode_options = ["All modes"]
        selected_mode = st.selectbox(
            "Analysis mode",
            mode_options,
            format_func=lambda x: "All modes" if x == "All modes" else human_label(x, ANALYSIS_MODE_LABELS),
        )
        # Coerce question ids to strings and deduplicate; sorting by string representation is safe
        if "question_id" in evidence_view.columns:
            question_vals = [str(x) for x in evidence_view["question_id"].fillna("").tolist() if str(x).strip()]
            question_options = ["All questions"] + sorted(set(question_vals), key=str)
        else:
            question_options = ["All questions"]

        selected_question = st.selectbox(
            "Question filter",
            question_options,
            format_func=lambda x: "All questions" if x == "All questions" else human_label(x, QUESTION_LABELS),
            key="evidence_question_filter",
        )

        browser_df = evidence_view.copy()
        # Only filter by mode if the column exists
        if selected_mode != "All modes" and "analysis_mode" in browser_df.columns:
            browser_df = browser_df[browser_df["analysis_mode"].astype(str) == selected_mode]
        if selected_question != "All questions" and "question_id" in browser_df.columns:
            browser_df = browser_df[browser_df["question_id"].astype(str) == selected_question]

        if browser_df.empty:
            st.info("No evidence rows match the current filters.")
        else:
            display_df = browser_df.copy()
            # Safely prepare excerpt only if present
            if "excerpt" in display_df.columns:
                display_df["excerpt"] = display_df["excerpt"].astype(str).apply(get_excerpt)

            # Only select columns that exist to avoid KeyError when some fields are missing
            desired_cols = [
                "cycle_id",
                "session_num",
                "manual_session_num",
                "topic_id",
                "topic_label",
                "analysis_mode",
                "question_id",
                "retrieval_rank",
                "manual_unit_id_best_match",
                "manual_unit_match_score",
                "excerpt",
            ]
            cols_to_show = [c for c in desired_cols if c in display_df.columns]

            # If producers used `manual_session_num` instead of `session_num`, expose it under the
            # canonical `session_num` column for consistent display and readable-column mapping.
            if "session_num" not in cols_to_show and "manual_session_num" in cols_to_show:
                display_df = display_df.copy()
                display_df["session_num"] = display_df["manual_session_num"].astype(str)
                cols_to_show = ["session_num" if c == "manual_session_num" else c for c in cols_to_show]

            #missing = [c for c in desired_cols if c not in cols_to_show]
            #if missing:
                #st.warning(f"Some expected columns are missing and will be hidden: {missing}")

            st.dataframe(
                add_readable_columns(display_df[cols_to_show]),
                width='stretch',
                hide_index=True,
            )

            row_options = browser_df.index.tolist()
            def _format_evidence_row(idx):
                cycle = browser_df.at[idx, 'cycle_id'] if 'cycle_id' in browser_df.columns else ''
                sess = browser_df.at[idx, 'session_num'] if 'session_num' in browser_df.columns else (browser_df.at[idx, 'manual_session_num'] if 'manual_session_num' in browser_df.columns else '')
                rank = browser_df.at[idx, 'retrieval_rank'] if 'retrieval_rank' in browser_df.columns else ''
                topic = browser_df.at[idx, 'topic_label'] if 'topic_label' in browser_df.columns else ''
                return f"{cycle} | Session {sess} | rank {rank} | {topic}"

            selected_evidence_idx = st.selectbox(
                "Select an evidence row for detail",
                row_options,
                format_func=_format_evidence_row,
            )
            row = browser_df.loc[selected_evidence_idx]
            e1, e2, e3 = st.columns(3)
            e1.metric("Mode", human_label(str(row.get("analysis_mode", "")), ANALYSIS_MODE_LABELS))
            e2.metric("Question", human_label(str(row.get("question_id", "")), QUESTION_LABELS) or "N/A")
            e3.metric("Best manual unit", str(row.get("manual_unit_id_best_match", "")) or "None")
            st.write(f"Manual-unit similarity: {row.get('manual_unit_match_score', '') or 'N/A'}")
            st.markdown("**Retrieved excerpt**")
            st.write(str(row.get("excerpt", "")))
            st.markdown("**Query text**")
            st.code(str(row.get("query_text", "")))

with tab5:
    st.subheader("Manual units")
    render_key()
    if manual_units.empty:
        st.info("No manual-unit file found yet.")
    else:
        manual_df = manual_view.copy()

        session_choices = ["All manual sessions"] + sorted(
            (x for x in manual_units["manual_section"].astype(str).unique() if x),
            key=numeric_sort_key,
        )
        selected_manual_session = st.selectbox("Manual session", session_choices)
        if selected_manual_session != "All manual sessions":
            manual_df = manual_df[manual_df["manual_section"].astype(str) == selected_manual_session]

        subsection_choices = ["All subsections"] + sorted(x for x in manual_df["manual_subsection"].astype(str).unique() if x)
        selected_manual_subsection = st.selectbox("Manual subsection", subsection_choices)
        if selected_manual_subsection != "All subsections":
            manual_df = manual_df[manual_df["manual_subsection"].astype(str) == selected_manual_subsection]

        if manual_df.empty:
            st.info("No manual units match the current filters.")
        else:
            sort_cols = [col for col in ["manual_week", "manual_chunk_index", "manual_unit_id"] if col in manual_df.columns]
            if sort_cols:
                manual_df = manual_df.copy()
                for col in [c for c in ["manual_week", "manual_chunk_index"] if c in manual_df.columns]:
                    manual_df[col] = pd.to_numeric(manual_df[col], errors="coerce")
                manual_df = manual_df.sort_values(sort_cols, kind="stable")
            st.dataframe(
                add_readable_columns(
                    manual_df[
                        [
                            "manual_unit_id",
                            "manual_section",
                            "manual_subsection",
                            "manual_text_short",
                        ]
                    ]
                ),
                width='stretch',
                hide_index=True,
            )

            row_options = manual_df.index.tolist()
            selected_manual_idx = st.selectbox(
                "Select a manual unit",
                row_options,
                format_func=lambda idx: f"{manual_df.at[idx, 'manual_unit_id']} | {manual_df.at[idx, 'manual_section']} | {manual_df.at[idx, 'manual_subsection']}",
            )
            row = manual_df.loc[selected_manual_idx]
            m1, m2, m3 = st.columns(3)
            m1.metric("Manual unit", str(row.get("manual_unit_id", "")))
            m2.metric("Session", str(row.get("manual_section", "")))
            m3.metric("Subsection", str(row.get("manual_subsection", "")))
            st.markdown("**Full manual content**")
            st.write(str(row.get("manual_text", "")))

with tab6:
    st.subheader("RAG-enabled chat")
    render_key()
    st.write("Ask a free-text question against the existing index.")
    st.info(
        "Recommended defaults:\n"
        "- Transcript-only exploratory questions: leave `Also search manual passages` off, keep `Generate answer with gpt-oss:120b` on, and start with `Document weight = 1.0`, `Topic weight = 0.0`.\n"
        "- Manual + transcript fidelity questions: turn `Also search manual passages` on, keep `Generate answer with gpt-oss:120b` on, and start with `Document weight = 0.5`, `Topic weight = 0.5`."
    )
    with st.expander("What the chat options mean", expanded=True):
        st.write("Search transcripts only: search just the de-identified transcript evidence. This is the best default if you want to know what people said in session.")
        st.write("Also search manual passages: allow retrieval to include manual sections as well as transcript evidence. Turn this on if you want protocol language or session instructions to appear in the evidence set.")
        st.write("Generate answer with gpt-oss:120b: after retrieval, send the retrieved evidence to the model and ask it to write a grounded answer. If this is off, the tab will only show retrieved evidence and will not call the model.")
        st.write("Document weight vs topic weight: document weight favors semantic similarity to your exact question. Topic weight favors the topic-embedding structure in the original index.")
        st.write("The first chat run can take longer because the embedding model has to warm up.")

    default_cycle = "" if selected_cycle == "All cycles" else selected_cycle
    chat_defaults = {
        "topk": get_int("chat", "topk", 8),
        "weight_doc": get_float("chat", "weight_doc", 1.0),
        "weight_topic": get_float("chat", "weight_topic", 0.0),
        "include_manual": get_str("chat", "include_manual", "false").lower() == "true",
        "answer_with_model": get_str("chat", "answer_with_model", "true").lower() == "true",
    }

    with st.form("rag_chat_form"):
        question = st.text_area(
            "Question",
            value="",
            height=120,
            placeholder="Example: How do participants talk about stress affecting eating in Cycle 1?",
        )
        cycle_dropdown_options = ["All cycles", "PMHCycle1", "PMHCycle2", "PMHCycle3", "PMHCycle4", "PMHCycle5"]
        default_cycle_option = default_cycle if default_cycle in cycle_dropdown_options else "All cycles"
        chat_cycle_option = st.selectbox(
            "Cycle filter",
            cycle_dropdown_options,
            index=cycle_dropdown_options.index(default_cycle_option),
            help="Choose one cycle to narrow retrieval, or leave it on All cycles.",
        )

        with st.expander("Advanced controls", expanded=False):
            c1, c2 = st.columns(2)
            chat_topk = c1.number_input("Top-k", min_value=1, max_value=50, value=chat_defaults["topk"], step=1)
            chat_include_manual = c2.checkbox(
                "Also search manual passages",
                value=chat_defaults["include_manual"],
                help="When on, retrieval can return manual sections alongside transcript evidence. When off, results are transcript-only.",
            )
            # Minimal manual-only option (no fallback): retrieve and then summarize using only manual chunks
            chat_manual_only = c2.checkbox(
                "Manual-only retrieval (no fallback)",
                value=False,
                help="When on, the app will retrieve evidence but synthesize the answer only from manual chunks (no transcript fallback).",
            )

            c3, c4, c5 = st.columns(3)
            chat_weight_doc = c3.number_input("Document weight", min_value=0.0, max_value=2.0, value=float(chat_defaults["weight_doc"]), step=0.1)
            chat_weight_topic = c4.number_input("Topic weight", min_value=0.0, max_value=2.0, value=float(chat_defaults["weight_topic"]), step=0.1)
            chat_answer_with_model = c5.checkbox(
                "Generate answer with gpt-oss:120b",
                value=chat_defaults["answer_with_model"],
                help="When on, the app sends the retrieved evidence to the model for a grounded answer. When off, the app only shows retrieved evidence.",
            )
            # Prompt variant selector: default | PI question | Fidelity adjudication
            c6, _ = st.columns([1, 2])
            prompt_variant = c6.selectbox(
                "Prompt variant",
                ["default", "pi_question", "fidelity"],
                index=0,
                help="Choose the prompt style sent to the model: default QA, PI question style, or fidelity adjudication style.",
            )

        # Show the configured default Ollama model and allow a per-run override.
        default_ollama_model = get_str("ollama", "default_model", "gpt-oss:120b")
        ollama_model_input = st.text_input(
            "Ollama model (optional)",
            value="",
            help="Override the default Ollama model for this run; leave blank to use settings.ini default.",
        )
        if not ollama_model_input:
            st.caption(f"Default Ollama model: {default_ollama_model}")
        else:
            st.caption(f"Using override model: {ollama_model_input}")

        preview_prompt = st.form_submit_button("Preview prompt")
        submit_chat = st.form_submit_button("Run RAG chat")

        if preview_prompt:
            if not question.strip():
                st.warning("Enter a question first to preview the prompt.")
            else:
                try:
                    with st.spinner("Building prompt preview (no model call)..."):
                        # Prepare a manual_units payload. For fidelity prompts prefer the
                        # canonical selector used by run_cycle_analysis so the UI prompt
                        # matches batch runs: get_manual_units_for_session(session, topic).
                        if prompt_variant == "fidelity":
                            # Map UI selections to selector args (empty string means no filter)
                            session_arg = "" if selected_session == "All sessions" else str(selected_session)
                            topic_arg = "" if selected_topic == "All topics" else str(selected_topic)
                            try:
                                manual_units_payload = get_manual_units_for_session(session_arg, topic_id=topic_arg) or []
                            except Exception:
                                # Fallback to the prior DataFrame-based filtering if selector fails
                                manual_units_payload = manual_units.copy()
                                if selected_session != "All sessions" and "manual_week" in manual_units_payload.columns:
                                    manual_units_payload = manual_units_payload[manual_units_payload["manual_week"].astype(str) == selected_session]
                                if selected_topic != "All topics" and "topic_id" in manual_units_payload.columns:
                                    manual_units_payload = manual_units_payload[manual_units_payload["topic_id"].astype(str) == selected_topic]
                        else:
                            manual_units_payload = manual_units.copy()

                        # Normalize to list-of-dicts for downstream callers
                        if isinstance(manual_units_payload, pd.DataFrame):
                            manual_units_payload = manual_units_payload.to_dict(orient="records")
                        else:
                            manual_units_payload = list(manual_units_payload or [])

                        # Perform retrieval to get evidence, but don't call the model.
                        # For the preview path we guard against failures in the retrieval
                        # pipeline (or absence of indexes) by falling back to a
                        # lightweight manual-only "retrieved" list built from the
                        # selected manual units. This ensures the prompt builder can
                        # always run in the UI without external dependencies.
                        retrieved = []
                        try:
                            retrieval_payload = run_chat_query(
                                question=question.strip(),
                                cycle_id="" if chat_cycle_option == "All cycles" else chat_cycle_option,
                                topk=int(chat_topk),
                                weight_doc=float(chat_weight_doc),
                                weight_topic=float(chat_weight_topic),
                                include_manual=chat_include_manual or chat_manual_only,
                                manual_only=chat_manual_only,
                                answer_with_model=False,
                                ollama_model=ollama_model_input,
                                manual_units=manual_units_payload,
                                prompt_variant=prompt_variant,
                            )

                            retrieved = retrieval_payload.get("evidence", []) or []
                            if chat_manual_only:
                                retrieved = [r for r in retrieved if str(r.get("source_type", "")).lower() == "manual"]
                        except Exception:
                            # Fallback: construct a minimal retrieved list from the
                            # manual units payload so build_chat_prompt has something
                            # realistic to format. Limit the number of manual units
                            # included in the preview to avoid huge prompts.
                            try:
                                preview_limit = get_int("prompting", "ui_preview_manual_limit", 12)
                            except Exception:
                                preview_limit = 12

                            if manual_units_payload:
                                # manual_units_payload may be a DataFrame or list; ensure list
                                mus = manual_units_payload if isinstance(manual_units_payload, list) else list(manual_units_payload)
                                retrieved = []
                                for i, mu in enumerate(mus[:preview_limit]):
                                    # Try common manual unit text fields, fall back to str(mu)
                                    text = mu.get("manual_text_short") or mu.get("manual_text") or mu.get("manual_excerpt") or ""
                                    retrieved.append(
                                        {
                                            "rank": i + 1,
                                            "source_type": "manual",
                                            "cycle_id": mu.get("cycle_id", "") if isinstance(mu, dict) else "",
                                            "session_id": mu.get("manual_week", mu.get("manual_section", "")) if isinstance(mu, dict) else "",
                                            "manual_unit_id_best_match": mu.get("manual_unit_id") if isinstance(mu, dict) else "",
                                            "manual_unit_match_score": "",
                                            "text": (text[:UI_EXCERPT_CHARS] if isinstance(text, str) else str(text)),
                                        }
                                    )

                        # Build the prompt locally using the chosen variant
                        prompt_preview = build_chat_prompt(
                            question.strip(),
                            retrieved,
                            variant=prompt_variant,
                        )
                        st.session_state["rag_chat_prompt_preview"] = prompt_preview
                        st.session_state["rag_chat_preview_evidence"] = retrieved
                except Exception as exc:
                    st.error("Prompt preview failed.")
                    st.code(str(exc))
                    st.session_state["rag_chat_error"] = str(exc)

        if submit_chat:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                # If user chose manual-only, perform retrieval (include_manual=True) without model answering,
                # filter to manual evidence rows client-side, then synthesize an answer using only those manual rows.
                if chat_manual_only:
                    try:
                        with st.spinner("Running manual-only retrieval..."):
                            # Prefer canonical selector for fidelity prompts
                            if prompt_variant == "fidelity":
                                session_arg = "" if selected_session == "All sessions" else str(selected_session)
                                topic_arg = "" if selected_topic == "All topics" else str(selected_topic)
                                try:
                                    manual_units_payload = get_manual_units_for_session(session_arg, topic_id=topic_arg) or []
                                except Exception:
                                    manual_units_payload = manual_units.copy()
                                    if selected_session != "All sessions" and "manual_week" in manual_units_payload.columns:
                                        manual_units_payload = manual_units_payload[manual_units_payload["manual_week"].astype(str) == selected_session]
                                    if selected_topic != "All topics" and "topic_id" in manual_units_payload.columns:
                                        manual_units_payload = manual_units_payload[manual_units_payload["topic_id"].astype(str) == selected_topic]
                            else:
                                manual_units_payload = manual_units.copy()

                            if isinstance(manual_units_payload, pd.DataFrame):
                                manual_units_payload = manual_units_payload.to_dict(orient="records")
                            else:
                                manual_units_payload = list(manual_units_payload or [])

                            retrieval_payload = run_chat_query(
                                question=question.strip(),
                                cycle_id="" if chat_cycle_option == "All cycles" else chat_cycle_option,
                                topk=int(chat_topk),
                                weight_doc=float(chat_weight_doc),
                                weight_topic=float(chat_weight_topic),
                                include_manual=True,
                                manual_only=chat_manual_only,
                                answer_with_model=False,
                                manual_units=manual_units_payload,
                                prompt_variant=prompt_variant,
                            )

                        # Filter retrieved evidence to only manual chunks (require explicit source_type == 'manual').
                        # We no longer accept inferred matches on transcript rows here to avoid transcript fallback.
                        retrieved = retrieval_payload.get("evidence", []) or []
                        manual_evidence = [r for r in retrieved if str(r.get("source_type", "")).lower() == "manual"]

                        # Always call the model on the manual evidence (even if the filtered
                        # manual_evidence list is empty). This keeps manual-only mode
                        # deterministic: the model is always asked to synthesize from the
                        # manual chunks selected by the filter.
                        try:
                            st.info(f"Manual evidence rows found: {len(manual_evidence)}. Calling model...")
                            prompt_text = build_chat_prompt(
                                question.strip(),
                                manual_evidence,
                                variant=prompt_variant,
                            )
                            raw = ""
                            try:
                                # Prefer per-run override when provided
                                chosen_model = (ollama_model_input.strip() or get_str("ollama", "default_model", "gpt-oss:120b"))
                                raw = call_ollama(prompt_text, chosen_model)
                                # record which model we used for this saved payload
                                answer_payload = parse_json_response(raw)
                                # ensure payload will include model below
                                answer_payload = parse_json_response(raw)
                            except Exception as exc:
                                st.error("Model call failed for manual-only summarization.")
                                st.code(str(exc))
                                st.session_state["rag_chat_error"] = str(exc)
                                answer_payload = {}

                            payload = {
                                "question": question,
                                "cycle_id": "" if chat_cycle_option == "All cycles" else chat_cycle_option,
                                "topk": int(chat_topk),
                                "weight_doc": float(chat_weight_doc),
                                "weight_topic": float(chat_weight_topic),
                                "include_manual": True,
                                "answer_with_model": True,
                                "prompt_variant": prompt_variant,
                                "prompt_text": prompt_text,
                                "raw_model_output": raw,
                                "fallback_to_transcript": False,
                                "answer": answer_payload,
                                "evidence": manual_evidence,
                            }
                            # persist which Ollama model was used for this run
                            payload["ollama_model"] = chosen_model
                            st.session_state["rag_chat_payload"] = payload
                            # clear any prior error if call succeeded (or we handled it)
                            st.session_state["rag_chat_error"] = st.session_state.get("rag_chat_error", "")
                        except Exception as exc:
                            st.error("Manual-only retrieval failed.")
                            st.code(str(exc))
                            st.session_state["rag_chat_error"] = str(exc)
                    except Exception as exc:
                        st.error("Manual-only retrieval failed.")
                        st.code(str(exc))
                        st.session_state["rag_chat_error"] = str(exc)
                else:
                    try:
                        with st.spinner("Running retrieval and preparing results..."):
                            # Prefer canonical selector for fidelity prompts
                            if prompt_variant == "fidelity":
                                session_arg = "" if selected_session == "All sessions" else str(selected_session)
                                topic_arg = "" if selected_topic == "All topics" else str(selected_topic)
                                try:
                                    manual_units_payload = get_manual_units_for_session(session_arg, topic_id=topic_arg) or []
                                except Exception:
                                    manual_units_payload = manual_units.copy()
                                    if selected_session != "All sessions" and "manual_week" in manual_units_payload.columns:
                                        manual_units_payload = manual_units_payload[manual_units_payload["manual_week"].astype(str) == selected_session]
                                    if selected_topic != "All topics" and "topic_id" in manual_units_payload.columns:
                                        manual_units_payload = manual_units_payload[manual_units_payload["topic_id"].astype(str) == selected_topic]
                            else:
                                manual_units_payload = manual_units.copy()

                            if isinstance(manual_units_payload, pd.DataFrame):
                                manual_units_payload = manual_units_payload.to_dict(orient="records")
                            else:
                                manual_units_payload = list(manual_units_payload or [])

                            payload = run_chat_query(
                                question=question.strip(),
                                cycle_id="" if chat_cycle_option == "All cycles" else chat_cycle_option,
                                topk=int(chat_topk),
                                weight_doc=float(chat_weight_doc),
                                weight_topic=float(chat_weight_topic),
                                include_manual=chat_include_manual,
                                manual_only=chat_manual_only,
                                answer_with_model=chat_answer_with_model,
                                ollama_model=ollama_model_input,
                                prompt_variant=prompt_variant,
                                manual_units=manual_units_payload,
                            )
                            # attach chosen model into the payload for reproducibility
                            chosen = ollama_model_input.strip() or get_str("ollama", "default_model", "gpt-oss:120b")
                            payload["ollama_model"] = chosen
                            st.session_state["rag_chat_payload"] = payload
                            st.session_state["rag_chat_error"] = ""
                    except Exception as exc:
                        st.error("RAG chat query failed.")
                        st.code(str(exc))
                        st.session_state["rag_chat_error"] = str(exc)

    payload = st.session_state.get("rag_chat_payload")
    chat_error = st.session_state.get("rag_chat_error", "")
    if chat_error:
        with st.expander("Last chat error", expanded=False):
            st.code(chat_error)
    if payload:
        with st.expander("Query used", expanded=False):
            st.code(str(payload.get("question", "")))
        mode_col1, mode_col2, mode_col3 = st.columns(3)
        mode_col1.metric("Search scope", "Transcript + manual" if payload.get("include_manual") else "Transcript only")
        mode_col2.metric("Output mode", "Grounded model answer" if payload.get("answer_with_model") else "Evidence only")
        mode_col3.metric("Retrieved rows", len(payload.get("evidence", []) or []))

        answer = payload.get("answer", {}) or {}
        if answer:
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", str(answer.get("confidence", "")) or "N/A")
            evidence_refs = answer.get("evidence_refs", [])
            m2.metric("Evidence refs", ", ".join(evidence_refs) if evidence_refs else "None")
            manual_ids = answer.get("manual_unit_ids", [])
            m3.metric("Manual units", ", ".join(manual_ids) if manual_ids else "None")

            # Show the session number the model returned (or the fallback value)
            session_number = answer.get("session_number")
            if session_number is None:
                session_display = "N/A"
            else:
                try:
                    # normalize numeric sessions and sentinel -1 as unknown
                    sn = int(session_number)
                    session_display = "unknown" if sn == -1 else str(sn)
                except Exception:
                    session_display = str(session_number)

            s1, s2, s3 = st.columns(3)
            s1.metric("Session (model)", session_display)
            # If the answer payload includes an explicit inferred flag or explanation, show it
            inferred_flag = answer.get("session_inferred") or answer.get("session_number_inferred")
            if inferred_flag:
                s1.caption("Session inferred from evidence (fallback)")
            else:
                # fallback: surface any confidence explanation text if present
                conf_expl = answer.get("confidence_explanation") or answer.get("explanation")
                if conf_expl:
                    s1.caption(str(conf_expl))

            # Show session_explanation (if provided) in a collapsible area
            sess_expl = answer.get("session_explanation")
            if sess_expl:
                with st.expander("How session was chosen", expanded=False):
                    st.write(str(sess_expl))

            st.markdown("**Grounded answer**")
            # Backwards-compat fallback: some saved runs store the model text under
            # `adjudication_summary` instead of `answer_summary`. Prefer the
            # canonical `answer_summary` but fall back to `adjudication_summary`.
            display_text = answer.get("answer_summary") or answer.get("adjudication_summary")
            st.write(str(display_text) or "No answer returned.")

            with st.expander("Full model prompt", expanded=False):
                st.code(str(payload.get("prompt_text", "")))
        else:
            st.info("This run used evidence-only mode. The app searched the index and returned retrieved evidence, but it did not send that evidence to gpt-oss:120b for an answer.")

        evidence_rows = payload.get("evidence", []) or []
        st.markdown("**Retrieved evidence**")

        # Enrich evidence rows with a human-friendly manual-session value when possible.
        if evidence_rows:
            try:
                # determine which column in manual_units corresponds to a session label
                manual_session_col = None
                for c in ["manual_section", "manual_week", "manual_session", "manual_session_label"]:
                    if c in manual_units.columns:
                        manual_session_col = c
                        break

                if manual_session_col and "manual_unit_id_best_match" in pd.DataFrame(evidence_rows).columns:
                    mapping = {str(k): str(v) for k, v in zip(manual_units.get("manual_unit_id", []), manual_units[manual_session_col])}
                    for r in evidence_rows:
                        mid = str(r.get("manual_unit_id_best_match", "") or "")
                        r["manual_session"] = mapping.get(mid, "")
                else:
                    for r in evidence_rows:
                        r["manual_session"] = ""
            except Exception:
                # Fall back to silently not providing manual_session if anything goes wrong
                for r in evidence_rows:
                    r.setdefault("manual_session", "")

            # Provide download/save controls for the payload
            try:
                json_bytes = json.dumps(payload, indent=2).encode("utf-8")
            except Exception:
                json_bytes = str(payload).encode("utf-8")

            cdl, cds = st.columns([1, 1])
            with cdl:
                st.download_button("Download result JSON", data=json_bytes, file_name=f"rag_chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
            with cds:
                if st.button("Save result to server"):
                    out_dir = CYCLE_ANALYSIS_DIR / "rag_chat_outputs"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"rag_chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                    path = out_dir / fname
                    try:
                        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                        st.success(f"Saved to {path}")
                    except Exception as exc:
                        st.error(f"Failed to save: {exc}")

            evidence_df = pd.DataFrame(evidence_rows)
            if "text" in evidence_df.columns:
                evidence_df["text"] = evidence_df["text"].astype(str).apply(get_excerpt)

            # Include manual_session in the displayed columns when present
            display_cols = [
                col
                for col in [
                    "rank",
                    "source_type",
                    "cycle_id",
                    "session_id",
                    "manual_session",
                    "score_combined",
                    "score_doc",
                    "score_topic",
                    "manual_unit_id_best_match",
                    "manual_unit_match_score",
                    "text",
                ]
                if col in evidence_df.columns
            ]

            st.dataframe(add_readable_columns(evidence_df[display_cols]), width='stretch', hide_index=True)

            for idx, row in enumerate(evidence_rows, start=1):
                title = f"E{idx} | {row.get('source_type','')} | rank {row.get('rank','')}"
                with st.expander(title, expanded=False):
                    st.write(f"Cycle: {row.get('cycle_id','')}")
                    st.write(f"Session: {row.get('session_id','')}")
                    st.write(f"Manual session: {row.get('manual_session','')}")
                    st.write(f"Manual unit: {row.get('manual_unit_id_best_match','')}")
                    st.write(f"Manual-unit similarity: {row.get('manual_unit_match_score','')}")
                    st.write(str(row.get("text", "")))
        else:
            st.info("No evidence rows were returned.")

    with tab7:
        st.subheader("Aggregated summary files")
        st.caption("Browse, preview, and download CSV summary tables generated by `scripts/aggregate_cycle_outputs.py`.")

        summary_dir = CYCLE_ANALYSIS_DIR / "summary"
        if not summary_dir.exists():
            st.info("No summary folder found yet. Run the aggregator first.")
        else:
            # Grouped summary folders for clearer UX
            fidelity_dir = summary_dir / "fidelity"
            pi_dir = summary_dir / "pi_questions"

            fidelity_files = sorted([p.name for p in fidelity_dir.iterdir() if p.is_file()]) if fidelity_dir.exists() else []
            pi_files = sorted([p.name for p in pi_dir.iterdir() if p.is_file()]) if pi_dir.exists() else []
            # fallback: root-level files (for backward compatibility)
            root_files = sorted([p.name for p in summary_dir.iterdir() if p.is_file()])

            if not fidelity_files and not pi_files and not root_files:
                st.info("No aggregator-generated CSV summary files found in the summary folder.")
            else:
                group_choice = st.radio("Summary group", ["Fidelity", "PI questions", "All (flat)"], index=0)
                if group_choice == "Fidelity":
                    files = fidelity_files or root_files
                elif group_choice == "PI questions":
                    files = pi_files or root_files
                else:
                    files = root_files

                chosen = st.selectbox("Choose summary file", files)
                path = summary_dir / chosen
                df = load_csv(path)
                if df.empty:
                    st.warning("Selected file is empty.")
                else:
                    display = add_readable_columns(df)
                    st.markdown(f"**Preview: {chosen}**")
                    st.dataframe(display, width='stretch')

                    # Altair-powered plotting: let user pick X and Y axes and plot type.
                    cols_all = [c for c in df.columns]
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    x_choice = st.selectbox("X axis (optional)", ["None"] + cols_all, index=0)
                    y_choice = st.selectbox("Y axis (numeric)", ["None"] + numeric_cols, index=0)
                    plot_type = st.selectbox("Plot type", ["auto", "bar", "line", "scatter", "box"])
                    agg_func = st.selectbox("Aggregation (when X is categorical)", ["mean", "median", "sum", "count"], index=0)

                    def aggregate(df_local, xcol, ycol, agg):
                        if agg == "count":
                            return df_local.groupby(xcol).size().reset_index(name="count")
                        if agg == "sum":
                            return df_local.groupby(xcol)[ycol].sum().reset_index()
                        if agg == "median":
                            return df_local.groupby(xcol)[ycol].median().reset_index()
                        return df_local.groupby(xcol)[ycol].mean().reset_index()

                    if y_choice and y_choice != "None":
                        # prepare DataFrame view for Altair
                        vis_df = df.copy()
                        # Drop rows with missing Y
                        vis_df = vis_df[pd.notna(vis_df[y_choice])]

                        if x_choice and x_choice != "None":
                            # If X is non-numeric, aggregate and use bar/box by default
                            if not pd.api.types.is_numeric_dtype(vis_df[x_choice]):
                                agg_df = aggregate(vis_df, x_choice, y_choice, agg_func)
                                if plot_type in ["auto", "bar"]:
                                    chart = alt.Chart(agg_df).mark_bar().encode(x=alt.X(x_choice, type="nominal"), y=alt.Y(y_choice if agg_func != "count" else "count", type="quantitative"), tooltip=list(agg_df.columns))
                                elif plot_type == "box":
                                    chart = alt.Chart(vis_df).mark_boxplot().encode(x=alt.X(x_choice, type="nominal"), y=alt.Y(y_choice, type="quantitative"), tooltip=[x_choice, y_choice])
                                elif plot_type == "scatter":
                                    chart = alt.Chart(agg_df).mark_point().encode(x=alt.X(x_choice, type="nominal"), y=alt.Y(y_choice if agg_func != "count" else "count", type="quantitative"), tooltip=list(agg_df.columns))
                                else:
                                    chart = alt.Chart(agg_df).mark_bar().encode(x=alt.X(x_choice, type="nominal"), y=alt.Y(y_choice if agg_func != "count" else "count", type="quantitative"), tooltip=list(agg_df.columns))
                            else:
                                # X is numeric: plot relationship between numeric X and numeric Y
                                if plot_type in ["auto", "scatter"]:
                                    chart = alt.Chart(vis_df).mark_point().encode(x=alt.X(x_choice, type="quantitative"), y=alt.Y(y_choice, type="quantitative"), tooltip=[x_choice, y_choice])
                                elif plot_type == "line":
                                    chart = alt.Chart(vis_df).mark_line().encode(x=alt.X(x_choice, type="quantitative"), y=alt.Y(y_choice, type="quantitative"), tooltip=[x_choice, y_choice])
                                else:
                                    # fallback to scatter
                                    chart = alt.Chart(vis_df).mark_point().encode(x=alt.X(x_choice, type="quantitative"), y=alt.Y(y_choice, type="quantitative"), tooltip=[x_choice, y_choice])
                        else:
                            # No X chosen: if cycle_id present, aggregate by cycle; else plot Y over index
                            if "cycle_id" in df.columns:
                                agg_df = aggregate(vis_df, "cycle_id", y_choice, agg_func)
                                chart = alt.Chart(agg_df).mark_bar().encode(x=alt.X("cycle_id", type="nominal"), y=alt.Y(y_choice if agg_func != "count" else "count", type="quantitative"), tooltip=list(agg_df.columns))
                            else:
                                chart = alt.Chart(vis_df.reset_index()).mark_line().encode(x=alt.X("index", type="quantitative"), y=alt.Y(y_choice, type="quantitative"), tooltip=[y_choice, "index"]) 

                        st.altair_chart(chart, width='stretch')

                    with open(path, "rb") as fh:
                        st.download_button(label="Download CSV", data=fh, file_name=chosen)
                    # Notebook visuals have been removed; keep aggregated CSV preview and Altair plotting above.
                    # Removed session fidelity images per user request.

with tab8:
    st.subheader("Visuals")
    st.caption("Manual Adherence and LLM Question Summary metrics")

    # Fidelity cards
    st.markdown("### Manual Adherence (Fidelity) summary")

    st.markdown("**Mean Adherence Score across cycles**")
    if session_fidelity_view.empty:
        st.info("No fidelity outputs match the current filters yet.")
    else:
        fidelity_chart = session_fidelity_view.copy()
        fidelity_chart["adherence_score"] = pd.to_numeric(fidelity_chart["adherence_score"], errors="coerce").fillna(0)
        st.bar_chart(fidelity_chart.groupby("manual_session_label")["adherence_score"].mean().sort_values(ascending=False))
        overview_cols = [
            "cycle_id",
            "manual_session_num",
            "manual_session_label",
            "retrieved_evidence_count",
            "manual_unit_coverage",
            "subsection_coverage",
            "evidence_density",
            "adherence_score",
            "adherence_label",
            "adjudication_label",
            "adjudication_confidence",
        ]
        
    
    st.markdown("**Adjudication Percent (LLM Generation Grades) across cycles**")
    # Load adjudication summary (preferred grouped folder, fallback to root)
    summary_root = CYCLE_ANALYSIS_DIR / "summary"
    adjud_group = summary_root / "fidelity" / "summary_adjudication_by_cycle_session.csv"
    adjud_root = summary_root / "summary_adjudication_by_cycle_session.csv"
    adjud_path = adjud_group if adjud_group.exists() else adjud_root
    df_adjud = load_csv(adjud_path) if adjud_path.exists() else pd.DataFrame()

    if df_adjud.empty:
        st.info("No adjudication summary found; run the aggregator to produce summary_adjudication_by_cycle_session.csv")
    else:
        # Expect columns: cycle_id, rows, pct_adjud_high, pct_adjud_moderate, pct_adjud_low
        pct_cols = [c for c in ["pct_adjud_high", "pct_adjud_moderate", "pct_adjud_low"] if c in df_adjud.columns]
        if pct_cols and "cycle_id" in df_adjud.columns:
            df_plot = df_adjud.copy()
            # Ensure numeric
            for c in pct_cols:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce").fillna(0)
            dfm = pd.melt(df_plot, id_vars=["cycle_id"], value_vars=pct_cols, var_name="confidence", value_name="pct")
            dfm["confidence"] = dfm["confidence"].map({"pct_adjud_high": "High", "pct_adjud_moderate": "Moderate", "pct_adjud_low": "Low"}).fillna(dfm["confidence"])
            try:
                chart = alt.Chart(dfm).mark_bar().encode(
                    x=alt.X("cycle_id:N", title="Cycle"),
                    y=alt.Y("pct:Q", title="Percent"),
                    color=alt.Color("confidence:N", title="Adjudication"),
                    order=alt.Order("confidence", sort="descending"),
                    tooltip=["cycle_id", "confidence", "pct"],
                ).properties(height=320)
                st.altair_chart(chart, width='stretch')
            except Exception:
                st.info("Unable to render adjudication stacked chart for the current data.")
        else:
            st.info("Adjudication percent columns not found in summary file.")
    
    st.markdown("**Adjudication Confidence Percent (LLM Generation Confidence) across current filters**")
    # Load adjudication summary (preferred grouped folder, fallback to root)
    summary_root = CYCLE_ANALYSIS_DIR / "summary"
    adjud_conf_group = summary_root / "fidelity" / "summary_adjudication_confidence_by_cycle_session.csv"
    adjud_conf_root = summary_root / "summary_adjudication_confidence_by_cycle_session.csv"
    adjud_conf_path = adjud_conf_group if adjud_conf_group.exists() else adjud_conf_root
    df_adjud_conf = load_csv(adjud_conf_path) if adjud_conf_path.exists() else pd.DataFrame()

    if df_adjud_conf.empty:
        st.info("No adjudication summary found; run the aggregator to produce summary_adjudication_confidence_by_cycle_session.csv")
    else:
        # Expect columns: cycle_id, rows, pct_conf_high, pct_conf_medium, pct_conf_low
        pct_cols = [c for c in ["pct_conf_high", "pct_conf_medium", "pct_conf_low"] if c in df_adjud_conf.columns]
        if pct_cols and "cycle_id" in df_adjud_conf.columns:
            df_plot = df_adjud_conf.copy()
            # Ensure numeric
            for c in pct_cols:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce").fillna(0)
            dfm = pd.melt(df_plot, id_vars=["cycle_id"], value_vars=pct_cols, var_name="confidence", value_name="pct")
            dfm["confidence"] = dfm["confidence"].map({"pct_conf_high": "High", "pct_conf_medium": "Medium", "pct_conf_low": "Low"}).fillna(dfm["confidence"])
            try:
                chart = alt.Chart(dfm).mark_bar().encode(
                    x=alt.X("cycle_id:N", title="Cycle"),
                    y=alt.Y("pct:Q", title="Percent"),
                    color=alt.Color("confidence:N", title="Adjudication confidence"),
                    order=alt.Order("confidence", sort="descending"),
                    tooltip=["cycle_id", "confidence", "pct"],
                ).properties(height=320)
                st.altair_chart(chart, width='stretch')
            except Exception:
                st.info("Unable to render adjudication-confidence stacked chart for the current data.")
        else:
            st.info("Adjudication confidence percent columns not found in summary file.")

    # Session-level adjudication (across cycles) by manual session
    st.markdown("**Adjudication Percent by manual session (Generation Grades)**")
    adjud_manual_group = summary_root / "fidelity" / "summary_adjudication_by_manual_session.csv"
    adjud_manual_root = summary_root / "summary_adjudication_by_manual_session.csv"
    adjud_manual_path = adjud_manual_group if adjud_manual_group.exists() else adjud_manual_root
    df_adjud_manual = load_csv(adjud_manual_path) if adjud_manual_path.exists() else pd.DataFrame()

    if df_adjud_manual.empty:
        st.info("No session-level adjudication summary found; run the aggregator to produce summary_adjudication_by_manual_session.csv")
    else:
        # Expect columns: manual_session_num, manual_session_label, rows, pct_adjud_high, pct_adjud_moderate, pct_adjud_low
        name_col = "manual_session_label" if "manual_session_label" in df_adjud_manual.columns else ("manual_session_num" if "manual_session_num" in df_adjud_manual.columns else None)
        pct_cols = [c for c in ["pct_adjud_high", "pct_adjud_moderate", "pct_adjud_low"] if c in df_adjud_manual.columns]
        if pct_cols and name_col:
            df_plot = df_adjud_manual.copy()
            for c in pct_cols:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce").fillna(0)
            try:
                dfm = pd.melt(df_plot, id_vars=[name_col], value_vars=pct_cols, var_name="adjudication", value_name="pct")
                dfm["adjudication"] = dfm["adjudication"].map({"pct_adjud_high": "High", "pct_adjud_moderate": "Moderate", "pct_adjud_low": "Low"}).fillna(dfm["adjudication"])
                chart = alt.Chart(dfm).mark_bar().encode(
                    x=alt.X(f"{name_col}:N", title="Manual session"),
                    y=alt.Y("pct:Q", title="Percent"),
                    color=alt.Color("adjudication:N", title="Generation Grade"),
                    order=alt.Order("adjudication", sort="descending"),
                    tooltip=[name_col, "adjudication", "pct"],
                ).properties(height=360)
                st.altair_chart(chart, width='stretch')
            except Exception:
                st.info("Unable to render session-level adjudication chart for the current data.")
        else:
            st.info("Session-level adjudication columns not found in summary file.")

    st.markdown("**Adjudication Confidence Percent by manual session**")
    adjud_conf_manual_group = summary_root / "fidelity" / "summary_adjudication_confidence_by_manual_session.csv"
    adjud_conf_manual_root = summary_root / "summary_adjudication_confidence_by_manual_session.csv"
    adjud_conf_manual_path = adjud_conf_manual_group if adjud_conf_manual_group.exists() else adjud_conf_manual_root
    df_adjud_conf_manual = load_csv(adjud_conf_manual_path) if adjud_conf_manual_path.exists() else pd.DataFrame()

    if df_adjud_conf_manual.empty:
        st.info("No session-level adjudication confidence summary found; run the aggregator to produce summary_adjudication_confidence_by_manual_session.csv")
    else:
        name_col = "manual_session_label" if "manual_session_label" in df_adjud_conf_manual.columns else ("manual_session_num" if "manual_session_num" in df_adjud_conf_manual.columns else None)
        pct_cols = [c for c in ["pct_conf_high", "pct_conf_medium", "pct_conf_low"] if c in df_adjud_conf_manual.columns]
        if pct_cols and name_col:
            df_plot = df_adjud_conf_manual.copy()
            for c in pct_cols:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce").fillna(0)
            try:
                dfm = pd.melt(df_plot, id_vars=[name_col], value_vars=pct_cols, var_name="confidence", value_name="pct")
                dfm["confidence"] = dfm["confidence"].map({"pct_conf_high": "High", "pct_conf_medium": "Medium", "pct_conf_low": "Low"}).fillna(dfm["confidence"])
                chart = alt.Chart(dfm).mark_bar().encode(
                    x=alt.X(f"{name_col}:N", title="Manual session"),
                    y=alt.Y("pct:Q", title="Percent"),
                    color=alt.Color("confidence:N", title="Adjudication confidence"),
                    order=alt.Order("confidence", sort="descending"),
                    tooltip=[name_col, "confidence", "pct"],
                ).properties(height=360)
                st.altair_chart(chart, width='stretch')
            except Exception:
                st.info("Unable to render session-level adjudication-confidence chart for the current data.")
        else:
            st.info("Session-level adjudication confidence columns not found in summary file.")

    # Paths to grouped summary folders (preferred)
    summary_root = CYCLE_ANALYSIS_DIR / "summary"
    fidelity_dir = summary_root / "fidelity"
    pi_dir = summary_root / "pi_questions"

    # Load candidate files with fallback
    def load_preferring(dirpath: Path, name: str):
        p = dirpath / name
        if p.exists():
            return load_csv(p)
        # fallback to root summary
        pr = summary_root / name
        return load_csv(pr) if pr.exists() else pd.DataFrame()

    # Fidelity data
    df_session_fid_cycle = load_preferring(fidelity_dir, "summary_session_fidelity_by_cycle.csv")
    df_session_fid_manual = load_preferring(fidelity_dir, "summary_session_fidelity_by_manual_session.csv")

    # PI data
    df_pi_by_cycle = load_preferring(pi_dir, "summary_pi_questions_by_cycle.csv")
    df_pi_by_topic = load_preferring(pi_dir, "summary_pi_questions_by_topic.csv")
    df_pi_by_question = load_preferring(pi_dir, "summary_pi_questions_by_type.csv")
    # by-cycle + topic breakdown
    df_pi_by_cycle_topic = load_preferring(pi_dir, "summary_pi_questions_by_cycle_and_topic.csv")
    # by-cycle + question/type breakdown
    df_pi_by_cycle_question = load_preferring(pi_dir, "summary_pi_questions_by_cycle_and_type.csv")

    
    # PI cards
    st.markdown("### LLM Summaries of PI Questions")

    # 1) PI by topic (top N topics by pi_rows) - stacked confidence
    st.markdown("#### By topic")
    if df_pi_by_topic.empty:
        st.info("No PI-by-topic summary found; run the aggregator to produce summary_pi_questions_by_topic.csv")
    else:
        df = df_pi_by_topic.copy()
        if "pi_rows" in df.columns:
            df["pi_rows"] = pd.to_numeric(df["pi_rows"], errors="coerce").fillna(0)
        pct_cols = [c for c in ["pct_confidence_high", "pct_confidence_medium", "pct_confidence_low"] if c in df.columns]
        top_n = min(10, len(df))
        top_n = st.slider("Topics to show", min_value=1, max_value=46, value=10)
        df_top = df.sort_values("pi_rows", ascending=False).head(top_n)
        if pct_cols and "topic_label" in df_top.columns:
            dfm = pd.melt(df_top, id_vars=["topic_label"], value_vars=pct_cols, var_name="confidence", value_name="pct")
            dfm["confidence"] = dfm["confidence"].map({"pct_confidence_high": "High", "pct_confidence_medium": "Medium", "pct_confidence_low": "Low"}).fillna(dfm["confidence"])
            chart = alt.Chart(dfm).mark_bar().encode(
                x=alt.X("topic_label:N", sort=alt.EncodingSortField(field="pct", op="sum", order="descending"), title="Topic"),
                y=alt.Y("pct:Q", title="Percent"),
                color=alt.Color("confidence:N", title="Confidence"),
                tooltip=["topic_label", "confidence", "pct"],
            ).properties(height=360)
            st.altair_chart(chart, width='stretch')
        else:
            # Fallback: show pi_rows counts
            if "topic_label" in df_top.columns and "pi_rows" in df_top.columns:
                chart = alt.Chart(df_top).mark_bar().encode(x=alt.X("topic_label:N", sort=alt.EncodingSortField(field="pi_rows", op="sum", order="descending")), y=alt.Y("pi_rows:Q", title="PI rows"), tooltip=["topic_label", "pi_rows"]).properties(height=320)
                st.altair_chart(chart, width='stretch')
            else:
                st.info("Insufficient columns in PI-by-topic summary for charts.")

    # 2) PI by cycle - stacked confidence
    st.markdown("#### By cycle")
    if df_pi_by_cycle.empty:
        st.info("No PI-by-cycle summary found; run the aggregator to produce summary_pi_questions_by_cycle.csv")
    else:
        df = df_pi_by_cycle.copy()
        pct_cols = [c for c in ["pct_confidence_high", "pct_confidence_medium", "pct_confidence_low"] if c in df.columns]
        if pct_cols and "cycle_id" in df.columns:
            for c in pct_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            dfm = pd.melt(df, id_vars=["cycle_id"], value_vars=pct_cols, var_name="confidence", value_name="pct")
            dfm["confidence"] = dfm["confidence"].map({"pct_confidence_high": "High", "pct_confidence_medium": "Medium", "pct_confidence_low": "Low"}).fillna(dfm["confidence"])
            chart = alt.Chart(dfm).mark_bar().encode(x=alt.X("cycle_id:N", title="Cycle"), y=alt.Y("pct:Q", title="Percent"), color=alt.Color("confidence:N", title="Confidence"), tooltip=["cycle_id", "confidence", "pct"]).properties(height=280)
            st.altair_chart(chart, width='stretch')
        else:
            st.info("Insufficient columns in PI-by-cycle summary for confidence chart; showing counts if available.")
            if "question_rows" in df.columns and "cycle_id" in df.columns:
                df["question_rows"] = pd.to_numeric(df["question_rows"], errors="coerce").fillna(0)
                chart = alt.Chart(df).mark_bar().encode(x=alt.X("cycle_id:N", title="Cycle"), y=alt.Y("question_rows:Q", title="Question rows"), tooltip=["cycle_id", "question_rows"]).properties(height=280)
                st.altair_chart(chart, width='stretch')

    # 3) PI by question/type - stacked confidence
    st.markdown("#### By question type")
    if df_pi_by_question.empty:
        st.info("No PI-by-type summary found; run the aggregator to produce summary_pi_questions_by_type.csv")
    else:
        df = df_pi_by_question.copy()
        pct_cols = [c for c in ["pct_confidence_high", "pct_confidence_medium", "pct_confidence_low"] if c in df.columns]
        if pct_cols and "question_label" in df.columns:
            for c in pct_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            dfm = pd.melt(df, id_vars=["question_label"], value_vars=pct_cols, var_name="confidence", value_name="pct")
            dfm["confidence"] = dfm["confidence"].map({"pct_confidence_high": "High", "pct_confidence_medium": "Medium", "pct_confidence_low": "Low"}).fillna(dfm["confidence"])
            chart = alt.Chart(dfm).mark_bar().encode(x=alt.X("question_label:N", sort=alt.EncodingSortField(field="pct", op="sum", order="descending"), title="Question"), y=alt.Y("pct:Q", title="Percent"), color=alt.Color("confidence:N", title="Confidence"), tooltip=["question_label", "confidence", "pct"]).properties(height=300)
            st.altair_chart(chart, width='stretch')
        else:
            if "question_rows" in df.columns and "question_label" in df.columns:
                df["question_rows"] = pd.to_numeric(df["question_rows"], errors="coerce").fillna(0)
                chart = alt.Chart(df).mark_bar().encode(x=alt.X("question_label:N", sort=alt.EncodingSortField(field="question_rows", op="sum", order="descending"), title="Question"), y=alt.Y("question_rows:Q", title="Question rows"), tooltip=["question_label", "question_rows"]).properties(height=300)
                st.altair_chart(chart, width='stretch')
            else:
                st.info("Insufficient columns in PI-by-type summary for charts.")

    # 4) PI by cycle + topic: choose a topic and view cycle-level confidence distribution
    st.markdown("#### By cycle and topic")
    if df_pi_by_cycle_topic.empty:
        st.info("No PI-by-cycle-and-topic summary found; run the aggregator to produce summary_pi_questions_by_cycle_and_topic.csv")
    else:
        df = df_pi_by_cycle_topic.copy()
        if "topic_label" in df.columns:
            topics = sorted(df["topic_label"].astype(str).unique())
            sel_topic = st.selectbox("Select topic", topics)
            sel_df = df[df["topic_label"].astype(str) == sel_topic]
            pct_cols = [c for c in ["pct_confidence_high", "pct_confidence_medium", "pct_confidence_low"] if c in sel_df.columns]
            if pct_cols and "cycle_id" in sel_df.columns:
                for c in pct_cols:
                    sel_df[c] = pd.to_numeric(sel_df[c], errors="coerce").fillna(0)
                dfm = pd.melt(sel_df, id_vars=["cycle_id"], value_vars=pct_cols, var_name="confidence", value_name="pct")
                dfm["confidence"] = dfm["confidence"].map({"pct_confidence_high": "High", "pct_confidence_medium": "Medium", "pct_confidence_low": "Low"}).fillna(dfm["confidence"])
                chart = alt.Chart(dfm).mark_bar().encode(x=alt.X("cycle_id:N", title="Cycle"), y=alt.Y("pct:Q", title="Percent"), color=alt.Color("confidence:N", title="Confidence"), tooltip=["cycle_id", "confidence", "pct"]).properties(height=300)
                st.altair_chart(chart, width='stretch')
            else:
                if "pi_rows" in sel_df.columns and "cycle_id" in sel_df.columns:
                    sel_df["pi_rows"] = pd.to_numeric(sel_df["pi_rows"], errors="coerce").fillna(0)
                    chart = alt.Chart(sel_df).mark_bar().encode(x=alt.X("cycle_id:N", title="Cycle"), y=alt.Y("pi_rows:Q", title="PI rows"), tooltip=["cycle_id", "pi_rows"]).properties(height=300)
                    st.altair_chart(chart, width='stretch')
                else:
                    st.info("Insufficient columns for cycle+topic chart for the selected topic.")
        else:
            st.info("Topic label column missing in cycle+topic summary.")

    # 5) PI by cycle + type: choose a question type and view percent-with-evidence (preferred), then confidence or counts
    st.markdown("#### By cycle and question type")
    if df_pi_by_cycle_question.empty:
        st.info("No PI-by-cycle-and-type summary found; run the aggregator to produce summary_pi_questions_by_cycle_and_type.csv")
    else:
        df = df_pi_by_cycle_question.copy()
        if "question_label" in df.columns:
            questions = sorted(df["question_label"].astype(str).unique())
            sel_q = st.selectbox("Select question type", questions)
            sel_df = df[df["question_label"].astype(str) == sel_q]
            # Prefer percent of rows with evidence refs if available
            if "pct_rows_with_evidence_refs" in sel_df.columns and "cycle_id" in sel_df.columns:
                sel_df["pct_rows_with_evidence_refs"] = pd.to_numeric(sel_df["pct_rows_with_evidence_refs"], errors="coerce").fillna(0)
                chart = alt.Chart(sel_df).mark_bar().encode(
                    x=alt.X("cycle_id:N", title="Cycle"),
                    y=alt.Y("pct_rows_with_evidence_refs:Q", title="% rows with evidence refs"),
                    tooltip=["cycle_id", "pct_rows_with_evidence_refs"],
                ).properties(height=300)
                st.altair_chart(chart, width='stretch')
            else:
                # fallback: prefer confidence cols if present
                pct_cols = [c for c in ["pct_confidence_high", "pct_confidence_medium", "pct_confidence_low"] if c in sel_df.columns]
                if pct_cols and "cycle_id" in sel_df.columns:
                    for c in pct_cols:
                        sel_df[c] = pd.to_numeric(sel_df[c], errors="coerce").fillna(0)
                    dfm = pd.melt(sel_df, id_vars=["cycle_id"], value_vars=pct_cols, var_name="confidence", value_name="pct")
                    dfm["confidence"] = dfm["confidence"].map({"pct_confidence_high": "High", "pct_confidence_medium": "Medium", "pct_confidence_low": "Low"}).fillna(dfm["confidence"])
                    chart = alt.Chart(dfm).mark_bar().encode(x=alt.X("cycle_id:N", title="Cycle"), y=alt.Y("pct:Q", title="Percent"), color=alt.Color("confidence:N", title="Confidence"), tooltip=["cycle_id", "confidence", "pct"]).properties(height=300)
                    st.altair_chart(chart, width='stretch')
                else:
                    if "question_rows" in sel_df.columns and "cycle_id" in sel_df.columns:
                        sel_df["question_rows"] = pd.to_numeric(sel_df["question_rows"], errors="coerce").fillna(0)
                        chart = alt.Chart(sel_df).mark_bar().encode(x=alt.X("cycle_id:N", title="Cycle"), y=alt.Y("question_rows:Q", title="Question rows"), tooltip=["cycle_id", "question_rows"]).properties(height=300)
                        st.altair_chart(chart, width='stretch')
                    else:
                        st.info("Insufficient columns for cycle+type chart for the selected question type.")

