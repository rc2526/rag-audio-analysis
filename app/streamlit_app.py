#!/usr/bin/env python3
import json
from pathlib import Path
import re
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_audio_analysis.chat_runner import run_chat_query
from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR, MANUAL_UNITS_CSV, SESSION_SUMMARIES_CSV, TOPIC_CATALOG_CSV
from rag_audio_analysis.settings import get_float, get_int, get_str


APP_TITLE = get_str("ui", "app_title", "rag-audio-analysis")
APP_CAPTION = get_str(
    "ui",
    "app_caption",
    "Cycle-by-cycle session-topic evidence analysis built from retrieved transcript evidence, manual units, and PI-question summaries.",
)
CYCLE_PREFIX = get_str("ui", "cycle_prefix", "PMHCycle")
UI_EXCERPT_CHARS = get_int("prompting", "ui_excerpt_chars", 280)
TOPIC_DEFINITION_EXCERPT_CHARS = 320

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_CAPTION)


QUESTION_LABELS = {
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


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


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
        "retrieved_evidence_count": "Retrieved evidence windows",
        "expected_manual_unit_count": "Expected manual units",
        "matched_manual_unit_count": "Matched manual units",
        "manual_unit_coverage": "Manual-unit coverage",
        "expected_subsection_count": "Expected subsections",
        "matched_subsection_count": "Matched subsections",
        "subsection_coverage": "Subsection coverage",
        "evidence_density": "Evidence density",
        "adherence_score": "Adherence score",
        "matched_manual_unit_ids": "Matched manual units",
        "matched_subsections": "Matched subsections",
        "sample_session_ids": "Transcript files",
        "query_text": "Query text",
        "answer_summary": "Automated answer",
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
    return view.rename(columns={k: v for k, v in rename_map.items() if k in view.columns})


TOPIC_DEFINITION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "session",
    "the",
    "this",
    "to",
    "using",
    "with",
}


def tokenize_topic_text(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if token and token not in TOPIC_DEFINITION_STOPWORDS
    ]


def split_summary_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def normalize_topic_label(topic_label: str) -> str:
    label = str(topic_label or "").strip()
    if not label:
        return ""
    label = label[0].lower() + label[1:]
    label = re.sub(r"\bPA\b", "physical activity", label)
    return label


def clean_summary_fragment(text: str) -> str:
    fragment = str(text or "").strip(" ,.;:")
    replacements = [
        r"^the session focuses on\s+",
        r"^the session deepens\s+",
        r"^the session also focuses on\s+",
        r"^participants are introduced to\s+",
        r"^participants discuss\s+",
        r"^participants reflect on\s+",
        r"^experiential practice includes\s+",
        r"^additional experiential practice includes\s+",
        r"^homework emphasizes\s+",
        r"^the nutrition and physical activity component focuses on\s+",
        r"^the nutrition component focuses on\s+",
    ]
    for pattern in replacements:
        fragment = re.sub(pattern, "", fragment, flags=re.IGNORECASE)
    return fragment.strip(" ,.;:")


def sentence_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for sentence in split_summary_sentences(text):
        parts = re.split(r";|, followed by |, with |, and ", sentence)
        for part in parts:
            cleaned = clean_summary_fragment(part)
            if cleaned:
                fragments.append(cleaned)
    return fragments


def choose_topic_context_fragments(topic_label: str, session_summary: str) -> list[str]:
    topic_tokens = set(tokenize_topic_text(topic_label))
    scored_fragments: list[tuple[int, int, str]] = []
    for idx, fragment in enumerate(sentence_fragments(session_summary)):
        fragment_tokens = set(tokenize_topic_text(fragment))
        overlap = len(topic_tokens & fragment_tokens)
        bonus = 1 if normalize_topic_label(topic_label) in fragment.lower() else 0
        score = overlap * 3 + bonus
        scored_fragments.append((score, -idx, fragment))

    scored_fragments.sort(reverse=True)
    chosen = [fragment for score, _, fragment in scored_fragments if score > 0][:2]
    if chosen:
        return chosen
    fallback = [clean_summary_fragment(sentence) for sentence in split_summary_sentences(session_summary)[:2]]
    return [fragment for fragment in fallback if fragment]


def derive_topic_definition(topic_label: str, session_summary: str) -> str:
    topic_label = str(topic_label or "").strip()
    session_summary = str(session_summary or "").strip()
    if not topic_label:
        return get_excerpt(session_summary, TOPIC_DEFINITION_EXCERPT_CHARS)
    if not session_summary:
        return f"Focuses on {normalize_topic_label(topic_label)} in this session."

    topic_phrase = normalize_topic_label(topic_label)
    context_fragments = choose_topic_context_fragments(topic_label, session_summary)
    context_text = "; ".join(context_fragments).strip()
    if not context_text:
        context_text = clean_summary_fragment(session_summary)

    gloss = f"Focuses on {topic_phrase} in the context of {context_text}."
    gloss = re.sub(r"\s+", " ", gloss).strip()
    return get_excerpt(gloss, TOPIC_DEFINITION_EXCERPT_CHARS)


def build_topic_catalog_for_display(topic_catalog: pd.DataFrame) -> pd.DataFrame:
    if topic_catalog.empty:
        return topic_catalog

    session_summaries = load_csv(SESSION_SUMMARIES_CSV)
    if session_summaries.empty:
        view = topic_catalog.copy()
        if "topic_definition" in view.columns:
            view["topic_definition"] = view.apply(
                lambda row: row["topic_definition"]
                if str(row.get("topic_definition", "")).strip()
                else f"This topic centers on {str(row.get('topic_label', '')).strip().lower()}."
                if str(row.get("topic_label", "")).strip()
                else "",
                axis=1,
            )
        return view

    view = topic_catalog.copy()
    summary_lookup = {
        str(row.get("session_num", "")).strip(): str(row.get("session_summary", "")).strip()
        for _, row in session_summaries.iterrows()
    }
    if "topic_definition" not in view.columns:
        view["topic_definition"] = ""

    view["topic_definition"] = view.apply(
        lambda row: str(row.get("topic_definition", "")).strip()
        or derive_topic_definition(
            str(row.get("topic_label", "")).strip(),
            summary_lookup.get(str(row.get("session_num", "")).strip(), ""),
        ),
        axis=1,
    )
    return view


def render_key() -> None:
    with st.expander("How to read this app", expanded=False):
        st.markdown("**What is shown here**")
        st.write("This app now centers on an automated cycle-level analysis pipeline rather than the older manual review queue.")
        st.markdown("**Fidelity values**")
        st.write("The primary fidelity view is cycle-level alignment to session-structured manual content.")
        st.write("Manual-unit coverage: matched manual units / expected manual units for a manual session within a cycle.")
        st.write("Subsection coverage: matched manual subsections / expected manual subsections for a manual session within a cycle.")
        st.write("Adherence score: `0.6 * manual-unit coverage + 0.4 * subsection coverage`.")
        st.write("Adherence label: `high` if score >= 0.66, `moderate` if >= 0.33, otherwise `low`.")
        st.markdown("**PI-question answers**")
        st.write("These are automated `gpt-oss:120b` summaries constrained to retrieved evidence and matching manual units.")
        st.markdown("**Retrieved evidence**")
        st.write("These are the evidence windows pulled from transcripts for cycle-level fidelity, topic-level fidelity, or PI-question mode.")


manual_units = load_csv(MANUAL_UNITS_CSV)
topic_catalog = build_topic_catalog_for_display(load_csv(TOPIC_CATALOG_CSV))
cycle_ids = list_cycle_ids()

all_fidelity = load_all_cycle_files("fidelity_summary.csv")
all_session_fidelity = load_all_cycle_files("session_fidelity_summary.csv")
all_pi_answers = load_all_cycle_files("pi_question_answers.csv")
all_evidence = load_all_cycle_files("topic_evidence.csv")
all_session_evidence = load_all_cycle_files("session_fidelity_evidence.csv")

with st.sidebar:
    st.header("Filters")
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


m1, m2, m3, m4 = st.columns(4)
m1.metric("Cycle folders", len(cycle_ids))
m2.metric("Manual-session fidelity rows", len(session_fidelity_view.index))
m3.metric("PI-question rows", len(answers_view.index))
m4.metric("Evidence rows", len(evidence_view.index))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Fidelity", "PI Questions", "Evidence Browser", "Manual Units", "RAG Chat"]
)

with tab1:
    st.subheader("Pipeline overview")
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
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Cycle-level manual-session fidelity across current filters**")
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
        ]
        st.dataframe(
            add_readable_columns(session_fidelity_view[[col for col in overview_cols if col in session_fidelity_view.columns]]),
            use_container_width=True,
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
        st.dataframe(
            add_readable_columns(
                fidelity_df[
                    [
                        "cycle_id",
                        "manual_session_num",
                        "manual_session_label",
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
                    ]
                ].sort_values(["cycle_id", "manual_session_num"])
            ),
            use_container_width=True,
            hide_index=True,
        )

        row_options = fidelity_df.index.tolist()
        selected_row_idx = st.selectbox(
            "Select a cycle/manual-session row for detail",
            row_options,
            format_func=lambda idx: f"{fidelity_df.at[idx, 'cycle_id']} | Session {fidelity_df.at[idx, 'manual_session_num']}",
        )
        row = fidelity_df.loc[selected_row_idx]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Manual-unit coverage", row.get("manual_unit_coverage", ""))
        d2.metric("Subsection coverage", row.get("subsection_coverage", ""))
        d3.metric("Evidence density", row.get("evidence_density", ""))
        d4.metric("Adherence", human_label(str(row.get("adherence_label", "")), ADHERENCE_LABELS))

        st.markdown("**Fidelity query**")
        st.code(str(row.get("fidelity_query", "")))
        st.markdown("**Session summary used for retrieval**")
        st.write(str(row.get("session_summary", "")) or "No session summary is saved yet for this manual session.")

        matched_ids = [x for x in str(row.get("matched_manual_unit_ids", "")).split(";") if x]
        expected_units = manual_units.copy()
        expected_units = expected_units[expected_units["manual_week"].astype(str) == str(row.get("manual_session_num", ""))]

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
                    use_container_width=True,
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
                    use_container_width=True,
                    hide_index=True,
                )

        supporting_evidence = session_evidence_view.copy()
        if not supporting_evidence.empty:
            supporting_evidence = supporting_evidence[
                (supporting_evidence["analysis_mode"].astype(str) == "session_fidelity")
                & (supporting_evidence["cycle_id"].astype(str) == str(row.get("cycle_id", "")))
                & (supporting_evidence["manual_session_num"].astype(str) == str(row.get("manual_session_num", "")))
            ]
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
            st.dataframe(
                add_readable_columns(
                    answer_df[
                        [
                            "cycle_id",
                            "session_num",
                            "topic_id",
                            "topic_label",
                            "question_id",
                            "retrieved_evidence_count",
                            "confidence",
                            "answer_summary",
                        ]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            row_options = answer_df.index.tolist()
            selected_answer_idx = st.selectbox(
                "Select an answer row for detail",
                row_options,
                format_func=lambda idx: f"{answer_df.at[idx, 'cycle_id']} | Session {answer_df.at[idx, 'session_num']} | {human_label(answer_df.at[idx, 'question_id'], QUESTION_LABELS)} | {answer_df.at[idx, 'topic_label']}",
            )
            row = answer_df.loc[selected_answer_idx]

            q1, q2, q3 = st.columns(3)
            q1.metric("Retrieved evidence windows", row.get("retrieved_evidence_count", ""))
            q2.metric("Confidence", row.get("confidence", ""))
            q3.metric("Question", human_label(str(row.get("question_id", "")), QUESTION_LABELS))

            st.markdown("**Query text**")
            st.code(str(row.get("query_text", "")))
            st.markdown("**Full model prompt**")
            prompt_text = str(row.get("prompt_text", ""))
            if prompt_text:
                st.code(prompt_text)
            else:
                st.info("This row was generated before prompt capture was added, so no saved prompt is available yet.")
            st.markdown("**Automated answer**")
            st.write(str(row.get("answer_summary", "")))
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
        mode_options = ["All modes"] + sorted(x for x in evidence_view["analysis_mode"].astype(str).unique() if x)
        selected_mode = st.selectbox(
            "Analysis mode",
            mode_options,
            format_func=lambda x: "All modes" if x == "All modes" else human_label(x, ANALYSIS_MODE_LABELS),
        )
        question_options = ["All questions"] + sorted(x for x in evidence_view["question_id"].astype(str).unique() if x)
        selected_question = st.selectbox(
            "Question filter",
            question_options,
            format_func=lambda x: "All questions" if x == "All questions" else human_label(x, QUESTION_LABELS),
            key="evidence_question_filter",
        )

        browser_df = evidence_view.copy()
        if selected_mode != "All modes":
            browser_df = browser_df[browser_df["analysis_mode"].astype(str) == selected_mode]
        if selected_question != "All questions":
            browser_df = browser_df[browser_df["question_id"].astype(str) == selected_question]

        if browser_df.empty:
            st.info("No evidence rows match the current filters.")
        else:
            display_df = browser_df.copy()
            display_df["excerpt"] = display_df["excerpt"].astype(str).apply(get_excerpt)
            st.dataframe(
                add_readable_columns(
                    display_df[
                        [
                            "cycle_id",
                            "session_num",
                            "topic_id",
                            "topic_label",
                            "analysis_mode",
                            "question_id",
                            "retrieval_rank",
                            "manual_unit_id_best_match",
                            "manual_unit_match_score",
                            "excerpt",
                        ]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            row_options = browser_df.index.tolist()
            selected_evidence_idx = st.selectbox(
                "Select an evidence row for detail",
                row_options,
                format_func=lambda idx: f"{browser_df.at[idx, 'cycle_id']} | Session {browser_df.at[idx, 'session_num']} | rank {browser_df.at[idx, 'retrieval_rank']} | {browser_df.at[idx, 'topic_label']}",
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
                use_container_width=True,
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

            c3, c4, c5 = st.columns(3)
            chat_weight_doc = c3.number_input("Document weight", min_value=0.0, max_value=2.0, value=float(chat_defaults["weight_doc"]), step=0.1)
            chat_weight_topic = c4.number_input("Topic weight", min_value=0.0, max_value=2.0, value=float(chat_defaults["weight_topic"]), step=0.1)
            chat_answer_with_model = c5.checkbox(
                "Generate answer with gpt-oss:120b",
                value=chat_defaults["answer_with_model"],
                help="When on, the app sends the retrieved evidence to the model for a grounded answer. When off, the app only shows retrieved evidence.",
            )

        submit_chat = st.form_submit_button("Run RAG chat")

    if submit_chat:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Running retrieval and preparing results..."):
                    payload = run_chat_query(
                        question=question.strip(),
                        cycle_id="" if chat_cycle_option == "All cycles" else chat_cycle_option,
                        topk=int(chat_topk),
                        weight_doc=float(chat_weight_doc),
                        weight_topic=float(chat_weight_topic),
                        include_manual=chat_include_manual,
                        answer_with_model=chat_answer_with_model,
                    )
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
        st.markdown("**Query used**")
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

            st.markdown("**Grounded answer**")
            st.write(str(answer.get("answer_summary", "")) or "No answer returned.")

            st.markdown("**Full model prompt**")
            st.code(str(payload.get("prompt_text", "")))
        else:
            st.info("This run used evidence-only mode. The app searched the index and returned retrieved evidence, but it did not send that evidence to gpt-oss:120b for an answer.")

        evidence_rows = payload.get("evidence", []) or []
        st.markdown("**Retrieved evidence**")
        if evidence_rows:
            evidence_df = pd.DataFrame(evidence_rows)
            if "text" in evidence_df.columns:
                evidence_df["text"] = evidence_df["text"].astype(str).apply(get_excerpt)
            display_cols = [
                col
                for col in [
                    "rank",
                    "source_type",
                    "cycle_id",
                    "session_id",
                    "score_combined",
                    "score_doc",
                    "score_topic",
                    "manual_unit_id_best_match",
                    "manual_unit_match_score",
                    "text",
                ]
                if col in evidence_df.columns
            ]
            st.dataframe(add_readable_columns(evidence_df[display_cols]), use_container_width=True, hide_index=True)

            for idx, row in enumerate(evidence_rows, start=1):
                title = f"E{idx} | {row.get('source_type','')} | rank {row.get('rank','')}"
                with st.expander(title, expanded=False):
                    st.write(f"Cycle: {row.get('cycle_id','')}")
                    st.write(f"Session: {row.get('session_id','')}")
                    st.write(f"Manual unit: {row.get('manual_unit_id_best_match','')}")
                    st.write(f"Manual-unit similarity: {row.get('manual_unit_match_score','')}")
                    st.write(str(row.get("text", "")))
        else:
            st.info("No evidence rows were returned.")
