import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

from rag_audio_analysis.config import (
    CYCLE_ANALYSIS_DIR,
    SESSION_SUMMARIES_CSV,
    SOURCE_BUILD_AND_QUERY,
    SOURCE_META,
    SOURCE_MANUAL,
    SOURCE_RAG_INDEX,
    SOURCE_TOPIC_LIST,
    SOURCE_TOPICS_CSV,
    SOURCE_MANUAL_DOC_INDICES,
    SOURCE_MANUAL_DOC_TOPIC_MAP,
    SOURCE_TRANSCRIPTS_GLOB,
    SPEAKER_ROLE_MAP_CSV,
)
from rag_audio_analysis.settings import get_float, get_int, get_str


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "topic"


def load_source_meta() -> list[dict[str, Any]]:
    if not SOURCE_META.exists():
        return []
    return json.loads(SOURCE_META.read_text(encoding="utf-8"))


_SOURCE_QUERY_MODULE = None
_EMBED_MODEL = None
_TOPIC_EMBED_CACHE = None
_MANUAL_UNIT_EMBED_CACHE = None
_SESSION_SUMMARY_CACHE = None

TOPIC_DEFINITION_MAP = {
    "role-of-stress-in-body-weight-and-food-intake": "Explores how stress affects appetite, eating behavior, and body-weight related health choices, especially at the start of the group.",
    "implementing-healthy-food-intake-and-pa-in-a-mindful-and-present-manner": "Frames healthy eating and physical activity as practices that can be approached with mindful, present-moment awareness and supported by weekly goals.",
    "stress-reaction-cycle-and-habitual-stress-responses": "Introduces the stress reaction cycle and helps participants notice repeated, habitual ways of reacting to stress through the buckets of experience worksheet.",
    "domains-buckets-of-stress-experience": "Organizes stress into thoughts, feelings, sensations, and urges or behaviors so participants can track their experience more clearly in daily homework.",
    "definition-of-mindfulness-and-stop-skill-teaching": "Defines mindfulness as present, nonjudgmental awareness and teaches STOP as a brief handout-based skill for pausing and responding more intentionally.",
    "focused-on-beverages": "Draws attention to beverage choices, especially sugar-sweetened drinks, and pairs this with concrete beverage goals and label awareness.",
    "physical-activity-steps-and-smart-goals": "Uses step tracking, Fitbit monitoring, and SMART goals to build realistic, measurable physical activity habits.",
    "stress-facilitated-autopilot-and-monkey-mind-effects-on-choices": "Examines how stress, autopilot, and monkey mind shape daily choices, including eating-related decisions and routine habits.",
    "eating-mindfully-types-of-hunger-and-tips-for-mindful-eating": "Teaches mindful eating through direct food exercises, the seven types of hunger handout, and practical tips for slowing down meals.",
    "mindful-eating-with-healthy-foods-choices-and-preparation": "Applies mindful eating to healthier food selection, food preparation, and ingredient choices for self and child, including visually appealing meals.",
    "stress-and-unpleasant-events-with-a-focus-on-emotional-reactivity": "Focuses on unpleasant events and emotional reactivity, especially how stress responses affect oneself, parenting situations, and relationships.",
    "pleasant-activity-list": "Uses pleasant activities and pleasant or unpleasant event calendars as deliberate tools for stress coping, self-care, and positive mood support.",
    "mindfulness-in-reacting-to-unpleasant-events": "Builds skill in noticing unpleasant experiences and stressful parenting moments without immediately reacting on autopilot.",
    "mindful-approach-to-portion-control-and-mindful-eating": "Links portion awareness with mindful eating by comparing serving sizes and portion sizes and slowing down food decisions.",
    "reacting-versus-responding-to-stress": "Distinguishes automatic stress reactions from slower, more intentional responses, especially in parenting situations recorded in the reactions or responses log.",
    "awareness-of-stress-experience-domains": "Strengthens awareness of how stress appears across thoughts, emotions, bodily sensations, and urges, especially during parenting stress.",
    "breaking-automatic-reactions-for-stress-coping": "Uses mindfulness and awareness to interrupt automatic stress responses and support more effective coping during parenting and daily stress.",
    "mindful-attention-to-calories-macro-and-micro-nutrients-and-food-groups": "Brings mindful attention to calories, nutrients, and food groups to support informed food decisions and label-based learning.",
    "effective-communication-during-stress": "Examines how stress affects communication and how mindful awareness can support clearer, more effective exchanges with children and other adults.",
    "awareness-of-stress-reactions-and-emotions": "Helps participants identify emotional and bodily stress reactions so they can respond with greater awareness during difficult conversations.",
    "interpersonal-stress-communication": "Explores how stress shapes communication in close relationships and how different communication styles affect outcomes.",
    "difficult-communications-calendar": "Uses a communication calendar to track challenging conversations, emotional reactions, and what each person wanted or received.",
    "mindful-attention-to-breakfast-and-healthy-choices": "Encourages mindful attention to breakfast habits, barriers to eating breakfast, and healthier food choices under everyday stress.",
    "deepening-mindfulness-practices": "Extends mindfulness practice beyond basic skills through repeated home exercises, logs, and more deliberate use in daily life.",
    "attitudes-towards-mindfulness": "Introduces core mindfulness attitudes such as nonjudging, patience, acceptance, trust, and letting go, and asks participants to track them during the week.",
    "linking-stress-experience-buckets-to-mindfulness-strategies": "Connects different parts of the stress experience to specific mindfulness strategies that can be used during difficult situations and distressing imagery.",
    "learning-the-letting-go-skill": "Develops the capacity to release fixation on difficult thoughts, feelings, or urges without avoidance and to practice letting go during stress.",
    "understanding-food-labels-and-incorporating-that-information-in-making-mindful-choices": "Uses food labels, percent daily value, and added sugar information to support more mindful and informed food choices.",
    "mindful-food-cravings-stress-stop-and-letting-go-skills": "Applies mindfulness, STOP, and letting-go skills to food cravings and stress-related urges to eat, with direct attention to craving cycles.",
    "healthy-foods-on-a-budget": "Shows how healthy food choices can be made while staying within a family budget through planning, shopping, and meal preparation.",
    "meal-planning-shopping-tips-to-stay-within-family-budget": "Uses meal planning, shopping strategies, and budgeting skills to support affordable, healthier eating across the week.",
    "mindful-parenting-skills": "Applies mindfulness to parenting by helping parents notice stress reactions, use tools like ONESIE, and respond more calmly and intentionally to their child.",
    "stressful-parenting-experiences": "Centers stressful parenting situations and helps participants reflect on their thoughts, feelings, sensations, and urges in those moments through the parenting log.",
    "mindfulness-activity-log-and-parenting-journal": "Uses logs and journals to strengthen daily mindfulness practice and reflection on parenting stress, reactions, and responses over time.",
    "parent-child-play-activities": "Encourages mindful connection and positive interaction through shared parent-child play and other relationship-building activities.",
    "mindful-activities-for-fruits-and-veggies-consumption-at-snacks-and-meals": "Uses mindful family routines to increase fruit and vegetable intake during snacks and meals and to involve children in healthier eating.",
    "mindfulness-during-stressful-situations": "Builds the ability to practice mindfulness in the middle of stressful situations, including fear, anxiety, shame, and guilt, rather than only during calm moments.",
    "noticing-and-managing-difficult-situations-and-emotions": "Strengthens awareness of difficult situations and emotions so they can be named, tolerated, and managed more skillfully.",
    "mindful-meal-preparations-and-awareness-of-food-substitutions": "Applies mindfulness to meal preparation and healthier food substitutions in everyday cooking and recipe modification.",
    "awareness-and-acceptance-of-personal-and-child-stress-and-emotions": "Promotes awareness and acceptance of stress and emotions in both parent and child without immediate judgment or reaction, especially during cravings and distress.",
    "emotion-reactivity-and-automaticity-regulation": "Targets emotional automaticity and reactivity so decisions about food, activity, and parenting become more intentional and less impulsive.",
    "application-of-mindful-eating-skills-for-choices-of-healthy-fast-foods-and-portion-sizes": "Applies mindful eating skills to fast-food choices, portion control, and healthier eating on the go by slowing down cravings and decisions.",
    "compassion-for-self-and-child": "Cultivates compassion toward both self and child during stress, difficulty, and change while reviewing how to continue mindfulness after the group ends.",
    "stress-regulation-and-reframing-stress-and-parenting": "Supports stress regulation and reframing of parenting challenges in ways that promote calmer, more adaptive responses after the program.",
    "changing-relationships-to-food-and-exercise": "Encourages a more nourishing, sustainable, and self-compassionate relationship to food and exercise as participants set post-group goals.",
    "integration-of-mindfulness-skills-with-healthy-nutrition-practices-for-self-and-child": "Integrates mindfulness skills with healthy nutrition practices, family routines, and future goal setting for both parent and child.",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def first_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    return match.group(1).strip() if match else text


def get_topic_definition(topic_id: str, topic_label: str = "", session_summary: str = "") -> str:
    topic_id = str(topic_id or "").strip()
    if topic_id in TOPIC_DEFINITION_MAP:
        return TOPIC_DEFINITION_MAP[topic_id]

    label = clean_text(topic_label)
    summary = first_sentence(session_summary).rstrip(".!?")
    if summary:
        return f"Addresses {label.lower()} in the context of {summary}."
    if label:
        return f"Addresses {label.lower()}."
    return ""


def build_session_fidelity_query(
    session_num: str,
    session_summary: str,
    topic_labels: list[str] | None = None,
) -> str:
    topic_labels = [clean_text(label) for label in (topic_labels or []) if clean_text(label)]
    session_summary = clean_text(session_summary)
    priority_clause = ""
    if topic_labels:
        priority_clause = f"Priority session elements: {'; '.join(topic_labels)}."

    parts = [
        f"Manual Session {session_num}.",
        "Retrieve transcript evidence for this exact manual session.",
        "Prioritize concrete facilitator teaching, participant practice, handouts, homework, mindfulness exercises, parenting applications, nutrition content, and physical activity content that belong to this session.",
        "Avoid generic themes from other sessions unless they are clearly enacted as part of this manual session.",
    ]
    if priority_clause:
        parts.append(priority_clause)
    if session_summary:
        parts.append(f"Session summary: {session_summary}")
    return " ".join(part for part in parts if part).strip()


def build_pi_query_text(
    session_num: str,
    topic_label: str,
    topic_definition: str,
    question_label: str,
    query_template: str,
    session_summary: str = "",
) -> str:
    base = clean_text(
        query_template.format(
            session_num=session_num,
            topic_label=topic_label,
            topic_definition=topic_definition,
            question_label=question_label,
            session_summary=first_sentence(session_summary),
        )
    )
    parts = [
        f"Manual Session {session_num}.",
        base,
    ]
    if topic_definition:
        parts.append(f"Topic definition: {topic_definition}")
    if session_summary:
        parts.append(f"Session context: {first_sentence(session_summary)}")
    return " ".join(part for part in parts if part).strip()


def load_source_query_module():
    global _SOURCE_QUERY_MODULE
    if _SOURCE_QUERY_MODULE is not None:
        return _SOURCE_QUERY_MODULE
    if not SOURCE_BUILD_AND_QUERY.exists():
        raise FileNotFoundError(f"Original RAG query module not found at {SOURCE_BUILD_AND_QUERY}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("rag_audio_build_and_query", str(SOURCE_BUILD_AND_QUERY))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SOURCE_BUILD_AND_QUERY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SOURCE_QUERY_MODULE = module
    return module


def get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    module = load_source_query_module()
    _EMBED_MODEL = module._get_model(None)
    return _EMBED_MODEL


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedding_model()
    matrix = model.encode(texts, convert_to_numpy=True).astype(np.float32)
    return normalize_rows(matrix)


def query_evidence_by_manual_unit_similarity(
    session_manual_units: list[dict[str, Any]],
    cycle_id: str,
    meta_rows: list[dict[str, Any]],
    path_lookup: dict[str, list[int]],
    window: int,
    topk_per_unit: int = 0,
    min_similarity: float | None = None,
    global_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Vectorized similarity: compute manual-unit vs cycle-window cosine similarities.

    Returns list of evidence rows annotated with query_manual_unit_id and mapped_manual_unit_match_score
    for pairs meeting the min_similarity threshold. If topk_per_unit>0, limit matches per unit.
    """
    if not session_manual_units:
        return []
    if min_similarity is None:
        min_similarity = get_float("topic_matching", "manual_unit_min_similarity", 0.6)

    # Prepare manual unit texts and ids
    manual_texts: list[str] = [str(u.get("matching_text", u.get("text", "")) or "").strip() for u in session_manual_units]
    manual_ids: list[str] = [str(u.get("manual_unit_id", "")) for u in session_manual_units]
    if not manual_texts:
        return []

    manual_embs = encode_texts(manual_texts)

    # Collect candidate transcript windows for the given cycle
    window_texts: list[str] = []
    window_doc_indices: list[int] = []
    window_meta_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(meta_rows):
        path = str(row.get("path", "") or row.get("file", "") or "")
        if not path:
            continue
        if infer_cycle_id(path) != cycle_id:
            continue
        # Expand context for this doc_index
        ctx = expand_transcript_context(idx, meta_rows=meta_rows, path_lookup=path_lookup, window=window)
        text = str(ctx.get("text", "") or "").strip()
        if not text:
            continue
        window_texts.append(text)
        window_doc_indices.append(idx)
        window_meta_rows.append(ctx)

    if not window_texts:
        return []

    window_embs = encode_texts(window_texts)

    # Compute cosine similarities matrix: manual_embs (m x d) dot window_embs.T (d x n) => (m x n)
    sims = np.matmul(manual_embs, window_embs.T)

    rows: list[dict[str, Any]] = []
    m, n = sims.shape
    for i in range(m):
        sims_row = sims[i]
        # find indices meeting threshold
        candidate_idxs = [j for j in range(n) if sims_row[j] >= min_similarity]
        # sort by descending similarity
        candidate_idxs.sort(key=lambda j: float(sims_row[j]), reverse=True)
        if topk_per_unit and topk_per_unit > 0:
            candidate_idxs = candidate_idxs[:topk_per_unit]
        for rank, j in enumerate(candidate_idxs, start=1):
            ctx = window_meta_rows[j]
            doc_index = int(window_doc_indices[j])
            text = window_texts[j]
            score = float(sims_row[j])
            rows.append(
                {
                    "doc_index": doc_index,
                    "retrieval_rank": rank,
                    "session_id": infer_session_id(str(ctx.get("path", "") or ctx.get("file", ""))),
                    "cycle_id": cycle_id,
                    "speaker": ctx.get("speaker", ""),
                    "score_combined": score,
                    "score_doc": "",
                    "score_topic": "",
                    "text": text,
                    "manual_unit_id_best_match": manual_ids[i],
                    "manual_unit_match_score": score,
                    "query_manual_unit_id": manual_ids[i],
                    "query_manual_unit_subsection": session_manual_units[i].get("manual_subsection", ""),
                    "query_text": manual_texts[i],
                    "mapped_manual_unit_match_score": score,
                    "mapped_manual_unit_accepted": bool(score >= float(min_similarity)),
                }
            )

    # Optionally cap globally by top similarity across all rows
    if global_cap and global_cap > 0 and len(rows) > global_cap:
        rows.sort(key=lambda r: float(r.get("mapped_manual_unit_match_score", 0.0)), reverse=True)
        rows = rows[:global_cap]

    return rows


def normalize_cycle_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize cycle-level DataFrame schemas for downstream consumers.

    Ensures common columns exist and renames obvious synonyms so the UI can
    safely select/intersect columns without KeyError. This is idempotent.

    Rules applied:
    - rename 'manual_session_num' -> 'session_num' when present
    - rename 'source_topic_id' -> 'topic_id' and 'source_topic_label' -> 'topic_label'
    - ensure the presence of canonical columns: session_num, question_id, topic_id, topic_label
      by adding them with empty-string values if missing
    - coerce dtype to string for those canonical columns
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    view = df.copy()

    # rename common synonyms
    if "manual_session_num" in view.columns and "session_num" not in view.columns:
        view = view.rename(columns={"manual_session_num": "session_num"})
    if "source_topic_id" in view.columns and "topic_id" not in view.columns:
        view = view.rename(columns={"source_topic_id": "topic_id"})
    if "source_topic_label" in view.columns and "topic_label" not in view.columns:
        view = view.rename(columns={"source_topic_label": "topic_label"})

    # ensure canonical columns exist
    for c in ["session_num", "question_id", "topic_id", "topic_label"]:
        if c not in view.columns:
            view[c] = ""

    # coerce to string to avoid mixed-type comparison issues later
    for c in ["session_num", "question_id", "topic_id", "topic_label"]:
        try:
            view[c] = view[c].astype(str)
        except Exception:
            view[c] = view[c].apply(lambda x: "" if pd.isna(x) else str(x))

    return view


def load_topic_entries() -> list[dict[str, str]]:
    csv_entries: list[dict[str, str]] = []
    csv_lookup: dict[str, dict[str, str]] = {}
    if SOURCE_TOPICS_CSV.exists():
        with open(SOURCE_TOPICS_CSV, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            field_map = {k.lower(): k for k in (reader.fieldnames or [])}
            topic_key = field_map.get("topic")
            ncomp_key = field_map.get("n component") or field_map.get("n_component") or field_map.get("ncomponent")
            session_key = field_map.get("session")
            for row in reader:
                label = ""
                if topic_key and row.get(topic_key):
                    label = row[topic_key].strip()
                elif ncomp_key and row.get(ncomp_key):
                    label = row[ncomp_key].strip()
                if not label:
                    continue
                session_raw = (row.get(session_key, "") if session_key else "").strip()
                session_num = extract_session_num(session_raw)
                entry = {
                    "id": slugify(label),
                    "label": label,
                    "session_num": session_num,
                    "session_label": f"Session {session_num}" if session_num else session_raw,
                }
                csv_entries.append(entry)
                csv_lookup[entry["id"]] = entry

    if SOURCE_TOPIC_LIST.exists():
        raw = json.loads(SOURCE_TOPIC_LIST.read_text(encoding="utf-8"))
        if raw and isinstance(raw[0], dict):
            return [
                {
                    "id": item.get("id") or slugify(item.get("label", "topic")),
                    "label": item.get("label", ""),
                    "session_num": csv_lookup.get(item.get("id") or slugify(item.get("label", "topic")), {}).get("session_num", ""),
                    "session_label": csv_lookup.get(item.get("id") or slugify(item.get("label", "topic")), {}).get("session_label", ""),
                }
                for item in raw
            ]
        return [
            {
                "id": slugify(label),
                "label": label,
                "session_num": csv_lookup.get(slugify(label), {}).get("session_num", ""),
                "session_label": csv_lookup.get(slugify(label), {}).get("session_label", ""),
            }
            for label in raw
            if str(label).strip()
        ]

    return csv_entries


def load_session_summaries() -> list[dict[str, str]]:
    global _SESSION_SUMMARY_CACHE
    if _SESSION_SUMMARY_CACHE is not None:
        return _SESSION_SUMMARY_CACHE
    if not SESSION_SUMMARIES_CSV.exists():
        _SESSION_SUMMARY_CACHE = []
        return _SESSION_SUMMARY_CACHE

    rows: list[dict[str, str]] = []
    with open(SESSION_SUMMARIES_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_num = str(row.get("session_num", "") or "").strip()
            rows.append(
                {
                    "session_num": session_num,
                    "session_label": str(row.get("session_label", "") or f"Session {session_num}").strip(),
                    "session_summary": str(row.get("session_summary", "") or "").strip(),
                }
            )
    _SESSION_SUMMARY_CACHE = rows
    return _SESSION_SUMMARY_CACHE


def get_session_summary(session_num: str) -> dict[str, str]:
    target = str(session_num or "").strip()
    for row in load_session_summaries():
        if str(row.get("session_num", "")).strip() == target:
            return row
    return {
        "session_num": target,
        "session_label": f"Session {target}" if target else "",
        "session_summary": "",
    }


def load_manual_topic_lookup() -> dict[int, dict[str, str]]:
    if not (SOURCE_MANUAL_DOC_INDICES.exists() and SOURCE_MANUAL_DOC_TOPIC_MAP.exists()):
        return {}

    topic_entries = load_topic_entries()
    manual_indices = json.loads(SOURCE_MANUAL_DOC_INDICES.read_text(encoding="utf-8"))
    topic_map = json.loads(SOURCE_MANUAL_DOC_TOPIC_MAP.read_text(encoding="utf-8"))
    lookup: dict[int, dict[str, str]] = {}
    for meta_idx, topic_idx in zip(manual_indices, topic_map):
        topic_i = int(topic_idx)
        if 0 <= topic_i < len(topic_entries):
            lookup[int(meta_idx)] = topic_entries[topic_i]
    return lookup


def get_manual_chunks() -> list[dict[str, Any]]:
    meta = load_source_meta()
    if meta:
        return [row for row in meta if (row.get("source") or "") == "manual.txt"]
    return chunk_manual_text(str(SOURCE_MANUAL))


def get_transcript_turns() -> list[dict[str, Any]]:
    return parse_transcript_turns(SOURCE_TRANSCRIPTS_GLOB)


def get_rag_index_rows() -> list[dict[str, Any]]:
    return load_source_meta()


def infer_cycle_id(path_str: str) -> str:
    path = Path(path_str)
    for part in path.parts:
        match = re.search(r"(pmhcycle\d+)", part, flags=re.IGNORECASE)
        if match:
            token = match.group(1)
            return token[0].upper() + token[1:]
    return ""


def infer_session_id(path_str: str) -> str:
    return Path(path_str).stem


def infer_week_num(text: str) -> str:
    match = re.search(r"week\s+(\d+)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def extract_session_num(value: str) -> str:
    match = re.search(r"(\d+)", str(value or ""))
    return match.group(1) if match else ""


def infer_speaker_role(speaker_label: str) -> str:
    s = (speaker_label or "").strip().lower()
    if not s:
        return "unknown"
    if any(token in s for token in ("facilitator", "leader", "counselor", "therapist", "instructor")):
        return "facilitator"
    if s in {"group", "all", "everyone"}:
        return "group"
    if s.startswith("participant"):
        return "participant"
    return "unknown"


def bool_to_str(value: bool) -> str:
    return "1" if value else "0"


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "about",
    "what",
    "when",
    "where",
    "have",
    "will",
    "they",
    "them",
    "their",
    "there",
    "were",
    "which",
    "while",
    "being",
    "each",
    "just",
    "also",
    "than",
    "then",
    "does",
    "really",
    "make",
    "made",
    "much",
    "more",
    "very",
    "because",
    "would",
    "could",
    "should",
    "here",
    "week",
    "session",
    "topic",
}


TS_SPK_RE = re.compile(r"^\s*\[(\d+(?:\.\d+)?)s\]\s*([A-Za-z0-9_ -]+):\s*(.*)$")
SESSION_HEADER_RE = re.compile(r"^\s*Session\s+(\d+)\s*:?\s*$", re.IGNORECASE)

MANUAL_SUBSECTION_RULES = [
    (re.compile(r"^\s*Handouts:\s*$", re.IGNORECASE), ("handouts", "Handouts")),
    (re.compile(r"^\s*Homework:\s*$", re.IGNORECASE), ("homework", "Homework")),
    (re.compile(r"^\s*Audio Recordings:\s*$", re.IGNORECASE), ("activity", "Audio Recordings")),
    (re.compile(r"^\s*Stress Discussion:\s*$", re.IGNORECASE), ("discussion", "Stress Discussion")),
    (re.compile(r"^\s*Discussion Points:\s*$", re.IGNORECASE), ("discussion", "Discussion Points")),
    (re.compile(r"^\s*Review/Discussion.*:\s*$", re.IGNORECASE), ("discussion", "Review / Discussion")),
    (re.compile(r"^\s*Breathing Exercise:\s*$", re.IGNORECASE), ("breathing_exercise", "Breathing Exercise")),
    (re.compile(r"^\s*Nutrition and Physical Activity\s*:?\s*$", re.IGNORECASE), ("physical_activity_and_nutrition", "Nutrition and Physical Activity")),
    (re.compile(r"^\s*Physical Activity and Nutrition\s*:?\s*$", re.IGNORECASE), ("physical_activity_and_nutrition", "Physical Activity and Nutrition")),
    (re.compile(r"^\s*Nutrition:\s*$", re.IGNORECASE), ("physical_activity_and_nutrition", "Nutrition")),
    (re.compile(r"^\s*Physical Activity:\s*$", re.IGNORECASE), ("physical_activity_and_nutrition", "Physical Activity")),
    (re.compile(r"^\s*Welcome, Introductions, Rules, Confidentiality:\s*$", re.IGNORECASE), ("other_instructions", "Welcome / Rules")),
    (re.compile(r"^\s*Program Information:\s*$", re.IGNORECASE), ("other_instructions", "Program Information")),
    (re.compile(r"^\s*Rules of the Group:\s*$", re.IGNORECASE), ("other_instructions", "Rules of the Group")),
    (re.compile(r"^\s*Group Content/Structure:\s*$", re.IGNORECASE), ("other_instructions", "Group Content / Structure")),
]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def parse_transcript_turns(glob_path: str) -> list[dict[str, Any]]:
    import glob

    rows: list[dict[str, Any]] = []
    for fp in sorted(glob.glob(glob_path, recursive=True)):
        try:
            lines = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        cur: Optional[dict[str, Any]] = None
        for line in lines:
            match = TS_SPK_RE.match(line)
            if match:
                if cur:
                    rows.append(cur)
                cur = {
                    "source": Path(fp).name,
                    "path": fp,
                    "speaker": match.group(2),
                    "text": match.group(3).strip(),
                }
            else:
                if not line.strip():
                    continue
                if cur:
                    cur["text"] = (cur["text"] + " " + line.strip()).strip()
                else:
                    cur = {
                        "source": Path(fp).name,
                        "path": fp,
                        "speaker": "unknown",
                        "text": line.strip(),
                    }
        if cur:
            rows.append(cur)
    return rows


def detect_manual_subsection(line: str) -> Optional[tuple[str, str]]:
    stripped = (line or "").strip()
    for pattern, labels in MANUAL_SUBSECTION_RULES:
        if pattern.match(stripped):
            return labels
    if stripped.endswith(":") and len(stripped) <= 90:
        label = stripped[:-1].strip()
        lowered = label.lower()
        if any(token in lowered for token in ("breathing", "meditation")):
            return ("breathing_exercise", label)
        if any(token in lowered for token in ("nutrition", "physical activity", "healthy eating", "beverage", "drink", "recipe", "food label")):
            return ("physical_activity_and_nutrition", label)
        if any(token in lowered for token in ("discussion", "review", "stress", "communication", "reacting", "responding")):
            return ("discussion", label)
        if any(token in lowered for token in ("activity", "exercise", "stretch", "onesie", "fear", "acceptance", "craving", "raisin")):
            return ("activity", label)
        if any(token in lowered for token in ("instruction", "program", "rules", "welcome")):
            return ("other_instructions", label)
        return ("other_instructions", label)
    return None


def split_text_within_section(text: str, max_words: int = 220) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def find_manual_content_start(lines: list[str]) -> int:
    first_subsection_idx = None
    for idx, line in enumerate(lines):
        if detect_manual_subsection(line):
            first_subsection_idx = idx
            break
    if first_subsection_idx is None:
        return 0
    for idx in range(first_subsection_idx, -1, -1):
        if SESSION_HEADER_RE.match((lines[idx] or "").strip()):
            return idx
    return 0


def get_structured_manual_units(max_words: Optional[int] = None) -> list[dict[str, Any]]:
    if max_words is None:
        max_words = get_int("manual_parsing", "max_words_per_unit", 220)
    text = Path(SOURCE_MANUAL).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    start_idx = find_manual_content_start(lines)
    lines = lines[start_idx:]

    topic_index = build_topic_keyword_index()
    units: list[dict[str, Any]] = []
    current_session_num = ""
    current_session_label = ""
    current_category = "other_instructions"
    current_subsection = "Other Instructions"
    current_lines: list[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        text_block = "\n".join(line.strip() for line in current_lines if line.strip()).strip()
        if not text_block or not current_session_label:
            current_lines = []
            return
        heading_prefix = f"{current_session_label} {current_subsection}".strip()
        for chunk in split_text_within_section(text_block, max_words=max_words):
            combined_text = f"{heading_prefix}\n{chunk}".strip()
            topic_match = infer_topic_for_text(combined_text, topic_index=topic_index)
            units.append(
                {
                    "source": SOURCE_MANUAL.name,
                    "path": str(SOURCE_MANUAL),
                    "speaker": "manual",
                    "text": chunk,
                    "session_label": current_session_label,
                    "session_num": current_session_num,
                    "manual_section": current_session_label,
                    "manual_subsection": current_subsection,
                    "manual_category": current_category,
                    "topic_id": topic_match.get("topic_id", ""),
                    "topic_match_score": topic_match.get("score", ""),
                    "manual_week": current_session_num,
                    "matching_text": combined_text,
                }
            )
        current_lines = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        session_match = SESSION_HEADER_RE.match(stripped)
        if session_match:
            flush_current()
            current_session_num = session_match.group(1)
            current_session_label = f"Session {current_session_num}"
            current_category = "other_instructions"
            current_subsection = "Other Instructions"
            continue

        subsection_match = detect_manual_subsection(stripped)
        if subsection_match and current_session_label:
            flush_current()
            current_category, current_subsection = subsection_match
            continue

        if current_session_label:
            current_lines.append(stripped)

    flush_current()
    return units


def chunk_manual_text(path: str, chunk_size: int = 250) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    words = text.split()
    chunks: list[dict[str, Any]] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(
            {
                "source": Path(path).name,
                "path": path,
                "speaker": "manual",
                "text": chunk,
            }
        )
        i += chunk_size
    return chunks


def is_manual_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source", "") or "")
    path = str(row.get("path", "") or "")
    return source == "manual.txt" or path.endswith("/manual.txt")


def build_topic_keyword_index() -> list[dict[str, Any]]:
    topics = load_topic_entries()
    return [
        {
            "id": topic["id"],
            "label": topic["label"],
            "session_num": topic.get("session_num", ""),
            "session_label": topic.get("session_label", ""),
            "manual_week": topic.get("session_num", ""),
            "embedding_text": " ".join(
                part
                for part in [
                    topic.get("session_label", ""),
                    topic.get("label", ""),
                ]
                if str(part).strip()
            ).strip(),
        }
        for topic in topics
    ]


def get_topic_embedding_index(topic_index: Optional[list[dict[str, Any]]] = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    global _TOPIC_EMBED_CACHE
    index = topic_index or build_topic_keyword_index()
    if topic_index is None and _TOPIC_EMBED_CACHE is not None:
        return _TOPIC_EMBED_CACHE
    texts = [str(topic.get("embedding_text", "") or topic.get("label", "")).strip() for topic in index]
    embeddings = encode_texts(texts)
    if topic_index is None:
        _TOPIC_EMBED_CACHE = (index, embeddings)
    return index, embeddings


def infer_topic_for_text(text: str, topic_index: Optional[list[dict[str, Any]]] = None) -> dict[str, str]:
    index, embeddings = get_topic_embedding_index(topic_index)
    if not index or embeddings.size == 0:
        return {"topic_id": "", "topic_label": "", "topic_confidence": "", "manual_week_expected": "", "score": ""}

    qvec = encode_texts([text])
    if qvec.size == 0:
        return {"topic_id": "", "topic_label": "", "topic_confidence": "", "manual_week_expected": "", "score": ""}
    sims = embeddings.dot(qvec[0])
    best_idx = int(np.argmax(sims))
    best = index[best_idx]
    best_score = float(sims[best_idx])

    min_score = get_float("topic_matching", "topic_min_similarity", 0.30)
    high_score = get_float("topic_matching", "topic_high_similarity", 0.55)
    medium_score = get_float("topic_matching", "topic_medium_similarity", 0.42)

    if best_score < min_score:
        return {"topic_id": "", "topic_label": "", "topic_confidence": "", "manual_week_expected": "", "score": ""}

    if best_score >= high_score:
        confidence = "high"
    elif best_score >= medium_score:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "topic_id": best["id"],
        "topic_label": best["label"],
        "topic_confidence": confidence,
        "manual_week_expected": best.get("manual_week", ""),
        "score": f"{best_score:.4f}",
    }


def build_manual_unit_index() -> list[dict[str, Any]]:
    manual_chunks = get_structured_manual_units()
    units: list[dict[str, Any]] = []
    for i, chunk in enumerate(manual_chunks, start=1):
        text = chunk.get("text", "")
        matching_text = chunk.get("matching_text", text)
        units.append(
            {
                "manual_unit_id": f"MAN_{i:04d}",
                "manual_chunk_index": i - 1,
                "topic_id": chunk.get("topic_id", ""),
                "manual_week": chunk.get("manual_week", infer_week_num(text)),
                "manual_section": chunk.get("manual_section", ""),
                "manual_subsection": chunk.get("manual_subsection", ""),
                "manual_category": chunk.get("manual_category", ""),
                "text": text,
                "matching_text": matching_text,
            }
        )
    return units


def get_manual_unit_embedding_index(manual_units: Optional[list[dict[str, Any]]] = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    global _MANUAL_UNIT_EMBED_CACHE
    units = manual_units or build_manual_unit_index()
    if manual_units is None and _MANUAL_UNIT_EMBED_CACHE is not None:
        return _MANUAL_UNIT_EMBED_CACHE
    texts = [str(unit.get("matching_text", "") or unit.get("text", "")).strip() for unit in units]
    embeddings = encode_texts(texts)
    if manual_units is None:
        _MANUAL_UNIT_EMBED_CACHE = (units, embeddings)
    return units, embeddings


def infer_manual_unit_for_text(text: str, topic_id: str = "", manual_units: Optional[list[dict[str, Any]]] = None) -> dict[str, str]:
    units, embeddings = get_manual_unit_embedding_index(manual_units)
    if not units or embeddings.size == 0:
        return {"manual_unit_id": "", "manual_week": "", "topic_id": "", "score": ""}

    qvec = encode_texts([text])
    if qvec.size == 0:
        return {"manual_unit_id": "", "manual_week": "", "topic_id": "", "score": ""}

    candidate_indices = [idx for idx, unit in enumerate(units) if not topic_id or unit.get("topic_id") == topic_id]
    if not candidate_indices:
        candidate_indices = list(range(len(units)))

    sims = embeddings[candidate_indices].dot(qvec[0])
    best_local = int(np.argmax(sims))
    best_idx = candidate_indices[best_local]
    best = units[best_idx]
    best_score = float(sims[best_local])

    min_similarity = get_float("topic_matching", "manual_unit_min_similarity", 0.28)
    if best_score < min_similarity:
        return {"manual_unit_id": "", "manual_week": "", "topic_id": "", "score": ""}
    return {
        "manual_unit_id": best["manual_unit_id"],
        "manual_week": best.get("manual_week", ""),
        "topic_id": best.get("topic_id", ""),
        "score": f"{best_score:.4f}",
    }


def query_topic_evidence(
    topic_label: str,
    topk: Optional[int] = None,
    weight_doc: Optional[float] = None,
    weight_topic: Optional[float] = None,
) -> list[dict[str, Any]]:
    if topk is None:
        topk = get_int("transcript_export", "topic_query_topk", 120)
    if weight_doc is None:
        weight_doc = get_float("transcript_export", "topic_query_weight_doc", 0.3)
    if weight_topic is None:
        weight_topic = get_float("transcript_export", "topic_query_weight_topic", 0.7)
    module = load_source_query_module()
    return module.query_index_weighted(
        str(SOURCE_RAG_INDEX),
        topic_label,
        topk=topk,
        weight_doc=weight_doc,
        weight_topic=weight_topic,
    )


def query_evidence(
    query_text: str,
    topk: Optional[int] = None,
    weight_doc: Optional[float] = None,
    weight_topic: Optional[float] = None,
    cycle_id: str = "",
    transcript_only: bool = True,
    manual_only: bool = False,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    if topk is None:
        topk = get_int("cycle_analysis", "fidelity_topk", 12)
    if weight_doc is None:
        weight_doc = get_float("cycle_analysis", "fidelity_weight_doc", 0.5)
    if weight_topic is None:
        weight_topic = get_float("cycle_analysis", "fidelity_weight_topic", 0.5)
    import numpy as _np

    # If caller requested canonical manual-only retrieval, search the
    # pre-built manual-unit embedding index directly and return the top-k
    # canonical units. This keeps RAG retrieval semantics unchanged for
    # non-manual flows while ensuring manual-only mode returns authoritative
    # `MAN_xxxx` units.
    if manual_only:
        units, unit_emb = get_manual_unit_embedding_index()
        if unit_emb is None or unit_emb.size == 0 or not units:
            return []
        # Use the same embedding model used elsewhere to encode the query.
        model = get_embedding_model()
        qvec = model.encode([query_text], convert_to_numpy=True)[0].astype(_np.float32)
        # Normalize query vector and compute cosine sims with unit embeddings
        qn = qvec / (_np.linalg.norm(qvec) or 1.0)
        sims = unit_emb.dot(qn)
        idxs = _np.argsort(-sims)[: int(topk) if topk is not None else len(units)]
        results: list[dict[str, Any]] = []
        for rank, loc in enumerate(idxs, start=1):
            i = int(loc)
            u = units[i]
            sim = float(sims[i])
            mw = str(u.get("manual_week", "") or "")
            results.append(
                {
                    "rank": rank,
                    "doc_index": i,
                    "file": str(SOURCE_MANUAL),
                    "text": u.get("matching_text", u.get("text", "")),
                    "score_doc": sim,
                    "score_topic": None,
                    "score_combined": sim,
                    "cycle_id": "",
                    "session_id": mw or "manual",
                    "speaker": "manual",
                    "source_type": "manual",
                    "manual_unit_id_best_match": u.get("manual_unit_id", ""),
                    "manual_unit_match_score": f"{sim:.4f}",
                    "manual_week": mw,
                    "manual_session": (f"Session {mw}" if mw else ""),
                    "evidence_source": "canonical_manual_unit_index",
                }
            )
        return results

    module = load_source_query_module()
    meta, emb_array = module._load_meta_and_embeddings(str(SOURCE_RAG_INDEX))
    if emb_array is None:
        raise RuntimeError("embeddings.npy required for weighted query")

    model = module._get_model(model_name)
    qvec = model.encode([query_text], convert_to_numpy=True)[0].astype(_np.float32)

    candidates: list[int] = []
    for idx, row in enumerate(meta):
        path = str(row.get("path", "") or "")
        if cycle_id and cycle_id not in path:
            continue
        # If caller requests manual-only, skip any non-manual rows at the index level.
        if manual_only and not is_manual_row(row):
            continue
        # If caller requested transcript-only, skip manual rows.
        if transcript_only and is_manual_row(row):
            continue
        candidates.append(idx)

    if not candidates:
        return []

    doc_emb = emb_array[candidates].astype(_np.float32)

    def row_norm(a):
        n = _np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n

    doc_norm = row_norm(doc_emb)
    qn = qvec / (_np.linalg.norm(qvec) or 1.0)
    sims_doc = doc_norm.dot(qn)

    use_topic_scores = weight_topic > 0
    topic_emb = None
    topic_list = None
    sims_topic = _np.zeros_like(sims_doc)
    doc_topic_sims = None
    if use_topic_scores:
        topic_emb, topic_list = module._load_topic_embeddings(str(SOURCE_RAG_INDEX))
    if topic_emb is not None:
        topic_norm = row_norm(topic_emb.astype(_np.float32))
        doc_topic_sims = doc_norm.dot(topic_norm.T)
        best_topic_idx = doc_topic_sims.argmax(axis=1)
        topic_for_doc = topic_norm[best_topic_idx]
        sims_topic = topic_for_doc.dot(qn)

    def z(x):
        m = x.mean()
        s = x.std() or 1.0
        return (x - m) / s

    combined = weight_doc * z(sims_doc) + weight_topic * z(sims_topic)
    idxs_local = _np.argsort(-combined)[:topk].tolist()

    results = []
    for rank, loc in enumerate(idxs_local, start=1):
        glob_idx = candidates[loc]
        row = meta[glob_idx]
        result = {
            "rank": rank,
            "doc_index": glob_idx,
            "file": row.get("path") or row.get("source", ""),
            "text": (row.get("text", "") or "").replace("\n", " "),
            "score_doc": float(sims_doc[loc]),
            "score_topic": float(sims_topic[loc]) if topic_emb is not None else None,
            "score_combined": float(combined[loc]),
            "cycle_id": infer_cycle_id(str(row.get("path", "") or "")),
            "session_id": infer_session_id(str(row.get("path", "") or "")),
            "speaker": row.get("speaker", ""),
            "source_type": "manual" if is_manual_row(row) else "transcript",
        }
        if topic_emb is not None and doc_topic_sims is not None:
            best_idx = int(doc_topic_sims[loc].argmax())
            result["assigned_topic"] = topic_list[best_idx] if topic_list and 0 <= best_idx < len(topic_list) else None
        results.append(result)
    return results


def get_topic_entries_for_session(session_num: str) -> list[dict[str, str]]:
    target = str(session_num or "")
    return [topic for topic in load_topic_entries() if str(topic.get("session_num", "")) == target]


def get_manual_units_for_session(session_num: str, topic_id: str = "") -> list[dict[str, Any]]:
    units = build_manual_unit_index()
    target = str(session_num or "")
    filtered = [unit for unit in units if str(unit.get("manual_week", "")) == target]
    if topic_id:
        topic_filtered = [unit for unit in filtered if str(unit.get("topic_id", "")) == str(topic_id)]
        if topic_filtered:
            return topic_filtered
    return filtered


def build_doc_index_by_path(meta_rows: Optional[list[dict[str, Any]]] = None) -> dict[str, list[int]]:
    rows = meta_rows or get_rag_index_rows()
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        path = str(row.get("path", "") or "")
        grouped[path].append(idx)
    return grouped


def expand_transcript_context(
    doc_index: int,
    meta_rows: Optional[list[dict[str, Any]]] = None,
    path_lookup: Optional[dict[str, list[int]]] = None,
    window: int = 2,
) -> dict[str, Any]:
    rows = meta_rows or get_rag_index_rows()
    if not (0 <= doc_index < len(rows)):
        return {"text": "", "turn_count": 0, "path": "", "source": ""}
    row = rows[doc_index]
    path = str(row.get("path", "") or "")
    if not path:
        return {
            "text": str(row.get("text", "") or ""),
            "turn_count": 1,
            "path": path,
            "source": row.get("source", ""),
        }
    by_path = path_lookup or build_doc_index_by_path(rows)
    indices = by_path.get(path, [])
    if doc_index not in indices:
        return {
            "text": str(row.get("text", "") or ""),
            "turn_count": 1,
            "path": path,
            "source": row.get("source", ""),
        }
    loc = indices.index(doc_index)
    start = max(loc - window, 0)
    end = min(loc + window + 1, len(indices))
    selected = [rows[i] for i in indices[start:end] if not is_manual_row(rows[i])]
    parts = []
    for item in selected:
        speaker = str(item.get("speaker", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        parts.append(f"{speaker}: {text}" if speaker else text)
    return {
        "text": "\n".join(parts),
        "turn_count": len(selected),
        "path": path,
        "source": row.get("source", ""),
        "speaker": row.get("speaker", ""),
    }


def load_speaker_role_map() -> dict[tuple[str, str], str]:
    if not SPEAKER_ROLE_MAP_CSV.exists():
        return {}
    with open(SPEAKER_ROLE_MAP_CSV, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (row.get("session_id", ""), row.get("speaker_label", "")): row.get("assigned_role", "")
            for row in rows
            if row.get("session_id") and row.get("speaker_label")
        }
