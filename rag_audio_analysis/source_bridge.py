import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
import numpy as np

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
from rag_audio_analysis.settings import get_float, get_int


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

SESSION_FIDELITY_PRIORITY_CUES = {
    "1": "introductions of group leaders and participants, orientation to the 12-week parent stress and health group, group rules, confidentiality, phone use, attendance expectations, binder use, participant goals, stressors, the connection between stress, parenting, health, nutrition, and physical activity, the brain-in-your-hand model, guided breathing, Fitbit use, step goals, and home practice expectations",
    "2": "review of stress homework, bodily sensations, reactions, the four buckets of experience, mindfulness as present-moment nonjudgmental attention, autopilot, monkey mind, the stress reaction cycle, reacting versus responding, the STOP skill, the raisin activity, mindful eating, guided breathing, sugar-sweetened beverages, healthier drink choices, and beverage-related goals",
    "3": "review of homework on autopilot and mindfulness, mindful eating, hints for mindful eating, the seven types of hunger, cravings, food choices, body awareness, stretch practice, body scan, visually appealing meals, SMART goals, GO SLOW WHOA food guidance, and physical activity recommendations",
    "4": "review of the seven types of hunger, visually appealing meals, SMART goals, body scan, seated yoga, stress and unpleasant events, emotional reactivity, impact on a young child, mindfulness of thoughts, feelings, sensations, and mood, pleasant activities, pleasant and unpleasant event tracking, portion size, MyPlate, and portion-control strategies",
    "5": "pleasant and unpleasant events homework, how planned pleasant activities affect mood, stress, and self-care, parenting-related stress, reacting versus responding with a child, noticing bodily sensations thoughts feelings and urges, sound meditation, pleasant-events guided imagery, parenting stressors, reactions and responses logs, micronutrients, and family physical activity goals",
    "6": "reacting versus responding homework, parenting triggers, negative emotions, communication stress, awareness during difficult conversations, passive aggressive passive-aggressive and assertive communication, mindful attention during communication, singing bowl meditation, standing body scan, breakfast habits, barriers to breakfast, family activity goals, and SMART or FITT exercise planning",
    "7": "communication homework, reflection on a difficult communication experience, stressful-experience imagery, the seven attitudes of mindfulness, non-judging, patience, beginner's mind, trust, non-striving, acceptance, letting go, lower-leg body scan, mindfulness checklist, mindfulness activity log, food journaling with hunger and fullness, healthy recipe substitution, and food label reading",
    "8": "mindfulness attitudes, mindfulness activity logs, food journaling, stress as an iceberg, mindful parenting, parenting on autopilot versus mindful parenting, children responding to parental stress, the ONESIE framework, standing yoga, guided imagery, parenting journal, mindfulness with a child, trying a new fruit or vegetable, meal planning on a budget, and family fitness goals",
    "9": "mindfulness practice logs, parenting logs, food journals, changes in responses to stress parenting communication and self-care, staying mindful during stress, observing and naming internal experiences, accepting difficult emotions, mindfulness in parenting and communication, the Face the FEAR framework, walking meditation, breath pacing, fruits and vegetables, colors of the rainbow, and family-based physical activity goals",
    "10": "mindfulness activity practice, how often and under what conditions participants use mindfulness on their own and with their children, acceptance as an active mindfulness process, fear anxiety shame and guilt during stress, noticing thoughts feelings sensations and urges without judgment, the ACCEPT framework, slowing the breath, tolerating discomfort, observing urges, mindful eating with choices, cravings, food decisions, recipe substitutions, and creative physical activity planning",
    "11": "mindfulness activity practice, food journals, food cravings, autopilot eating, seven types of hunger, environmental cues, awareness and acceptance of stress emotions and urges, observing cravings without acting, seated floor stretch, food-craving imagery, Svasana, the STOP Food Cravings handout, healthier fast food choices, chronic illness risk, and goals to reduce fast food intake",
    "12": "review of mindfulness practice, cravings, mindful eating, yoga, closing reflection on change across the 12-week group, the four components of experience, slowing down to respond rather than react, non-action, non-striving, continuing mindfulness after the group for self and child, changes in eating habits cravings and family food routines, Fitbit use, original goals, future goals, standing yoga, and summing up helpful nutrition and exercise strategies",
}


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


def build_session_fidelity_query(
    session_num: str,
    session_summary: str,
    topic_labels: Optional[list[str]] = None,
) -> str:
    session_num = str(session_num or "").strip()
    session_summary = str(session_summary or "").strip()
    labels = [str(label or "").strip() for label in (topic_labels or []) if str(label or "").strip()]
    priority_cues = SESSION_FIDELITY_PRIORITY_CUES.get(session_num, "")
    if not priority_cues and labels:
        priority_cues = ", ".join(labels)

    parts = [f"Manual Session {session_num}. Retrieve transcript evidence specific to this manual session only."]
    if priority_cues:
        parts.extend(
            [
                "",
                "Prioritize evidence about:",
                priority_cues,
            ]
        )
    parts.extend(
        [
            "",
            "Avoid evidence that is primarily about generic themes from other sessions unless it is clearly tied to this session summary.",
        ]
    )
    if session_summary:
        parts.extend(
            [
                "",
                "Session summary:",
                session_summary,
            ]
        )
    elif labels:
        parts.extend(
            [
                "",
                "Session topics:",
                "; ".join(labels),
            ]
        )
    return "\n".join(parts).strip()


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
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    if topk is None:
        topk = get_int("cycle_analysis", "fidelity_topk", 12)
    if weight_doc is None:
        weight_doc = get_float("cycle_analysis", "fidelity_weight_doc", 0.5)
    if weight_topic is None:
        weight_topic = get_float("cycle_analysis", "fidelity_weight_topic", 0.5)
    import numpy as _np

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

    topic_emb, topic_list = module._load_topic_embeddings(str(SOURCE_RAG_INDEX))
    sims_topic = _np.zeros_like(sims_doc)
    doc_topic_sims = None
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
