#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR
from rag_audio_analysis.settings import get_float, get_int, get_list, get_str
from rag_audio_analysis.source_bridge import (
    build_session_fidelity_query,
    build_doc_index_by_path,
    expand_transcript_context,
    get_manual_units_for_session,
    get_rag_index_rows,
    get_session_summary,
    get_topic_entries_for_session,
    infer_manual_unit_for_text,
    infer_session_id,
    load_topic_entries,
    query_evidence,
)

PI_QUESTION_SPECS = [
    {
        "question_id": "facilitator_reference",
        "label": get_str("pi_questions", "facilitator_reference_label", "How often do facilitators refer to this topic?"),
        "query_template": get_str("pi_questions", "facilitator_reference_query", "Session {session_num} {topic_label}. Facilitator introducing, teaching, reviewing, or cueing this topic."),
    },
    {
        "question_id": "facilitator_demonstration",
        "label": get_str("pi_questions", "facilitator_demonstration_label", "How do facilitators demonstrate this skill?"),
        "query_template": get_str("pi_questions", "facilitator_demonstration_query", "Session {session_num} {topic_label}. Facilitator modeling, guiding, demonstrating, or leading practice of this skill."),
    },
    {
        "question_id": "participant_practice",
        "label": get_str("pi_questions", "participant_practice_label", "How do participants practice this skill in the session or individually at home?"),
        "query_template": get_str("pi_questions", "participant_practice_query", "Session {session_num} {topic_label}. Participant describing practicing this skill in session or individually at home."),
    },
    {
        "question_id": "participant_child_home",
        "label": get_str("pi_questions", "participant_child_home_label", "What do participants share about practicing this skill with their child at home?"),
        "query_template": get_str("pi_questions", "participant_child_home_query", "Session {session_num} {topic_label}. Participant describing using this skill with their child at home."),
    },
]

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def call_ollama(
    prompt: str,
    model: str,
    ollama_ssh_host: str = "",
    ollama_ssh_key: str = "",
    ollama_remote_bin: str = "/usr/local/bin/ollama",
) -> str:
    if ollama_ssh_host:
        ssh_cmd = ["ssh"]
        expanded_key = os.path.expanduser(ollama_ssh_key.strip())
        if expanded_key:
            ssh_cmd.extend(["-i", expanded_key])
        ssh_cmd.extend(
            [
                "-o",
                "BatchMode=yes",
                ollama_ssh_host,
                f"{shlex.quote(ollama_remote_bin)} run {shlex.quote(model)}",
            ]
        )
        cmd = ssh_cmd
    else:
        cmd = ["ollama", "run", model]

    proc = subprocess.run(
        cmd,
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.decode("utf-8").strip()


def parse_json_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {"raw_response": text}
        return {"raw_response": text}


def format_evidence_excerpt(text: str, limit: int | None = None) -> str:
    if limit is None:
        limit = get_int("prompting", "display_excerpt_chars", 500)
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


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
        return format_evidence_excerpt(session_summary, 320)
    if not session_summary:
        return f"Focuses on {normalize_topic_label(topic_label)} in this session."

    context_fragments = choose_topic_context_fragments(topic_label, session_summary)
    context_text = "; ".join(context_fragments).strip() or clean_summary_fragment(session_summary)
    gloss = f"Focuses on {normalize_topic_label(topic_label)} in the context of {context_text}."
    gloss = re.sub(r"\s+", " ", gloss).strip()
    return format_evidence_excerpt(gloss, 320)


def build_pi_question_query(
    session_num: str,
    topic: dict[str, str],
    question_spec: dict[str, str],
) -> str:
    topic_label = str(topic.get("label", "")).strip()
    session_summary_row = get_session_summary(session_num)
    session_summary = str(session_summary_row.get("session_summary", "")).strip()
    topic_definition = derive_topic_definition(topic_label, session_summary)
    retrieval_focus = question_spec["query_template"].format(session_num=session_num, topic_label=topic_label).strip()

    parts = [
        f"Session {session_num}. Topic: {topic_label}.",
        "",
        "Topic definition:",
        topic_definition,
        "",
        "Question:",
        question_spec["label"],
        "",
        "Retrieval focus:",
        retrieval_focus,
    ]
    if session_summary:
        parts.extend(
            [
                "",
                "Session summary context:",
                format_evidence_excerpt(session_summary, 500),
            ]
        )
    return "\n".join(parts).strip()


def build_transcript_windows(
    query_text: str,
    topic_id: str,
    session_num: str,
    cycle_id: str,
    topk: int,
    weight_doc: float,
    weight_topic: float,
    meta_rows: list[dict[str, Any]],
    path_lookup: dict[str, list[int]],
    window: int,
    manual_units: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    raw_results = query_evidence(
        query_text,
        topk=topk,
        weight_doc=weight_doc,
        weight_topic=weight_topic,
        cycle_id=cycle_id,
        transcript_only=True,
    )

    session_manual_units = manual_units if manual_units is not None else get_manual_units_for_session(session_num, topic_id=topic_id)
    windows: list[dict[str, Any]] = []
    seen_doc_indices: set[int] = set()

    for result in raw_results:
        doc_index = int(result.get("doc_index", -1))
        if doc_index < 0 or doc_index in seen_doc_indices:
            continue
        seen_doc_indices.add(doc_index)
        context = expand_transcript_context(doc_index, meta_rows=meta_rows, path_lookup=path_lookup, window=window)
        text = str(context.get("text", "") or "").strip()
        if not text:
            continue
        manual_match = infer_manual_unit_for_text(text, topic_id=topic_id, manual_units=session_manual_units)
        windows.append(
            {
                "doc_index": doc_index,
                "retrieval_rank": result.get("rank", ""),
                "session_id": infer_session_id(str(context.get("path", "") or result.get("file", ""))),
                "cycle_id": result.get("cycle_id", cycle_id),
                "speaker": context.get("speaker", "") or result.get("speaker", ""),
                "score_combined": result.get("score_combined"),
                "score_doc": result.get("score_doc"),
                "score_topic": result.get("score_topic"),
                "text": text,
                "manual_unit_id_best_match": manual_match.get("manual_unit_id", ""),
                "manual_topic_id_best_match": manual_match.get("topic_id", ""),
                "manual_unit_match_score": manual_match.get("score", ""),
            }
        )
    return windows


def build_session_fidelity_windows(
    cycle_id: str,
    session_num: str,
    session_summary: str,
    session_topics: list[dict[str, str]],
    topk: int,
    weight_doc: float,
    weight_topic: float,
    meta_rows: list[dict[str, Any]],
    path_lookup: dict[str, list[int]],
    window: int,
) -> list[dict[str, Any]]:
    session_manual_units = get_manual_units_for_session(session_num)
    dynamic_topk = len(session_manual_units) or topk
    topic_labels = [str(topic.get("label", "")).strip() for topic in session_topics if str(topic.get("label", "")).strip()]
    query_text = build_session_fidelity_query(session_num, session_summary, topic_labels)
    return build_transcript_windows(
        query_text=query_text,
        topic_id="",
        session_num=session_num,
        cycle_id=cycle_id,
        topk=dynamic_topk,
        weight_doc=weight_doc,
        weight_topic=weight_topic,
        meta_rows=meta_rows,
        path_lookup=path_lookup,
        window=window,
        manual_units=session_manual_units,
    )


def summarize_fidelity(
    cycle_id: str,
    session_num: str,
    topic: dict[str, str],
    windows: list[dict[str, Any]],
    manual_units: list[dict[str, Any]],
) -> dict[str, str]:
    expected_ids = {unit["manual_unit_id"] for unit in manual_units}
    observed_ids = {w["manual_unit_id_best_match"] for w in windows if w.get("manual_unit_id_best_match") in expected_ids}
    expected_subsections = {unit.get("manual_subsection", "") for unit in manual_units if unit.get("manual_subsection")}
    observed_subsections = {
        unit.get("manual_subsection", "")
        for unit in manual_units
        if unit.get("manual_unit_id") in observed_ids and unit.get("manual_subsection")
    }

    manual_cov = (len(observed_ids) / len(expected_ids)) if expected_ids else 0.0
    subsection_cov = (len(observed_subsections) / len(expected_subsections)) if expected_subsections else 0.0
    manual_weight = get_float("fidelity", "manual_coverage_weight", 0.6)
    subsection_weight = get_float("fidelity", "subsection_coverage_weight", 0.4)
    high_cutoff = get_float("fidelity", "adherence_high_cutoff", 0.66)
    moderate_cutoff = get_float("fidelity", "adherence_moderate_cutoff", 0.33)
    combined_score = manual_weight * manual_cov + subsection_weight * subsection_cov
    if combined_score >= high_cutoff:
        adherence = "high"
    elif combined_score >= moderate_cutoff:
        adherence = "moderate"
    else:
        adherence = "low"

    return {
        "cycle_id": cycle_id,
        "session_num": session_num,
        "session_label": topic.get("session_label", f"Session {session_num}"),
        "topic_id": topic.get("id", ""),
        "topic_label": topic.get("label", ""),
        "fidelity_query": f"Session {session_num} {topic.get('label', '')}".strip(),
        "retrieved_evidence_count": str(len(windows)),
        "expected_manual_unit_count": str(len(expected_ids)),
        "matched_manual_unit_count": str(len(observed_ids)),
        "manual_unit_coverage": f"{manual_cov:.3f}",
        "expected_subsection_count": str(len(expected_subsections)),
        "matched_subsection_count": str(len(observed_subsections)),
        "subsection_coverage": f"{subsection_cov:.3f}",
        "adherence_score": f"{combined_score:.3f}",
        "adherence_label": adherence,
        "matched_manual_unit_ids": ";".join(sorted(observed_ids)),
        "matched_subsections": ";".join(sorted(observed_subsections)),
        "sample_session_ids": ";".join(sorted({w.get('session_id', '') for w in windows if w.get('session_id')})),
    }


def summarize_session_fidelity(
    cycle_id: str,
    session_num: str,
    topics: list[dict[str, str]],
    windows: list[dict[str, Any]],
    manual_units: list[dict[str, Any]],
) -> dict[str, str]:
    session_summary_row = get_session_summary(session_num)
    expected_ids = {unit["manual_unit_id"] for unit in manual_units}
    observed_ids = {w["manual_unit_id_best_match"] for w in windows if w.get("manual_unit_id_best_match") in expected_ids}
    expected_subsections = {unit.get("manual_subsection", "") for unit in manual_units if unit.get("manual_subsection")}
    observed_subsections = {
        unit.get("manual_subsection", "")
        for unit in manual_units
        if unit.get("manual_unit_id") in observed_ids and unit.get("manual_subsection")
    }

    manual_cov = (len(observed_ids) / len(expected_ids)) if expected_ids else 0.0
    subsection_cov = (len(observed_subsections) / len(expected_subsections)) if expected_subsections else 0.0
    evidence_density = (len(windows) / len(expected_ids)) if expected_ids else 0.0
    manual_weight = get_float("fidelity", "manual_coverage_weight", 0.6)
    subsection_weight = get_float("fidelity", "subsection_coverage_weight", 0.4)
    high_cutoff = get_float("fidelity", "adherence_high_cutoff", 0.66)
    moderate_cutoff = get_float("fidelity", "adherence_moderate_cutoff", 0.33)
    combined_score = manual_weight * manual_cov + subsection_weight * subsection_cov
    if combined_score >= high_cutoff:
        adherence = "high"
    elif combined_score >= moderate_cutoff:
        adherence = "moderate"
    else:
        adherence = "low"

    topic_ids = [str(topic.get("id", "")).strip() for topic in topics if str(topic.get("id", "")).strip()]
    topic_labels = [str(topic.get("label", "")).strip() for topic in topics if str(topic.get("label", "")).strip()]
    session_summary = session_summary_row.get("session_summary", "")
    fidelity_query = build_session_fidelity_query(session_num, session_summary, topic_labels)

    return {
        "cycle_id": cycle_id,
        "manual_session_num": session_num,
        "manual_session_label": session_summary_row.get("session_label", f"Session {session_num}"),
        "fidelity_query": fidelity_query,
        "session_summary": session_summary,
        "session_topic_ids": ";".join(topic_ids),
        "session_topic_labels": ";".join(topic_labels),
        "retrieved_evidence_count": str(len(windows)),
        "expected_manual_unit_count": str(len(expected_ids)),
        "matched_manual_unit_count": str(len(observed_ids)),
        "manual_unit_coverage": f"{manual_cov:.3f}",
        "expected_subsection_count": str(len(expected_subsections)),
        "matched_subsection_count": str(len(observed_subsections)),
        "subsection_coverage": f"{subsection_cov:.3f}",
        "evidence_density": f"{evidence_density:.3f}",
        "adherence_score": f"{combined_score:.3f}",
        "adherence_label": adherence,
        "matched_manual_unit_ids": ";".join(sorted(observed_ids)),
        "matched_subsections": ";".join(sorted(observed_subsections)),
        "sample_session_ids": ";".join(sorted({w.get('session_id', '') for w in windows if w.get('session_id')})),
    }


def build_question_prompt(
    cycle_id: str,
    session_num: str,
    topic: dict[str, str],
    question_spec: dict[str, str],
    evidence_rows: list[dict[str, Any]],
    manual_units: list[dict[str, Any]],
) -> str:
    manual_unit_limit = get_int("prompting", "manual_units_in_prompt", 6)
    manual_excerpt_chars = get_int("prompting", "manual_excerpt_chars", 220)
    evidence_excerpt_chars = get_int("prompting", "evidence_excerpt_chars", 400)

    parts = [
        "You are analyzing intervention transcripts with retrieved evidence only.",
        "Return valid JSON only.",
        "",
        f"Cycle: {cycle_id}",
        f"Session: Session {session_num}",
        f"Topic: {topic.get('label', '')}",
        f"Question: {question_spec['label']}",
        "",
        "Relevant manual units for this session/topic:",
    ]
    if manual_units:
        for unit in manual_units[:manual_unit_limit]:
            parts.append(
                f"- {unit['manual_unit_id']} | {unit.get('manual_subsection', '')} | "
                f"{format_evidence_excerpt(unit.get('text', ''), manual_excerpt_chars)}"
            )
    else:
        parts.append("- No matching manual units were available.")
    parts.append("")
    parts.append("Retrieved transcript evidence:")
    for idx, row in enumerate(evidence_rows, start=1):
        parts.append(
            f"- E{idx} | session_id={row.get('session_id','')} | "
            f"manual_unit={row.get('manual_unit_id_best_match','')} | "
            f"{format_evidence_excerpt(row.get('text', ''), evidence_excerpt_chars)}"
        )
    parts.extend(
        [
            "",
            "Respond with JSON with exactly these keys:",
            '{',
            '  "answer_summary": "short paragraph",',
            '  "evidence_count": 0,',
            '  "evidence_refs": ["E1"],',
            '  "manual_unit_ids": ["MAN_0001"],',
            '  "confidence": "low|medium|high"',
            '}',
            "Base the answer only on the retrieved evidence and manual units shown above. If evidence is weak, say so in the summary.",
        ]
    )
    return "\n".join(parts)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="+", default=get_list("cycle_analysis", "cycles", ["1", "2", "3", "4", "5"]), help="Cycle numbers to analyze, e.g. 1 2 3")
    parser.add_argument("--fidelity-topk", type=int, default=get_int("cycle_analysis", "fidelity_topk", 12))
    parser.add_argument("--question-topk", type=int, default=get_int("cycle_analysis", "question_topk", 8))
    parser.add_argument("--fidelity-weight-doc", type=float, default=get_float("cycle_analysis", "fidelity_weight_doc", 0.5))
    parser.add_argument("--fidelity-weight-topic", type=float, default=get_float("cycle_analysis", "fidelity_weight_topic", 0.5))
    parser.add_argument("--question-weight-doc", type=float, default=get_float("cycle_analysis", "question_weight_doc", 1.0))
    parser.add_argument("--question-weight-topic", type=float, default=get_float("cycle_analysis", "question_weight_topic", 0.0))
    parser.add_argument("--context-window", type=int, default=get_int("transcript_export", "context_window", 2))
    parser.add_argument("--ollama-model", default="", help="Optional Ollama model for automated PI-question summaries")
    parser.add_argument(
        "--ollama-ssh-host",
        default=get_str("ollama", "ssh_host", "rc2526@10.168.224.148"),
        help="Optional SSH target for remote Ollama execution",
    )
    parser.add_argument(
        "--ollama-ssh-key",
        default=get_str("ollama", "ssh_key", "~/.ssh/ollama_remote"),
        help="SSH private key used to reach the remote Ollama host",
    )
    parser.add_argument(
        "--ollama-remote-bin",
        default=get_str("ollama", "remote_bin", "/usr/local/bin/ollama"),
        help="Path to the Ollama binary on the remote host",
    )
    parser.add_argument("--limit-topics", type=int, default=get_int("cycle_analysis", "limit_topics", 0), help="Optional limit for debugging")
    args = parser.parse_args()

    ensure_dir(CYCLE_ANALYSIS_DIR)
    meta_rows = get_rag_index_rows()
    path_lookup = build_doc_index_by_path(meta_rows)
    session_numbers = sorted({str(topic.get("session_num", "")) for topic in load_topic_entries() if str(topic.get("session_num", ""))})

    for cycle_num in args.cycles:
        cycle_id = f"PMHCycle{cycle_num}"
        cycle_dir = CYCLE_ANALYSIS_DIR / cycle_id
        ensure_dir(cycle_dir)

        fidelity_rows: list[dict[str, Any]] = []
        session_fidelity_rows: list[dict[str, Any]] = []
        question_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        session_fidelity_evidence_rows: list[dict[str, Any]] = []
        question_json_rows: list[dict[str, Any]] = []

        topic_count = 0
        for session_num in session_numbers:
            session_topics = get_topic_entries_for_session(session_num)

            for topic in session_topics:
                topic_count += 1
                if args.limit_topics and topic_count > args.limit_topics:
                    break
                topic_id = topic.get("id", "")
                topic_label = topic.get("label", "")
                session_manual_units = get_manual_units_for_session(session_num, topic_id=topic_id)

                fidelity_query = get_str("fidelity", "fidelity_query_template", "Session {session_num} {topic_label}").format(
                    session_num=session_num,
                    topic_label=topic_label,
                ).strip()
                fidelity_windows = build_transcript_windows(
                    fidelity_query,
                    topic_id=topic_id,
                    session_num=session_num,
                    cycle_id=cycle_id,
                    topk=args.fidelity_topk,
                    weight_doc=args.fidelity_weight_doc,
                    weight_topic=args.fidelity_weight_topic,
                    meta_rows=meta_rows,
                    path_lookup=path_lookup,
                    window=args.context_window,
                )
                fidelity_rows.append(summarize_fidelity(cycle_id, session_num, topic, fidelity_windows, session_manual_units))

                for row in fidelity_windows:
                    evidence_rows.append(
                        {
                            "cycle_id": cycle_id,
                            "session_num": session_num,
                            "topic_id": topic_id,
                            "topic_label": topic_label,
                            "analysis_mode": "fidelity",
                            "question_id": "",
                            "query_text": fidelity_query,
                            "retrieval_rank": row.get("retrieval_rank", ""),
                            "session_id": row.get("session_id", ""),
                            "speaker": row.get("speaker", ""),
                            "score_combined": row.get("score_combined", ""),
                            "score_doc": row.get("score_doc", ""),
                            "score_topic": row.get("score_topic", ""),
                            "manual_unit_id_best_match": row.get("manual_unit_id_best_match", ""),
                            "manual_unit_match_score": row.get("manual_unit_match_score", ""),
                            "text": row.get("text", ""),
                                "excerpt": format_evidence_excerpt(row.get("text", "")),
                            }
                        )

                for question_spec in PI_QUESTION_SPECS:
                    query_text = build_pi_question_query(session_num, topic, question_spec)
                    question_windows = build_transcript_windows(
                        query_text,
                        topic_id=topic_id,
                        session_num=session_num,
                        cycle_id=cycle_id,
                        topk=args.question_topk,
                        weight_doc=args.question_weight_doc,
                        weight_topic=args.question_weight_topic,
                        meta_rows=meta_rows,
                        path_lookup=path_lookup,
                        window=args.context_window,
                    )
                    answer_payload: dict[str, Any] = {}
                    prompt = ""
                    if args.ollama_model and question_windows:
                        prompt = build_question_prompt(cycle_id, session_num, topic, question_spec, question_windows, session_manual_units)
                        answer_payload = parse_json_response(
                            call_ollama(
                                prompt,
                                args.ollama_model,
                                ollama_ssh_host=args.ollama_ssh_host,
                                ollama_ssh_key=args.ollama_ssh_key,
                                ollama_remote_bin=args.ollama_remote_bin,
                            )
                        )

                    answer_summary = answer_payload.get("answer_summary", "")
                    confidence = answer_payload.get("confidence", "")
                    evidence_refs = ";".join(answer_payload.get("evidence_refs", [])) if isinstance(answer_payload.get("evidence_refs"), list) else ""
                    manual_ids = ";".join(answer_payload.get("manual_unit_ids", [])) if isinstance(answer_payload.get("manual_unit_ids"), list) else ""
                    question_rows.append(
                        {
                            "cycle_id": cycle_id,
                            "session_num": session_num,
                            "session_label": topic.get("session_label", f"Session {session_num}"),
                            "topic_id": topic_id,
                            "topic_label": topic_label,
                            "question_id": question_spec["question_id"],
                            "question_label": question_spec["label"],
                            "query_text": query_text,
                            "retrieved_evidence_count": str(len(question_windows)),
                            "prompt_text": prompt,
                            "answer_summary": answer_summary,
                            "confidence": confidence,
                            "evidence_refs": evidence_refs,
                            "manual_unit_ids": manual_ids,
                            "raw_response": json.dumps(answer_payload, ensure_ascii=False) if answer_payload else "",
                        }
                    )
                    question_json_rows.append(
                        {
                            "cycle_id": cycle_id,
                            "session_num": session_num,
                            "topic_id": topic_id,
                            "topic_label": topic_label,
                            "question_id": question_spec["question_id"],
                            "query_text": query_text,
                            "prompt_text": prompt,
                            "answer": answer_payload,
                            "evidence": question_windows,
                        }
                    )

                    for row in question_windows:
                        evidence_rows.append(
                            {
                                "cycle_id": cycle_id,
                                "session_num": session_num,
                                "topic_id": topic_id,
                                "topic_label": topic_label,
                                "analysis_mode": "pi_question",
                                "question_id": question_spec["question_id"],
                                "query_text": query_text,
                                "retrieval_rank": row.get("retrieval_rank", ""),
                                "session_id": row.get("session_id", ""),
                                "speaker": row.get("speaker", ""),
                                "score_combined": row.get("score_combined", ""),
                                "score_doc": row.get("score_doc", ""),
                                "score_topic": row.get("score_topic", ""),
                                "manual_unit_id_best_match": row.get("manual_unit_id_best_match", ""),
                                "manual_unit_match_score": row.get("manual_unit_match_score", ""),
                                "text": row.get("text", ""),
                                "excerpt": format_evidence_excerpt(row.get("text", "")),
                            }
                        )
            if args.limit_topics and topic_count >= args.limit_topics:
                break

            session_summary_row = get_session_summary(session_num)
            session_summary = session_summary_row.get("session_summary", "")
            session_topic_labels = [str(topic.get("label", "")).strip() for topic in session_topics if str(topic.get("label", "")).strip()]
            session_fidelity_query = build_session_fidelity_query(session_num, session_summary, session_topic_labels)
            session_manual_units = get_manual_units_for_session(session_num)
            session_fidelity_windows = build_session_fidelity_windows(
                cycle_id=cycle_id,
                session_num=session_num,
                session_summary=session_summary,
                session_topics=session_topics,
                topk=args.fidelity_topk,
                weight_doc=args.fidelity_weight_doc,
                weight_topic=args.fidelity_weight_topic,
                meta_rows=meta_rows,
                path_lookup=path_lookup,
                window=args.context_window,
            )
            session_fidelity_rows.append(
                summarize_session_fidelity(
                    cycle_id,
                    session_num,
                    session_topics,
                    session_fidelity_windows,
                    session_manual_units,
                )
            )
            for row in session_fidelity_windows:
                session_fidelity_evidence_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "manual_session_num": session_num,
                        "manual_session_label": f"Session {session_num}",
                        "analysis_mode": "session_fidelity",
                        "query_text": session_fidelity_query,
                        "source_topic_id": "",
                        "source_topic_label": "",
                        "retrieval_rank": row.get("retrieval_rank", ""),
                        "session_id": row.get("session_id", ""),
                        "speaker": row.get("speaker", ""),
                        "score_combined": row.get("score_combined", ""),
                        "score_doc": row.get("score_doc", ""),
                        "score_topic": row.get("score_topic", ""),
                        "manual_unit_id_best_match": row.get("manual_unit_id_best_match", ""),
                        "manual_unit_match_score": row.get("manual_unit_match_score", ""),
                        "text": row.get("text", ""),
                        "excerpt": format_evidence_excerpt(row.get("text", "")),
                    }
                )

        write_csv(
            cycle_dir / "fidelity_summary.csv",
            [
                "cycle_id",
                "session_num",
                "session_label",
                "topic_id",
                "topic_label",
                "fidelity_query",
                "retrieved_evidence_count",
                "expected_manual_unit_count",
                "matched_manual_unit_count",
                "manual_unit_coverage",
                "expected_subsection_count",
                "matched_subsection_count",
                "subsection_coverage",
                "adherence_score",
                "adherence_label",
                "matched_manual_unit_ids",
                "matched_subsections",
                "sample_session_ids",
            ],
            fidelity_rows,
        )
        write_csv(
            cycle_dir / "session_fidelity_summary.csv",
            [
                "cycle_id",
                "manual_session_num",
                "manual_session_label",
                "fidelity_query",
                "session_summary",
                "session_topic_ids",
                "session_topic_labels",
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
                "matched_manual_unit_ids",
                "matched_subsections",
                "sample_session_ids",
            ],
            session_fidelity_rows,
        )
        write_csv(
            cycle_dir / "pi_question_answers.csv",
            [
                "cycle_id",
                "session_num",
                "session_label",
                "topic_id",
                "topic_label",
                "question_id",
                "question_label",
                "query_text",
                "retrieved_evidence_count",
                "prompt_text",
                "answer_summary",
                "confidence",
                "evidence_refs",
                "manual_unit_ids",
                "raw_response",
            ],
            question_rows,
        )
        write_csv(
            cycle_dir / "session_fidelity_evidence.csv",
            [
                "cycle_id",
                "manual_session_num",
                "manual_session_label",
                "analysis_mode",
                "query_text",
                "source_topic_id",
                "source_topic_label",
                "retrieval_rank",
                "session_id",
                "speaker",
                "score_combined",
                "score_doc",
                "score_topic",
                "manual_unit_id_best_match",
                "manual_unit_match_score",
                "text",
                "excerpt",
            ],
            session_fidelity_evidence_rows,
        )
        write_csv(
            cycle_dir / "topic_evidence.csv",
            [
                "cycle_id",
                "session_num",
                "topic_id",
                "topic_label",
                "analysis_mode",
                "question_id",
                "query_text",
                "retrieval_rank",
                "session_id",
                "speaker",
                "score_combined",
                "score_doc",
                "score_topic",
                "manual_unit_id_best_match",
                "manual_unit_match_score",
                "text",
                "excerpt",
            ],
            evidence_rows,
        )
        (cycle_dir / "pi_question_answers.json").write_text(
            json.dumps(question_json_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote cycle analysis outputs to {cycle_dir}")


if __name__ == "__main__":
    main()
