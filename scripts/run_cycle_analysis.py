#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import subprocess
import re
from pathlib import Path
from typing import Any, Optional

from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR
from rag_audio_analysis.settings import get_float, get_int, get_list, get_str
from rag_audio_analysis.source_bridge import (
    build_doc_index_by_path,
    build_pi_query_text,
    build_session_fidelity_query,
    expand_transcript_context,
    get_manual_units_for_session,
    get_rag_index_rows,
    get_session_summary,
    get_topic_definition,
    get_topic_entries_for_session,
    infer_manual_unit_for_text,
    infer_session_id,
    load_topic_entries,
    query_evidence,
)

PI_QUESTION_SPECS = [
    {
        "question_id": "facilitator_delivery",
        "label": get_str("pi_questions", "facilitator_delivery_label", "How do facilitators introduce, reinforce, or demonstrate this topic or skill?"),
        "query_template": get_str("pi_questions", "facilitator_delivery_query", "Session {session_num} {topic_label}. Facilitator introducing, teaching, reviewing, cueing, modeling, guiding, demonstrating, or leading practice related to this topic or skill."),
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
        # Try to extract an embedded JSON object if present
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                # fall through to fallback extraction
                pass

        # If we couldn't parse JSON, attempt a lightweight heuristic to extract
        # a sensible fallback answer paragraph from the raw text so downstream
        # code can populate UI fields instead of a placeholder.
        def _extract_fallback_answer(s: str) -> str:
            s = s.strip()
            # Remove verbose thinking prefixes
            s = re.sub(r'(?i)^thinking[\.\s\n]*', '', s)
            # If model includes an explicit 'Answer:' marker, take what follows
            m = re.search(r'(?im)\banswer[:\-]\s*(.+)', s, flags=re.DOTALL)
            if m:
                candidate = m.group(1).strip()
            else:
                # Split into paragraphs and take the first substantive paragraph
                parts = [p.strip() for p in re.split(r'\n{2,}', s) if p.strip()]
                candidate = parts[0] if parts else s
            # Trim before common instruction-like markers (e.g., 'We have retrieved', 'E1:')
            candidate = re.split(r'\n(?:We have retrieved|Retrieved|E\d+:)', candidate)[0].strip()
            # Limit length
            if len(candidate) > 800:
                candidate = candidate[:800].rstrip() + "..."
            return candidate

        return {"raw_response": text, "fallback_answer": _extract_fallback_answer(text)}


def format_evidence_excerpt(text: str, limit: int | None = None) -> str:
    if limit is None:
        limit = get_int("prompting", "display_excerpt_chars", 500)
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


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
    topic_labels = [str(topic.get("label", "")).strip() for topic in session_topics if str(topic.get("label", "")).strip()]
    query_text = build_session_fidelity_query(
        session_num=session_num,
        session_summary=session_summary,
        topic_labels=topic_labels,
    )
    return build_transcript_windows(
        query_text=query_text,
        topic_id="",
        session_num=session_num,
        cycle_id=cycle_id,
        topk=topk,
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
    fidelity_query = build_session_fidelity_query(
        session_num=session_num,
        session_summary=session_summary,
        topic_labels=topic_labels,
    )

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
            '  "confidence": "low|medium|high",',
            '  "confidence_explanation": "short sentence explaining why that confidence was chosen; cite evidence refs if relevant",',
            '}',
            "Base the answer only on the retrieved evidence and manual units shown above. If evidence is weak, say so in the summary.",
            "Include inline evidence refs in the `answer_summary` next to the claims they support (for example: 'The parent described practicing mindful breathing with their child (E1)').",
            "For the confidence field include an explanatory sentence in `confidence_explanation` that references evidence refs (for example: 'medium — E2 and E4 show clear facilitator modeling; E5 is noisy').",
        ]
    )
    return "\n".join(parts)


def build_fidelity_adjudication_prompt(
    cycle_id: str,
    session_num: str,
    manual_units: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    *,
    scope_label: str,
    scope_description: str,
    include_all_manual_units: bool = False,
) -> str:
    manual_unit_limit = len(manual_units) if include_all_manual_units else get_int("prompting", "manual_units_in_prompt", 6)
    manual_excerpt_chars = get_int("prompting", "manual_excerpt_chars", 220)
    evidence_excerpt_chars = get_int("prompting", "evidence_excerpt_chars", 400)

    parts = [
        "You are adjudicating manual adherence from retrieved transcript evidence only.",
        "Return valid JSON only.",
        "",
        f"Cycle: {cycle_id}",
        f"Session: Session {session_num}",
        f"Scope: {scope_label}",
        f"Task: {scope_description}",
        "",
        "Expected manual units for this scope:",
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
    if evidence_rows:
        for idx, row in enumerate(evidence_rows, start=1):
            parts.append(
                f"- E{idx} | session_id={row.get('session_id','')} | "
                f"manual_unit={row.get('manual_unit_id_best_match','')} | "
                f"{format_evidence_excerpt(row.get('text', ''), evidence_excerpt_chars)}"
            )
    else:
        parts.append("- No retrieved evidence windows were available.")
    parts.extend(
        [
            "",
            "Respond with JSON with exactly these keys:",
            "{",
            '  "adjudication_summary": "short paragraph",',
            '  "adherence_label": "high|moderate|low",',
            '  "evidence_refs": ["E1"],',
            '  "manual_unit_ids": ["MAN_0001"],',
            '  "confidence": "low|medium|high"',
            "}",
            "Base the judgment only on the retrieved evidence and expected manual units shown above.",
            "Use 'low' adherence when evidence is weak or clearly incomplete.",
        ]
    )
    return "\n".join(parts)


def build_topic_fidelity_adjudication_prompt(
    cycle_id: str,
    session_num: str,
    topic: dict[str, str],
    manual_units: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> str:
    return build_fidelity_adjudication_prompt(
        cycle_id,
        session_num,
        manual_units,
        evidence_rows,
        scope_label=topic.get("label", ""),
        scope_description=(
            f"Judge how well the retrieved evidence reflects delivery of the manual topic "
            f"'{topic.get('label', '')}' for this session."
        ),
    )


def build_session_fidelity_adjudication_prompt(
    cycle_id: str,
    session_num: str,
    topics: list[dict[str, str]],
    manual_units: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> str:
    topic_labels = [str(topic.get("label", "")).strip() for topic in topics if str(topic.get("label", "")).strip()]
    scope_label = f"Session {session_num} manual content"
    if topic_labels:
        scope_label = f"{scope_label} ({'; '.join(topic_labels)})"
    return build_fidelity_adjudication_prompt(
        cycle_id,
        session_num,
        manual_units,
        evidence_rows,
        scope_label=scope_label,
        scope_description="Judge how well the retrieved evidence reflects the overall manual-session content for this session.",
        include_all_manual_units=True,
    )


def fidelity_generation_fields(payload: dict[str, Any], prompt_text: str) -> dict[str, str]:
    evidence_refs = ";".join(payload.get("evidence_refs", [])) if isinstance(payload.get("evidence_refs"), list) else ""
    manual_ids = ";".join(payload.get("manual_unit_ids", [])) if isinstance(payload.get("manual_unit_ids"), list) else ""
    return {
        "adjudication_prompt_text": prompt_text,
        "adjudication_summary": str(payload.get("adjudication_summary", "") or ""),
        "adjudication_label": str(payload.get("adherence_label", "") or ""),
        "adjudication_confidence": str(payload.get("confidence", "") or ""),
        "adjudication_evidence_refs": evidence_refs,
        "adjudication_manual_unit_ids": manual_ids,
        "adjudication_raw_response": json.dumps(payload, ensure_ascii=False) if payload else "",
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filter_cycle_rows(
    rows: list[dict[str, Any]],
    predicate,
) -> list[dict[str, Any]]:
    return [row for row in rows if not predicate(row)]


def filter_cycle_json_rows(
    rows: list[dict[str, Any]],
    predicate,
) -> list[dict[str, Any]]:
    return [row for row in rows if not predicate(row)]


def dedupe_rows_keep_last(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    """Dedupe rows by key_fields, keeping the last occurrence for each key.

    Returns a list of rows in the original order of last-seen keys.
    """
    seen = {}
    order: list[tuple] = []
    for row in rows:
        key = tuple(row.get(k, "") for k in key_fields)
        seen[key] = row
        if key not in order:
            order.append(key)
    # Preserve the order of last-seen keys
    return [seen[k] for k in order]


def has_target_filters(args: argparse.Namespace) -> bool:
    return bool(args.session_num or args.topic_id or args.question_id)


def should_process_session(session_num: str, args: argparse.Namespace) -> bool:
    return not args.session_num or session_num in args.session_num


def should_process_topic(topic_id: str, args: argparse.Namespace) -> bool:
    return not args.topic_id or topic_id in args.topic_id


def should_process_question(question_id: str, args: argparse.Namespace) -> bool:
    return not args.question_id or question_id in args.question_id


def resolve_fidelity_topk(
    manual_units: list[dict[str, Any]],
    explicit_topk: int,
    dynamic_topk: bool,
) -> int:
    if dynamic_topk and manual_units:
        return max(len(manual_units), 1)
    return explicit_topk


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
    parser.add_argument("--fidelity-ollama-model", default="", help="Optional Ollama model for fidelity adjudication summaries")
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
    parser.add_argument("--mode", choices=["all", "fidelity", "pi"], default="all", help="Run all outputs, only fidelity outputs, or only PI question outputs")
    parser.add_argument("--session-num", nargs="+", default=[], help="Optional manual session numbers to rerun")
    parser.add_argument("--topic-id", nargs="+", default=[], help="Optional topic ids to rerun")
    parser.add_argument("--question-id", nargs="+", default=[], help="Optional PI question ids to rerun")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cycle outputs instead of merging targeted reruns into existing files")
    parser.add_argument("--fixed-fidelity-topk", dest="dynamic_fidelity_topk", action="store_false", help="Use the fixed --fidelity-topk value instead of expected manual unit counts")
    parser.set_defaults(dynamic_fidelity_topk=True)
    parser.add_argument(
        "--enable-topic-fidelity",
        action="store_true",
        help="When set, run topic-level fidelity/adjudication in addition to session-level (default: session-level only)",
    )
    args = parser.parse_args()

    ensure_dir(CYCLE_ANALYSIS_DIR)
    meta_rows = get_rag_index_rows()
    path_lookup = build_doc_index_by_path(meta_rows)
    session_numbers = sorted({str(topic.get("session_num", "")) for topic in load_topic_entries() if str(topic.get("session_num", ""))})

    for cycle_num in args.cycles:
        cycle_id = f"PMHCycle{cycle_num}"
        cycle_dir = CYCLE_ANALYSIS_DIR / cycle_id
        ensure_dir(cycle_dir)
        # run-wide topk mode for this invocation: 'dynamic' (default) or 'fixed'
        run_topk_mode = "dynamic" if args.dynamic_fidelity_topk else "fixed"

        fidelity_rows: list[dict[str, Any]] = []
        session_fidelity_rows: list[dict[str, Any]] = []
        question_rows: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, Any]] = []
        session_fidelity_evidence_rows: list[dict[str, Any]] = []
        question_json_rows: list[dict[str, Any]] = []

        topic_count = 0
        for session_num in session_numbers:
            if not should_process_session(session_num, args):
                continue
            session_topics = get_topic_entries_for_session(session_num)
            selected_session_topics = [topic for topic in session_topics if should_process_topic(str(topic.get("id", "")), args)]
            if args.topic_id and not selected_session_topics:
                continue
            # Iterate topics for the session. Topic-level fidelity is optional and
            # only executed when the CLI flag is set; PI question generation runs
            # per-topic whenever mode includes 'pi' (or 'all').
            for topic in selected_session_topics:
                topic_count += 1
                if args.limit_topics and topic_count > args.limit_topics:
                    break
                topic_id = topic.get("id", "")
                topic_label = topic.get("label", "")
                session_summary = get_session_summary(session_num).get("session_summary", "")
                topic_definition = get_topic_definition(topic_id, topic_label, session_summary)
                session_manual_units = get_manual_units_for_session(session_num, topic_id=topic_id)

                # Topic-level fidelity/adjudication remains optional and only runs when the flag is set.
                if args.enable_topic_fidelity and args.mode in {"all", "fidelity"}:
                    fidelity_query = get_str("fidelity", "fidelity_query_template", "Session {session_num} {topic_label}").format(
                        session_num=session_num,
                        topic_label=topic_label,
                    ).strip()
                    resolved_topk = resolve_fidelity_topk(session_manual_units, args.fidelity_topk, args.dynamic_fidelity_topk)
                    fidelity_windows = build_transcript_windows(
                        fidelity_query,
                        topic_id=topic_id,
                        session_num=session_num,
                        cycle_id=cycle_id,
                        topk=resolved_topk,
                        weight_doc=args.fidelity_weight_doc,
                        weight_topic=args.fidelity_weight_topic,
                        meta_rows=meta_rows,
                        path_lookup=path_lookup,
                        window=args.context_window,
                    )
                    fidelity_row = summarize_fidelity(cycle_id, session_num, topic, fidelity_windows, session_manual_units)
                    # annotate which topk mode/value produced this row
                    fidelity_row["topk_mode"] = run_topk_mode
                    fidelity_row["topk_value"] = str(resolved_topk)
                    if args.fidelity_ollama_model:
                        prompt = build_topic_fidelity_adjudication_prompt(
                            cycle_id,
                            session_num,
                            topic,
                            session_manual_units,
                            fidelity_windows,
                        )
                        payload = parse_json_response(
                            call_ollama(
                                prompt,
                                args.fidelity_ollama_model,
                                ollama_ssh_host=args.ollama_ssh_host,
                                ollama_ssh_key=args.ollama_ssh_key,
                                ollama_remote_bin=args.ollama_remote_bin,
                            )
                        )
                        fidelity_row.update(fidelity_generation_fields(payload, prompt))

                    fidelity_rows.append(fidelity_row)
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
                                "topk_mode": run_topk_mode,
                                "topk_value": str(resolved_topk),
                            }
                        )

                # PI question generation runs per-topic when mode includes 'pi'
                if args.mode in {"all", "pi"}:
                    for question_spec in PI_QUESTION_SPECS:
                        if args.mode == "fidelity" or not should_process_question(question_spec["question_id"], args):
                            continue
                        query_text = build_pi_query_text(
                            session_num=session_num,
                            topic_label=topic_label,
                            topic_definition=topic_definition,
                            question_label=question_spec["label"],
                            query_template=question_spec["query_template"],
                            session_summary=session_summary,
                        )
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
                        confidence_explanation = answer_payload.get("confidence_explanation", "")

                        if not answer_summary and isinstance(answer_payload, dict) and answer_payload.get("fallback_answer"):
                            answer_summary = answer_payload.get("fallback_answer", "")
                            if not confidence_explanation:
                                confidence_explanation = "Parsed fallback answer from non-JSON model output; see raw_response for full text."

                        if not answer_summary and not question_windows:
                            answer_summary = "No retrieved evidence was found for this question."
                            confidence_explanation = "No evidence — model not invoked."
                        evidence_refs = ";".join(answer_payload.get("evidence_refs", [])) if isinstance(answer_payload.get("evidence_refs", list), list) else ""
                        manual_ids = ";".join(answer_payload.get("manual_unit_ids", [])) if isinstance(answer_payload.get("manual_unit_ids", list), list) else ""

                        if evidence_refs:
                            if evidence_refs not in (answer_summary or ""):
                                sep = " " if answer_summary and not answer_summary.endswith(".") else " "
                                answer_summary = (answer_summary or "") + f"{sep}Evidence refs: {evidence_refs}"
                        if not confidence_explanation:
                            if isinstance(answer_payload, dict) and answer_payload.get("raw_response"):
                                if not answer_payload.get("fallback_answer"):
                                    confidence_explanation = "Model returned a non-JSON response; see raw_response for details."
                            elif answer_payload:
                                confidence_explanation = "No confidence explanation provided."

                        if not answer_summary and question_windows:
                            if isinstance(answer_payload, dict) and answer_payload.get("raw_response"):
                                if not answer_payload.get("fallback_answer"):
                                    answer_summary = "Model returned a non-JSON response; see raw_response."
                            else:
                                answer_summary = "Model did not produce an answer."
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
                                "confidence_explanation": confidence_explanation,
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

            if args.mode == "pi":
                continue
            session_summary_row = get_session_summary(session_num)
            session_summary = session_summary_row.get("session_summary", "")
            session_topic_labels = [str(topic.get("label", "")).strip() for topic in selected_session_topics if str(topic.get("label", "")).strip()]
            session_fidelity_query = build_session_fidelity_query(
                session_num=session_num,
                session_summary=session_summary,
                topic_labels=session_topic_labels,
            )
            session_manual_units = get_manual_units_for_session(session_num)
            resolved_topk_session = resolve_fidelity_topk(session_manual_units, args.fidelity_topk, args.dynamic_fidelity_topk)
            session_fidelity_windows = build_session_fidelity_windows(
                cycle_id=cycle_id,
                session_num=session_num,
                session_summary=session_summary,
                session_topics=session_topics,
                topk=resolved_topk_session,
                weight_doc=args.fidelity_weight_doc,
                weight_topic=args.fidelity_weight_topic,
                meta_rows=meta_rows,
                path_lookup=path_lookup,
                window=args.context_window,
            )
            session_fidelity_row = summarize_session_fidelity(
                cycle_id,
                session_num,
                selected_session_topics,
                session_fidelity_windows,
                session_manual_units,
            )
            session_fidelity_row["topk_mode"] = run_topk_mode
            session_fidelity_row["topk_value"] = str(resolved_topk_session)
            if args.fidelity_ollama_model:
                prompt = build_session_fidelity_adjudication_prompt(
                    cycle_id,
                    session_num,
                    selected_session_topics,
                    session_manual_units,
                    session_fidelity_windows,
                )
                payload = parse_json_response(
                    call_ollama(
                        prompt,
                        args.fidelity_ollama_model,
                        ollama_ssh_host=args.ollama_ssh_host,
                        ollama_ssh_key=args.ollama_ssh_key,
                        ollama_remote_bin=args.ollama_remote_bin,
                    )
                )
                session_fidelity_row.update(fidelity_generation_fields(payload, prompt))
            session_fidelity_rows.append(session_fidelity_row)
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
                        "topk_mode": run_topk_mode,
                        "topk_value": str(resolved_topk_session),
                    }
                )

        targeted_run = has_target_filters(args)

        if args.mode in {"all", "fidelity"} and targeted_run and not args.overwrite:
            existing = read_csv_rows(cycle_dir / "fidelity_summary.csv")
            fidelity_rows = filter_cycle_rows(
                existing,
                lambda row: (
                    should_process_session(str(row.get("session_num", "")), args)
                    and should_process_topic(str(row.get("topic_id", "")), args)
                    and str(row.get("topk_mode", "dynamic")) == run_topk_mode
                    and str(row.get("topk_value", "")) == str(resolve_fidelity_topk(get_manual_units_for_session(str(row.get("session_num", "")), topic_id=row.get("topic_id", "")), args.fidelity_topk, args.dynamic_fidelity_topk))
                ),
            ) + fidelity_rows

            existing = read_csv_rows(cycle_dir / "session_fidelity_summary.csv")
            session_fidelity_rows = filter_cycle_rows(
                existing,
                lambda row: (
                    should_process_session(str(row.get("manual_session_num", "")), args)
                    and str(row.get("topk_mode", "dynamic")) == run_topk_mode
                    and str(row.get("topk_value", "")) == str(resolve_fidelity_topk(get_manual_units_for_session(str(row.get("manual_session_num", ""))), args.fidelity_topk, args.dynamic_fidelity_topk))
                ),
            ) + session_fidelity_rows

            existing = read_csv_rows(cycle_dir / "session_fidelity_evidence.csv")
            session_fidelity_evidence_rows = filter_cycle_rows(
                existing,
                lambda row: (
                    should_process_session(str(row.get("manual_session_num", "")), args)
                    and str(row.get("topk_mode", "dynamic")) == run_topk_mode
                    and str(row.get("topk_value", "")) == str(resolve_fidelity_topk(get_manual_units_for_session(str(row.get("manual_session_num", ""))), args.fidelity_topk, args.dynamic_fidelity_topk))
                ),
            ) + session_fidelity_evidence_rows

        if args.mode in {"all", "pi"} and targeted_run and not args.overwrite:
            existing = read_csv_rows(cycle_dir / "pi_question_answers.csv")
            question_rows = filter_cycle_rows(
                existing,
                lambda row: should_process_session(str(row.get("session_num", "")), args)
                and should_process_topic(str(row.get("topic_id", "")), args)
                and should_process_question(str(row.get("question_id", "")), args),
            ) + question_rows

            json_path = cycle_dir / "pi_question_answers.json"
            existing_json = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
            question_json_rows = filter_cycle_json_rows(
                existing_json,
                lambda row: should_process_session(str(row.get("session_num", "")), args)
                and should_process_topic(str(row.get("topic_id", "")), args)
                and should_process_question(str(row.get("question_id", "")), args),
            ) + question_json_rows

        if targeted_run and not args.overwrite:
            existing = read_csv_rows(cycle_dir / "topic_evidence.csv")
            evidence_rows = filter_cycle_rows(
                existing,
                lambda row: (
                    (
                        str(row.get("analysis_mode", "")) == "fidelity"
                        and args.mode in {"all", "fidelity"}
                        and should_process_session(str(row.get("session_num", "")), args)
                        and should_process_topic(str(row.get("topic_id", "")), args)
                        and str(row.get("topk_mode", "dynamic")) == run_topk_mode
                        and str(row.get("topk_value", "")) == str(resolve_fidelity_topk(get_manual_units_for_session(str(row.get("session_num", "")), topic_id=row.get("topic_id", "")), args.fidelity_topk, args.dynamic_fidelity_topk))
                    )
                    or (
                        str(row.get("analysis_mode", "")) == "pi_question"
                        and args.mode in {"all", "pi"}
                        and should_process_session(str(row.get("session_num", "")), args)
                        and should_process_topic(str(row.get("topic_id", "")), args)
                        and should_process_question(str(row.get("question_id", "")), args)
                    )
                ),
            ) + evidence_rows

        if args.mode in {"all", "fidelity"}:
            # Dedupe fidelity rows by (cycle_id, session_num, topic_id) keeping last-seen
            fidelity_rows = dedupe_rows_keep_last(fidelity_rows, ["cycle_id", "session_num", "topic_id"])
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
                    "topk_mode",
                    "topk_value",
                    "adjudication_prompt_text",
                    "adjudication_summary",
                    "adjudication_label",
                    "adjudication_confidence",
                    "adjudication_evidence_refs",
                    "adjudication_manual_unit_ids",
                    "adjudication_raw_response",
                ],
                fidelity_rows,
            )
            # Dedupe session fidelity rows by (cycle_id, manual_session_num)
            session_fidelity_rows = dedupe_rows_keep_last(session_fidelity_rows, ["cycle_id", "manual_session_num"])
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
                    "topk_mode",
                    "topk_value",
                    "adjudication_prompt_text",
                    "adjudication_summary",
                    "adjudication_label",
                    "adjudication_confidence",
                    "adjudication_evidence_refs",
                    "adjudication_manual_unit_ids",
                    "adjudication_raw_response",
                ],
                session_fidelity_rows,
            )
            # Dedupe session fidelity evidence by (cycle_id, manual_session_num, retrieval_rank)
            session_fidelity_evidence_rows = dedupe_rows_keep_last(session_fidelity_evidence_rows, ["cycle_id", "manual_session_num", "retrieval_rank"])
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
        if args.mode in {"all", "pi"}:
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
                    "confidence_explanation",
                    "evidence_refs",
                    "manual_unit_ids",
                    "raw_response",
                ],
                question_rows,
            )
            (cycle_dir / "pi_question_answers.json").write_text(
                json.dumps(question_json_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if args.mode in {"all", "fidelity", "pi"}:
            # Dedupe topic evidence by (cycle_id, session_num, topic_id, question_id, retrieval_rank)
            evidence_rows = dedupe_rows_keep_last(evidence_rows, ["cycle_id", "session_num", "topic_id", "question_id", "retrieval_rank"])
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
                    "topk_mode",
                    "topk_value",
                ],
                evidence_rows,
            )
        print(f"Wrote cycle analysis outputs to {cycle_dir}")


if __name__ == "__main__":
    main()
