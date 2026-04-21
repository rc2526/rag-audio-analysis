import json
import os
import shlex
import subprocess
from typing import Any, Optional

from rag_audio_analysis.settings import get_str
from rag_audio_analysis.source_bridge import infer_manual_unit_for_text, query_evidence


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


def format_evidence_excerpt(text: str, limit: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def build_chat_prompt(
    question: str,
    evidence_rows: list[dict[str, Any]],
    variant: str = "default",
    cycle_id: str = "",
    session_num: str = "",
    topic: dict[str, Any] | None = None,
    manual_units: list[dict[str, Any]] | None = None,
    question_spec: dict[str, Any] | None = None,
) -> str:
    """
    Build a model prompt for RAG chat. Supports variants:
      - "default": concise QA prompt (legacy behavior)
      - "pi_question": replicate the PI-question prompt style used by the cycle analysis
      - "fidelity": replicate the fidelity adjudication prompt style used by the cycle analysis

    The function accepts optional metadata so callers (UI or CLI) can supply
    cycle/session/topic/manual_units when available. When metadata is missing,
    the prompt falls back to sensible placeholders.
    """
    # Helper to render evidence list lines
    def _evidence_lines(rows: list[dict[str, Any]], excerpt_limit: int) -> list[str]:
        lines: list[str] = []
        for idx, row in enumerate(rows, start=1):
            # Include manual_session label and manual match score to aid judgment
            lines.append(
                f"- E{idx} | session_id={row.get('session_id','')} | manual_session={row.get('manual_session','')} | "
                f"manual_unit={row.get('manual_unit_id_best_match','')} | score={row.get('manual_unit_match_score','')} | "
                f"{format_evidence_excerpt(row.get('text', ''), excerpt_limit)}"
            )
        return lines

    if variant == "pi_question":
        manual_unit_limit = 6
        manual_excerpt_chars = 220
        evidence_excerpt_chars = 400

        parts = [
            "You are analyzing intervention transcripts with retrieved evidence only.",
            "Return valid JSON only.",
            "",
            f"Cycle: {cycle_id}",
            f"Session: Session {session_num}",
            f"Topic: {topic.get('label','') if topic else ''}",
            f"Question: {question_spec['label'] if question_spec and 'label' in question_spec else question}",
            "",
            "Relevant manual units for this session/topic:",
        ]
        if manual_units:
            for unit in (manual_units[:manual_unit_limit] if isinstance(manual_units, list) else []):
                parts.append(
                    f"- {unit.get('manual_unit_id','')} | {unit.get('manual_subsection','')} | "
                    f"{format_evidence_excerpt(unit.get('text',''), manual_excerpt_chars)}"
                )
        else:
            parts.append("- No matching manual units were available.")
        parts.append("")
        parts.append("Retrieved transcript evidence:")
        parts.extend(_evidence_lines(evidence_rows, evidence_excerpt_chars))
        parts.extend(
            [
                "",
                "Respond with JSON with exactly these keys (include `session_number` and `session_explanation`):",
                '{',
                '  "answer_summary": "short paragraph",',
                '  "session_number": 0,',
                '  "session_explanation": "short 1-3 sentence justification that cites evidence refs (for example: \"E2, E5\")",',
                '  "evidence_count": 0,',
                '  "evidence_refs": ["E1","E2"],',
                '  "manual_unit_ids": ["MAN_0001","MAN_0002"],',
                '  "confidence": "low|medium|high",',
                '  "confidence_explanation": "short sentence explaining why that confidence was chosen; cite evidence refs if relevant",',
                '}',
                "Base the answer only on the retrieved evidence and manual units shown above. You MUST set `session_number` to an integer 1..12 representing the best guess of the manual session (or -1/\"unknown\" if you cannot decide).",
                "In `session_explanation`, briefly explain (1–3 sentences) how you arrived at `session_number` and cite the specific evidence lines by their IDs (for example: 'E2, E5'). Use manual_session labels and match scores where relevant. If evidence is weak or contradictory, say so and include the refs that illustrate the contradiction.",
                "Include inline evidence refs in the `answer_summary` next to the claims they support (for example: 'The parent described practicing mindful breathing with their child (E1, E4)').",
                "For the confidence field include an explanatory sentence in `confidence_explanation` that references evidence refs (for example: 'medium — E2 and E4 show clear facilitator modeling; E5 is noisy').",
            ]
        )
        return "\n".join(parts)

    if variant == "fidelity":
        manual_unit_limit = len(manual_units) if manual_units else 6
        manual_excerpt_chars = 220
        evidence_excerpt_chars = 400

        scope_label = question if question else ""
        scope_description = "Judge how well the retrieved evidence reflects delivery of the manual content for this session."

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
            for unit in (manual_units[:manual_unit_limit] if isinstance(manual_units, list) else []):
                parts.append(
                    f"- {unit.get('manual_unit_id','')} | {unit.get('manual_subsection','')} | "
                    f"{format_evidence_excerpt(unit.get('text',''), manual_excerpt_chars)}"
                )
        else:
            parts.append("- No matching manual units were available.")
        parts.append("")
        parts.append("Retrieved transcript evidence:")
        if evidence_rows:
            parts.extend(_evidence_lines(evidence_rows, evidence_excerpt_chars))
        else:
            parts.append("- No retrieved evidence windows were available.")
        parts.extend(
            [
                "",
                "Respond with JSON with exactly these keys (include `session_number` and `session_explanation`):",
                "{",
                '  "adjudication_summary": "short paragraph",',
                '  "session_number": 0,',
                '  "session_explanation": "short 1-3 sentence justification that cites evidence refs (for example: \"E2, E5\")",',
                '  "adherence_label": "high|moderate|low",',
                '  "evidence_refs": ["E1","E3"],',
                '  "manual_unit_ids": ["MAN_0001","MAN_0003"],',
                '  "confidence": "low|medium|high"',
                "}",
                "Base the judgment only on the retrieved evidence and expected manual units shown above. You MUST set `session_number` to an integer 1..12 representing the best guess of the manual session (or -1/\"unknown\" if you cannot decide).",
                "In `session_explanation`, briefly explain (1–3 sentences) how you arrived at `session_number` and cite the specific evidence lines by their IDs (for example: 'E2, E5').",
                "Use 'low' adherence when evidence is weak or clearly incomplete.",
            ]
        )
        return "\n".join(parts)

    # default (legacy) prompt
    parts = [
        "You are answering a research user's question using only the retrieved evidence below.",
        "Return valid JSON only.",
        "",
        f"Question: {question}",
        "",
        "Retrieved evidence:",
    ]
    if evidence_rows:
        # Use the same evidence-line renderer (includes manual_session and match score)
        parts.extend(_evidence_lines(evidence_rows, 500))
    else:
        parts.append("- No retrieved evidence windows were available.")
    parts.extend(
        [
            "",
            "Respond with JSON with exactly these keys (include `session_number` and `session_explanation`):",
            "{",
            '  "answer_summary": "short paragraph",',
            '  "session_number": 0,',
            '  "session_explanation": "short 1-3 sentence justification that cites evidence refs (for example: \"E2, E5\")",',
            '  "evidence_refs": ["E1","E2"],',
            '  "manual_unit_ids": ["MAN_0001","MAN_0002"],',
            '  "confidence": "low|medium|high"',
            "}",
            "Base the answer only on the retrieved evidence. You MUST set `session_number` to an integer 1..12 representing the best guess of the manual session (or -1/\"unknown\" if you cannot decide).",
            "In `session_explanation`, briefly explain (1–3 sentences) how you arrived at `session_number` and cite the specific evidence lines by their IDs (for example: 'E2, E5'). If evidence is weak or contradictory, say so and include the refs that show the contradiction.",
        ]
    )
    return "\n".join(parts)


def run_chat_query(
    question: str,
    cycle_id: str = "",
    topk: int = 8,
    weight_doc: float = 1.0,
    weight_topic: float = 0.0,
    include_manual: bool = False,
    manual_only: bool = False,
    answer_with_model: bool = True,
    ollama_model: Optional[str] = None,
    ollama_ssh_host: Optional[str] = None,
    ollama_ssh_key: Optional[str] = None,
    ollama_remote_bin: Optional[str] = None,
    *,
    prompt_variant: str = "default",
    session_num: str = "",
    topic: dict[str, Any] | None = None,
    manual_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_name = ollama_model or get_str("ollama", "default_model", "gpt-oss:120b")
    ssh_host = ollama_ssh_host if ollama_ssh_host is not None else get_str("ollama", "ssh_host", "rc2526@10.168.224.148")
    ssh_key = ollama_ssh_key if ollama_ssh_key is not None else get_str("ollama", "ssh_key", "~/.ssh/ollama_remote")
    remote_bin = ollama_remote_bin if ollama_remote_bin is not None else get_str("ollama", "remote_bin", "/usr/local/bin/ollama")

    rows = query_evidence(
        question,
        topk=topk,
        weight_doc=weight_doc,
        weight_topic=weight_topic,
        cycle_id=cycle_id,
    transcript_only=not include_manual,
    manual_only=manual_only,
    )

    enriched_rows = []
    for row in rows:
        enriched = dict(row)
        # If the row already contains a canonical manual match (for example
        # returned by the `manual_only` canonical branch), skip expensive
        # re-matching and trust the provided fields.
        if not enriched.get("manual_unit_id_best_match"):
            manual_match = infer_manual_unit_for_text(str(row.get("text", "") or ""))
            # preserve best-match id and score
            enriched["manual_unit_id_best_match"] = manual_match.get("manual_unit_id", "")
            enriched["manual_unit_match_score"] = manual_match.get("score", "")
            # attach manual_week (numeric session) and a human-friendly manual_session label
            enriched["manual_week"] = manual_match.get("manual_week", "")
        else:
            # ensure manual_week is present when canonical fields exist
            enriched.setdefault("manual_week", enriched.get("manual_week", ""))
            enriched.setdefault("manual_unit_match_score", enriched.get("manual_unit_match_score", ""))
        try:
            if enriched.get("manual_week"):
                enriched["manual_session"] = f"Session {str(enriched.get('manual_week'))}"
            else:
                # fallback to any session-like id present on the row
                enriched["manual_session"] = str(row.get("session_id", "") or "")
        except Exception:
            enriched["manual_session"] = str(row.get("session_id", "") or "")
        enriched_rows.append(enriched)

    answer_payload = {}
    prompt_text = ""
    if answer_with_model and enriched_rows:
        prompt_text = build_chat_prompt(
            question,
            enriched_rows,
            variant=prompt_variant,
            cycle_id=cycle_id,
            session_num=session_num,
            topic=topic,
            manual_units=manual_units,
        )
        answer_payload = parse_json_response(
            call_ollama(
                prompt_text,
                model_name,
                ollama_ssh_host=ssh_host,
                ollama_ssh_key=ssh_key,
                ollama_remote_bin=remote_bin,
            )
        )
        # If the model didn't provide a session_number, attempt a deterministic fallback
        # by taking the manual_week that appears most frequently among the top evidence rows
        # (only if evidence rows contain a manual_week). This gives a reproducible heuristic
        # when the model omits the explicit field.
        try:
            if isinstance(answer_payload, dict) and "session_number" not in answer_payload:
                # collect manual_week occurrences
                counts: dict[int, int] = {}
                for r in enriched_rows:
                    try:
                        mw = r.get("manual_week")
                        if mw is None or mw == "":
                            continue
                        # manual_week may be numeric string
                        mw_int = int(mw)
                        counts[mw_int] = counts.get(mw_int, 0) + 1
                    except Exception:
                        continue
                if counts:
                    # pick the manual_week with highest count; tie-breaker: highest summed match score
                    top_week = max(counts.items(), key=lambda kv: kv[1])[0]
                    answer_payload["session_number"] = int(top_week)
                    # annotate confidence_explanation if present or create one
                    conf = answer_payload.get("confidence_explanation", "")
                    note = f"inferred session_number={top_week} from majority of retrieved evidence."
                    answer_payload["confidence_explanation"] = (conf + " " + note).strip()
                    # provide a short machine-generated session_explanation and mark as inferred
                    if "session_explanation" not in answer_payload or not answer_payload.get("session_explanation"):
                        # collect supporting evidence ids
                        supporting = []
                        for idx, r in enumerate(enriched_rows, start=1):
                            try:
                                if int(r.get("manual_week", "")) == int(top_week):
                                    supporting.append(f"E{idx}")
                            except Exception:
                                continue
                        sup_text = ", ".join(supporting) if supporting else ""
                        gen_note = f"Inferred from retrieved evidence {sup_text} which most frequently referenced manual session {top_week}."
                        answer_payload["session_explanation"] = gen_note
                    answer_payload["session_inferred"] = True
                else:
                    # no manual_week info available; set unknown sentinel
                    answer_payload.setdefault("session_number", -1)
                    answer_payload.setdefault("session_explanation", "No manual session information found in retrieved evidence.")
                    answer_payload["session_inferred"] = True
        except Exception:
            # be conservative: ensure session_number exists
            if isinstance(answer_payload, dict):
                answer_payload.setdefault("session_number", -1)

    return {
        "question": question,
        "cycle_id": cycle_id,
        "topk": topk,
        "weight_doc": weight_doc,
        "weight_topic": weight_topic,
    "include_manual": include_manual,
    "manual_only": manual_only,
        "answer_with_model": answer_with_model,
    "prompt_variant": prompt_variant,
        "prompt_text": prompt_text,
        "answer": answer_payload,
        "evidence": enriched_rows,
    }
