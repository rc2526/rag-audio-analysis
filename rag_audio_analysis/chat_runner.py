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


def build_chat_prompt(question: str, evidence_rows: list[dict[str, Any]]) -> str:
    parts = [
        "You are answering a research user's question using only the retrieved evidence below.",
        "Return valid JSON only.",
        "",
        f"Question: {question}",
        "",
        "Retrieved evidence:",
    ]
    for idx, row in enumerate(evidence_rows, start=1):
        parts.append(
            f"- E{idx} | source_type={row.get('source_type','')} | file={row.get('file','')} | "
            f"manual_unit={row.get('manual_unit_id_best_match','')} | "
            f"{format_evidence_excerpt(row.get('text', ''), 500)}"
        )
    parts.extend(
        [
            "",
            "Respond with JSON with exactly these keys:",
            "{",
            '  "answer_summary": "short paragraph",',
            '  "evidence_refs": ["E1"],',
            '  "manual_unit_ids": ["MAN_0001"],',
            '  "confidence": "low|medium|high"',
            "}",
            "Base the answer only on the retrieved evidence. If evidence is weak or contradictory, say so.",
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
    answer_with_model: bool = True,
    ollama_model: Optional[str] = None,
    ollama_ssh_host: Optional[str] = None,
    ollama_ssh_key: Optional[str] = None,
    ollama_remote_bin: Optional[str] = None,
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
    )

    enriched_rows = []
    for row in rows:
        manual_match = infer_manual_unit_for_text(str(row.get("text", "") or ""))
        enriched = dict(row)
        enriched["manual_unit_id_best_match"] = manual_match.get("manual_unit_id", "")
        enriched["manual_unit_match_score"] = manual_match.get("score", "")
        enriched_rows.append(enriched)

    answer_payload = {}
    prompt_text = ""
    if answer_with_model and enriched_rows:
        prompt_text = build_chat_prompt(question, enriched_rows)
        answer_payload = parse_json_response(
            call_ollama(
                prompt_text,
                model_name,
                ollama_ssh_host=ssh_host,
                ollama_ssh_key=ssh_key,
                ollama_remote_bin=remote_bin,
            )
        )

    return {
        "question": question,
        "cycle_id": cycle_id,
        "topk": topk,
        "weight_doc": weight_doc,
        "weight_topic": weight_topic,
        "include_manual": include_manual,
        "answer_with_model": answer_with_model,
        "prompt_text": prompt_text,
        "answer": answer_payload,
        "evidence": enriched_rows,
    }
