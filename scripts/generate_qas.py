"""Generate grounded QA pairs from manual units.

Usage:
  python3 scripts/generate_qas.py --seeds MAN_0011,MAN_0022 --output tests/rag_eval_10.json
  python3 scripts/generate_qas.py --count 10 --output tests/rag_eval_10.json

Behavior:
- If --seeds is provided (comma-separated manual ids), use those.
- Otherwise pick the first `--count` manual units from the built manual index.
- For each manual id, load `matching_text` and prompt the model to produce JSON:
    {"question": "...", "gold_answer": "...", "gold_evidence": ["MAN_0011"]}
  The model is instructed to use ONLY the provided evidence.
- Validate JSON and a simple grounded token overlap (>= 0.3 recommended).
- Write resulting list into the output JSON.

This script uses the local `call_ollama` interface (same as the app). It does not auto-publish results.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Optional

from rag_audio_analysis.source_bridge import build_manual_unit_index
from rag_audio_analysis.chat_runner import call_ollama
from rag_audio_analysis.settings import get_str

# Simple tokenizer for groundedness checks
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
STOPWORDS = set(["the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are"])


def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s) if t and t.lower() not in STOPWORDS]


def extract_json_from_text(s: str) -> Optional[Dict]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(s[first : last + 1])
            except Exception:
                return None
    return None


PROMPT_TEMPLATE = (
    "You are a careful annotator.\n"
    "I will provide a single piece of evidence labeled with an id.\n"
    "Using ONLY the evidence text provided, produce ONE user-facing question and a concise gold answer (1-2 sentences) that is strictly supported by the evidence.\n"
    "Output a single valid JSON object matching exactly this schema (no extra keys):\n"
    "{\\n  \"question\": string,\\n  \"gold_answer\": string,\\n  \"gold_evidence\": [string]\\n}\n"
    "The value for \"gold_evidence\" MUST be a list containing the provided id (example: [\"MAN_0011\"]).\n"
    "Do NOT invent facts or add content not in the evidence. If the evidence does not support a factual answer, write a question that asks for the content directly present in the passage.\n\n"
    "Evidence (id={id}):\n" "{evidence}\n\n" "Return only the JSON object."
)


def build_prompt(manual_id: str, evidence: str) -> str:
    # PROMPT_TEMPLATE contains literal JSON braces which conflict with
    # str.format() placeholders. Use simple replace to fill {id} and
    # {evidence} safely.
    return PROMPT_TEMPLATE.replace("{id}", str(manual_id)).replace("{evidence}", str(evidence))


def validate_grounding(answer: str, evidence: str, threshold: float = 0.3) -> float:
    a_tokens = tokenize(answer)
    if not a_tokens:
        return 0.0
    evidence_lower = evidence.lower()
    covered = sum(1 for t in a_tokens if t in evidence_lower)
    return covered / len(a_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated manual ids to use as seeds")
    p.add_argument("--count", type=int, default=10, help="If --seeds omitted, pick this many manual units")
    p.add_argument("--output", type=str, default="tests/rag_eval_10.json")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--min_grounded", type=float, default=0.3)
    args = p.parse_args()

    # load canonical manual units
    manual_units = build_manual_unit_index() or []
    manu_map = {str(u.get("manual_unit_id")): u for u in manual_units if u.get("manual_unit_id")}

    seeds: List[str] = []
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    else:
        # pick first N manual ids sorted by id
        all_ids = sorted([str(u.get("manual_unit_id")) for u in manual_units if u.get("manual_unit_id")])
        seeds = all_ids[: args.count]

    if not seeds:
        print("No seed manual ids found. Aborting.")
        return

    model = args.model or get_str("ollama", "default_model", "gpt-oss:120b")

    outputs = []
    for i, sid in enumerate(seeds, start=1):
        mu = manu_map.get(str(sid))
        if not mu:
            print(f"Seed {sid} not found in manual index, skipping.")
            continue
        evidence = str(mu.get("matching_text") or mu.get("text") or "").strip()
        if not evidence:
            print(f"Manual {sid} has no matching_text/text, skipping.")
            continue
        prompt = build_prompt(sid, evidence)
        print(f"Calling model for seed {sid} ({i}/{len(seeds)}) using model {model}...")
        try:
            resp = call_ollama(prompt, model)
        except Exception as exc:
            print(f"Model call failed for {sid}: {exc}")
            continue
        parsed = extract_json_from_text(resp)
        if not parsed:
            print(f"Failed to parse JSON from model for {sid}. Saving raw response and skipping.")
            print("RAW:", resp)
            continue
        # basic sanity checks
        q = str(parsed.get("question", "")).strip()
        a = str(parsed.get("gold_answer", "")).strip()
        ge = parsed.get("gold_evidence") or []
        if not q or not a or not isinstance(ge, list):
            print(f"Invalid JSON structure for {sid}, missing required keys. Raw: {parsed}")
            continue
        # ensure the provided seed id is included in gold_evidence
        if str(sid) not in [str(x) for x in ge]:
            print(f"Warning: model did not include seed id {sid} in gold_evidence; adding it.")
            ge = [str(sid)] + [str(x) for x in ge]
        grounded = validate_grounding(a, evidence)
        print(f"Grounded token coverage for {sid}: {grounded:.2f}")
        if grounded < args.min_grounded:
            print(f"Grounding below threshold ({args.min_grounded}); keeping pair but flagging warning.")
        outputs.append({"id": f"Q{i}", "question": q, "gold_answer": a, "gold_evidence": ge})
        if len(outputs) >= args.count:
            break

    # write output
    with open(args.output, "w") as fh:
        json.dump(outputs, fh, indent=2)
    print(f"Wrote {len(outputs)} QA pairs to {args.output}")


if __name__ == "__main__":
    main()
