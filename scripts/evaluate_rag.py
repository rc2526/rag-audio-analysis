"""Simple RAG evaluation script

Usage:
    python3 scripts/evaluate_rag.py --input tests/rag_eval_10.json --top_k 8 --min_sim 0.28

Input JSON format: list of examples, each:
{
  "id": "Q1",
  "question": "What session discusses cravings?",
  "gold_answer": "Session 3 addresses cravings...",
  "gold_evidence": ["MAN_0011", "path/to/transcript:123"]
}

Metrics computed per example:
- retrieval_precision_at_k (precision)
- retrieval_recall_at_k (recall vs gold_evidence)
- grounded_token_coverage (percent of answer content tokens present in retrieved texts)
- answer_relevance (cosine sim between answer and question embeddings)
- answer_correctness_sim (cosine sim between answer and gold answer embeddings)
- answer_token_f1 (token-overlap F1 vs gold answer)

Outputs a per-example row and aggregate means.

This script uses the project's `retrieve_for_question` and `answer_rag` and `source_bridge.encode_texts`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from typing import List, Dict

import numpy as np

from rag_audio_analysis.rag_service import retrieve_for_question, answer_rag
from rag_audio_analysis.source_bridge import encode_texts

# tiny tokenizer
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
STOPWORDS = set([
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was",
    "were", "it", "that", "this", "as", "by", "be", "have", "has", "had",
])


def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s) if t and t.lower() not in STOPWORDS]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def extract_human_summary(answer_raw: str) -> str:
    # try to parse a leading JSON object with 'answer_summary'
    s = (answer_raw or "").strip()
    if not s:
        return ""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return str(obj.get("answer_summary") or obj.get("summary") or s)
    except Exception:
        # attempt to find first {...} block
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                obj = json.loads(s[first : last + 1])
                if isinstance(obj, dict):
                    return str(obj.get("answer_summary") or obj.get("summary") or s)
            except Exception:
                pass
    # fallback: return whole text
    return s


def token_f1(pred: str, gold: str) -> float:
    p_tokens = tokenize(pred)
    g_tokens = tokenize(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    p_counts = {}
    for t in p_tokens:
        p_counts[t] = p_counts.get(t, 0) + 1
    g_counts = {}
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1
    common = 0
    for t, c in p_counts.items():
        common += min(c, g_counts.get(t, 0))
    if common == 0:
        return 0.0
    prec = common / len(p_tokens)
    rec = common / len(g_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def evaluate_example(example: Dict, top_k: int, min_sim: float, prompt_variant: str) -> Dict:
    q = example.get("question", "")
    gold_answer = example.get("gold_answer", "")
    gold_evidence = set(example.get("gold_evidence", []))
    # retrieval
    res = retrieve_for_question(q, "", top_k=top_k, min_similarity=min_sim)
    retrieved = res.get("manuals", []) + res.get("windows", [])
    # build retrieved ids set to compare against gold (we accept manual_unit_id or path matches)
    retrieved_ids = []
    retrieved_texts = []
    for r in retrieved:
        # prefer manual_unit_id, else path, else doc_index
        rid = r.get("manual_unit_id") or r.get("path") or str(r.get("doc_index"))
        if rid:
            retrieved_ids.append(str(rid))
        retrieved_texts.append(str(r.get("text", "")))
    k = max(1, min(len(retrieved_ids), top_k))
    num_rel = sum(1 for rid in retrieved_ids[:top_k] if rid in gold_evidence)
    precision_at_k = num_rel / float(top_k) if top_k else 0.0
    recall_at_k = num_rel / float(len(gold_evidence)) if gold_evidence else (1.0 if num_rel == 0 else 0.0)

    # generation
    rag_out = answer_rag(q, "", top_k=top_k, min_similarity=min_sim, prompt_variant=prompt_variant)
    answer_raw = rag_out.get("answer_raw", "")
    answer_text = extract_human_summary(answer_raw)

    # groundedness: percent of answer tokens present in retrieved_texts
    retrieved_join = "\n".join(retrieved_texts).lower()
    a_tokens = [t for t in tokenize(answer_text)]
    if not a_tokens:
        grounded_coverage = 0.0
    else:
        covered = sum(1 for t in a_tokens if t in retrieved_join)
        grounded_coverage = covered / len(a_tokens)

    # embeddings for relevance/correctness
    try:
        emb_q = encode_texts([q])[0]
        emb_a = encode_texts([answer_text])[0]
        emb_gold = encode_texts([gold_answer])[0] if gold_answer else None
    except Exception:
        emb_q = emb_a = emb_gold = None

    answer_relevance = cosine_sim(emb_a, emb_q) if emb_a is not None and emb_q is not None else 0.0
    answer_correctness_sim = cosine_sim(emb_a, emb_gold) if emb_a is not None and emb_gold is not None else 0.0
    answer_token_f1 = token_f1(answer_text, gold_answer)

    return {
        "id": example.get("id"),
        "question": q,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "grounded_coverage": grounded_coverage,
        "answer_relevance": answer_relevance,
        "answer_correctness_sim": answer_correctness_sim,
        "answer_token_f1": answer_token_f1,
        "retrieved_ids": retrieved_ids[:top_k],
        "answer_text": answer_text,
    }


def summarize(results: List[Dict]) -> Dict:
    keys = [
        "precision_at_k",
        "recall_at_k",
        "grounded_coverage",
        "answer_relevance",
        "answer_correctness_sim",
        "answer_token_f1",
    ]
    agg = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in results]
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        agg[f"std_{k}"] = float(np.std(vals)) if vals else 0.0
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="tests/rag_eval_10.json")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--min_sim", type=float, default=0.28)
    p.add_argument("--prompt_variant", type=str, default="default")
    args = p.parse_args()

    with open(args.input, "r") as fh:
        examples = json.load(fh)

    results = []
    for ex in examples:
        print("Evaluating", ex.get("id"))
        r = evaluate_example(ex, top_k=args.top_k, min_sim=args.min_sim, prompt_variant=args.prompt_variant)
        results.append(r)
        print(json.dumps(r, indent=2))

    agg = summarize(results)
    print("\nSUMMARY:")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
