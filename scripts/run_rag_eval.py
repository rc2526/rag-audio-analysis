"""Run RAG and non-RAG systems on an eval CSV and output per-query JSONs.

CSV expected columns: question_id, question, cycle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from rag_audio_analysis.rag_service import answer_rag, answer_non_rag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-file", required=True)
    p.add_argument("--out-dir", default="data/derived/rag_eval")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.eval_file)
    rows = []
    for _, r in df.iterrows():
        qid = r.get("question_id") or r.get("id") or "q"
        question = r.get("question")
        cycle = r.get("cycle")
        rag = answer_rag(question, cycle)
        non = answer_non_rag(question)
        out = {"id": qid, "question": question, "cycle": cycle, "rag": rag, "nonrag": non}
        Path(out_dir / f"{qid}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        rows.append({"id": qid, "cycle": cycle, "rag_file": str(out_dir / f"{qid}.json")})

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
