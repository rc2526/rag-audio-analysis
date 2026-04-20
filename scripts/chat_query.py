#!/usr/bin/env python3
import argparse
import json

from rag_audio_analysis.chat_runner import run_chat_query
from rag_audio_analysis.settings import get_float, get_int, get_str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--cycle-id", default="")
    parser.add_argument("--topk", type=int, default=get_int("chat", "topk", 8))
    parser.add_argument("--weight-doc", type=float, default=get_float("chat", "weight_doc", 1.0))
    parser.add_argument("--weight-topic", type=float, default=get_float("chat", "weight_topic", 0.0))
    parser.add_argument("--include-manual", action="store_true")
    parser.add_argument("--manual-only", action="store_true", help="Restrict index-level retrieval to manual passages only")
    parser.add_argument("--answer-with-model", action="store_true")
    parser.add_argument("--ollama-model", default=get_str("ollama", "default_model", "gpt-oss:120b"))
    parser.add_argument("--ollama-ssh-host", default=get_str("ollama", "ssh_host", "rc2526@10.168.224.148"))
    parser.add_argument("--ollama-ssh-key", default=get_str("ollama", "ssh_key", "~/.ssh/ollama_remote"))
    parser.add_argument("--ollama-remote-bin", default=get_str("ollama", "remote_bin", "/usr/local/bin/ollama"))
    args = parser.parse_args()

    output = run_chat_query(
        question=args.question,
        cycle_id=args.cycle_id,
        topk=args.topk,
        weight_doc=args.weight_doc,
        weight_topic=args.weight_topic,
        include_manual=args.include_manual,
    manual_only=args.manual_only,
        answer_with_model=args.answer_with_model,
        ollama_model=args.ollama_model,
        ollama_ssh_host=args.ollama_ssh_host,
        ollama_ssh_key=args.ollama_ssh_key,
        ollama_remote_bin=args.ollama_remote_bin,
    )
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
