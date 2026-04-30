#!/usr/bin/env python3
from pathlib import Path
import json
from rag_audio_analysis.config import SOURCE_META
from rag_audio_analysis.source_bridge import is_manual_row


def main():
    meta_path = Path(SOURCE_META)
    print("SOURCE_META:", meta_path)
    if not meta_path.exists():
        print("Meta file not found at expected path.")
        return
    try:
        raw = json.loads(meta_path.read_text(encoding='utf-8'))
    except Exception as e:
        print("Failed to read/parse meta.json:", e)
        return
    print("Total index entries:", len(raw))
    manual = [r for r in raw if is_manual_row(r)]
    print("Manual entries detected:", len(manual))
    if manual:
        print("\nSample manual entries (first 10):")
        for i, r in enumerate(manual[:10], start=1):
            print(f"{i}. source={r.get('source')}, path={r.get('path')}, snippet={str(r.get('text',''))[:120]!r}")
    else:
        print("\nNo manual entries found; sample non-manual rows (first 10):")
        for i, r in enumerate(raw[:10], start=1):
            print(f"{i}. source={r.get('source')}, path={r.get('path')}, snippet={str(r.get('text',''))[:120]!r}")


if __name__ == '__main__':
    main()
