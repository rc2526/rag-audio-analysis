#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "export_topic_catalog.py",
    ROOT / "scripts" / "export_manual_units.py",
    ROOT / "scripts" / "export_speaker_role_map.py",
    ROOT / "scripts" / "export_transcript_spans.py",
    ROOT / "scripts" / "generate_content_review_queue.py",
    ROOT / "scripts" / "generate_analysis_summaries.py",
    ROOT / "scripts" / "init_empty_tables.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"Running {script.name}...")
        subprocess.run([sys.executable, str(script)], check=True)
    print("Bootstrap complete.")


if __name__ == "__main__":
    main()
