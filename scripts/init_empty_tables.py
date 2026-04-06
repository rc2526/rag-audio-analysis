#!/usr/bin/env python3
import shutil

from rag_audio_analysis.config import CODED_EVIDENCE_CSV, QUOTE_BANK_CSV, TEMPLATES_DIR


def copy_if_missing(template_name: str, destination) -> None:
    template_path = TEMPLATES_DIR / template_name
    if not destination.exists():
        shutil.copyfile(template_path, destination)
        print(f"Initialized {destination}")
    else:
        print(f"Keeping existing {destination}")


def main() -> None:
    copy_if_missing("quote_bank_template.csv", QUOTE_BANK_CSV)


if __name__ == "__main__":
    main()
