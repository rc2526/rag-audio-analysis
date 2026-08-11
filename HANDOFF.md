# Research Handoff

This repo is the analysis layer on top of the external `rag-audio` project.
It is meant to be shared with another researcher or engineer who needs to continue
the session-analysis work without reverse-engineering the current state.

## Current Working Direction

- Session-level fidelity is the canonical fidelity output.
- PI questions remain topic-based.
- Topic-level fidelity is still available, but only when `--enable-topic-fidelity` is set.
- Session summaries should be read as the primary fidelity artifact; topic-level rows are optional.

## What Lives Where

- `scripts/run_cycle_analysis.py` orchestrates cycle runs.
- `scripts/aggregate_cycle_outputs.py` builds summary tables under `data/derived/cycle_analysis/summary/`.
- `scripts/rebuild_topic_evidence_from_pi_json.py` rebuilds `topic_evidence.csv` from existing PI JSON.
- `app/streamlit_app.py` is the UI.
- `rag_audio_analysis/` contains shared helpers, config, and retrieval logic.

## Data And Outputs

- `data/` is working data and is intentionally ignored by git.
- The repo expects the external `rag-audio` source tree configured in `settings.ini` / `rag_audio_analysis/config.py`.
- Cycle outputs are generated under `data/derived/cycle_analysis/PMHCycleX/`.
- Treat logs, caches, backups, and derived CSVs as rebuildable artifacts, not hand-edited source.

## Setup

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

If the derived tables are missing, bootstrap them from the repo scripts:

```zsh
python scripts/bootstrap_analysis_data.py
python scripts/aggregate_cycle_outputs.py
```

## Common Runs

Session-level fidelity only:

```zsh
python scripts/run_cycle_analysis.py --cycles 1 --mode fidelity
```

PI only:

```zsh
python scripts/run_cycle_analysis.py --cycles 1 --mode pi --ollama-model gpt-oss:120b
```

Session-level fidelity plus PI:

```zsh
python scripts/run_cycle_analysis.py --cycles 1 --mode all --ollama-model gpt-oss:120b
```

Topic-level fidelity, only when explicitly needed:

```zsh
python scripts/run_cycle_analysis.py --cycles 1 --mode all --enable-topic-fidelity
```

## Verification

After a rerun or cleanup pass:

```zsh
git status --short
python scripts/aggregate_cycle_outputs.py
```

Then spot-check the generated cycle CSVs for the relevant cycle and session.

## Handoff Notes

- Do not treat the current tree as pristine; there are active code changes in the main repo.
- Keep `README.md` and `AGENTS.md` aligned with the current run behavior.
- Prefer small reruns over full cycle rebuilds when iterating on retrieval or prompts.

