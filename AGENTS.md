# AGENTS.md

## Project State

This repository is `rag-audio-analysis`.

The current working direction is:

- keep session-level manual-session fidelity
- retire topic-level fidelity as the intended future design
- keep PI questions topic-based
- enrich PI queries with topic glosses derived from session summaries

# AGENTS.md — Agent handoff

This file documents the project's current design, runbook, troubleshooting, and a compact handoff checklist so another engineer or automated agent can continue work reliably.

Keep this file short and actionable. For longer background see `INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md` and `SESSION_LEVEL_FIDELITY_REDESIGN.md`.

## Quick summary

- Canonical fidelity mode: session-level fidelity (one row per `cycle_id + manual_session_num`).
- PI questions (topic-anchored) are separate and saved to `pi_question_answers.csv` and `pi_question_answers.json` per cycle.
- Topic-level fidelity is deprecated for long-term architecture; topic-level PI remains topic-anchored.

## Recent agent actions

- Patched `scripts/rebuild_topic_evidence_from_pi_json.py` to include `question_id` in the dedupe key so PI windows from different questions that share a retrieval_rank are not collapsed.
- Rebuilt `topic_evidence.csv` for `PMHCycle1` from `pi_question_answers.json` (created `topic_evidence.csv.bak`); result: PMHCycle1 now contains all PI windows (1104) and had no topic-level `analysis_mode=fidelity` rows to preserve.
- Ran the same rebuild script with `--apply` for PMHCycle2..PMHCycle5; backups were created for each cycle's `topic_evidence.csv`. Verify counts with the grep/wc checks in the runbook.
 - Updated `scripts/run_cycle_analysis.py` to unguard PI question generation so PI runs per-topic whenever `--mode` includes `pi` (no need to set `--enable-topic-fidelity` for PI). Topic-level fidelity remains gated by `--enable-topic-fidelity`.
 - Updated topic evidence dedupe in `scripts/run_cycle_analysis.py` to include `question_id` so evidence windows from different PI questions are not collapsed.

### Recent tooling and UI updates

- Added adjudication-level aggregates to `scripts/aggregate_cycle_outputs.py` (two new summary tables for generation grade and generation confidence by cycle) and also added session-level adjudication rollups. These are written under `data/derived/cycle_analysis/summary/` as `summary_adjudication_*` CSVs.
- Added a dedicated "Summaries" tab to the Streamlit UI (`app/streamlit_app.py`) that lists only the CSVs produced by the aggregator, previews them with human-friendly column labels, offers a simple numeric plot, and provides a download button.
- Added a starter Jupyter notebook at `notebooks/summary_visualizations.ipynb` to make quick visual exploration reproducible (loads aggregator CSVs and renders a few default charts).
- Cleaned the `data/derived/cycle_analysis/summary/` folder to contain only aggregator-generated outputs and moved older auxiliary files (images, ad-hoc CSVs, JSON summaries, and a small report) into a timestamped archive folder `data/derived/cycle_analysis/summary/archived_20260413_200700/`. This archive is reversible — move files back if needed.
- Updated `README.md` with documentation for the new summary files and the Summaries tab so the README remains the canonical pipeline source-of-truth.

## What this agent is responsible for (contract)

Inputs
- Raw transcripts and index metadata (local rag index rows).
- `data/inputs/session_summaries.csv`, `data/derived/topic_catalog.csv`.

Outputs (per-cycle folder: `data/derived/cycle_analysis/PMHCycleX`)
- `fidelity_summary.csv` (topic-level fidelity rows, only written when topic-level fidelity is enabled)
- `session_fidelity_summary.csv` (session-level fidelity summaries)
- `session_fidelity_evidence.csv` (retrieved windows for sessions)
- `pi_question_answers.csv` and `pi_question_answers.json` (PI answers + evidence)
- `topic_evidence.csv` (merged topic evidence for both fidelity and PI windows)

Success criteria
- CSVs are readable by the Streamlit app without KeyError (canonical columns present: `session_num` or `manual_session_num`, `question_id` when PI evidence exists).
- `pi_question_answers.json` contains an `evidence` list of window dicts if PI retrieval succeeded.

Error modes
- Missing `pi_question_answers.json` or missing `evidence` arrays prevents full topic evidence reconstruction without re-running retrievals.
- Mismatched column names across cycles (schema drift) can break the app; use `rag_audio_analysis/source_bridge.normalize_cycle_frame()` to harmonize.

## Where to edit (short map)

- Retrieval, matching: `rag_audio_analysis/source_bridge.py`
- High-level orchestration and CSV schema: `scripts/run_cycle_analysis.py`
- Aggregation: `scripts/aggregate_cycle_outputs.py`
- Streamlit app: `app/streamlit_app.py`
- Settings and tunables: `settings.ini`

## Quick runbook (common tasks)

Run a full cycle build (slow):

```zsh
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 2 3 4 5 --ollama-model gpt-oss:120b
```

Run a narrow targeted rerun (recommended for iteration):

```zsh
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 --mode fidelity --session-num 1 --overwrite
.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 --mode pi --session-num 1 --ollama-model gpt-oss:120b
.venv/bin/python scripts/aggregate_cycle_outputs.py
```

Notes
- Default flow is merge-safe (no `--overwrite`) for partial reruns.
- Use `--overwrite` only when you intend to replace the per-cycle files.

## Rebuilding `topic_evidence.csv` from PI outputs (without full re-run)

When PI runs completed and `pi_question_answers.json` exists, you can reconstruct `topic_evidence.csv` for that cycle from the JSON without re-running retrievals or model calls. High-level approach:

- Read `pi_question_answers.json` (it's a list of entries). Each entry contains fields like `cycle_id`, `session_num`, `topic_id`, `question_id`, and an `evidence` array.
- For each evidence window in `evidence`, expand into one CSV row with `analysis_mode="pi_question"`, `question_id` set, and map window fields to the `topic_evidence.csv` columns (coalesce `rank`/`retrieval_rank`, `score`/`score_combined`, `manual_unit_id`/`manual_unit_id_best_match`).
- If `pi_question_answers.json` is missing or `evidence` arrays are empty, full reconstruction is not possible — you can only create placeholder rows.

Practical tips
- Back up the existing `topic_evidence.csv` before overwriting (`.bak`).
- Use retrieval window enumeration as a fallback `retrieval_rank` if one is not present.
- After rebuilding per-cycle `topic_evidence.csv`, run `scripts/aggregate_cycle_outputs.py` and restart Streamlit if you want the UI to reflect changes.

Reference script
- A minimal, safe, stdlib-only script is provided in the developer notes (not checked into the repo by default); it reads `pi_question_answers.json`, expands `evidence` windows, writes `topic_evidence.csv`, and creates a `.bak` of the previous file.

## Common flags and their meaning

- `--mode` ∈ {`all`, `fidelity`, `pi`} — controls which outputs are produced.
- `--enable-topic-fidelity` — runs topic-level fidelity/adjudication and writes topic-level outputs; deprecated long-term but still available.
- `--fidelity-topk` / `--question-topk` / `--fixed-fidelity-topk` — control top-k behavior.
- `--session-num`, `--topic-id`, `--question-id` — targeted rerun filters.
- `--overwrite` — replace per-cycle outputs rather than merge.

## Troubleshooting quicklist

- App KeyError on `question_id` or `session_num`: verify per-cycle CSVs contain canonical columns; aggregator or UI will tolerate `manual_session_num` vs `session_num` if normalized by `source_bridge.normalize_cycle_frame()`.
- Empty PI evidence in UI but PI CSVs present: check `pi_question_answers.json` `evidence` arrays — if they are empty the retrieval returned no windows.
- Model parsing failures (non-JSON): inspect `pi_question_answers.csv` `raw_response` column and use the rerun helper (review JSON -> rerun with `--dry-run`) to re-invoke the model with the stored prompt.
- Slow runs: use targeted reruns with `--session-num` / `--topic-id` and prefer merge-mode to avoid rewrites of unrelated rows.

## Verification checklist (post-change)

Run these after any rerun or after major edits:

1. Confirm per-cycle outputs exist:
  - `ls data/derived/cycle_analysis/PMHCycle*/pi_question_answers.json`
2. If you rebuilt `topic_evidence.csv`, confirm non-empty `question_id` values:
  - `csvcut -c question_id data/derived/cycle_analysis/PMHCycle5/topic_evidence.csv | tail -n +2 | grep -v '^$' | wc -l`
3. Aggregate summaries refreshed:
  - `python scripts/aggregate_cycle_outputs.py`
4. Restart Streamlit and spot-check the Evidence Browser filters.

## Handoff checklist (what I expect the next agent/engineer to do)

1. Open a Yale-attached VS Code session (recommended host: `rhea.chatterjeeyale.edu@MEDLQ9WNJ5CXJ.med.yale.internal`).
2. Pull latest changes and confirm git status in that environment.
3. If you need updated PI-backed topic evidence and `pi_question_answers.json` exists, run the rebuild-from-json script (backup first).
4. Run `scripts/aggregate_cycle_outputs.py` and restart Streamlit to validate the UI.
5. If further fixes are needed, change the small set of files listed above and use targeted reruns.

## Contacts & ownership

- Primary owner (runtime & infra): rhea.chatterjeeyale.edu
- Code owner (analysis logic): look at `git blame` on `scripts/run_cycle_analysis.py` and `rag_audio_analysis/source_bridge.py` for recent authorship.

## Next suggested engineering work

- Add an automated `rebuild_topic_evidence_from_pi_json.py` script to `scripts/` that also supports safe merge into existing `topic_evidence.csv` and preserves `topk_mode`/`topk_value` when present.
- Add a small CI check that validates per-cycle CSV headers to catch schema drift early.

---

If you want, I can add the rebuild script into `scripts/` and run a dry-run locally to show the delta rows that would be written.

  - `--question-id`
- after any accepted cycle/session rerun, rerun `scripts/aggregate_cycle_outputs.py` before evaluating the app

## Narrow Inspection Workflow

When a quick inspectable refresh is needed, the most practical workflow is:

1. run a narrow fidelity pass first
2. run the matching narrow PI pass second
3. rerun aggregation
4. restart Streamlit if needed

Validated working example on Yale for `PMHCycle1`, session 1:

```bash
cd /Users/rhea.chatterjeeyale.edu/rag-audio-analysis
env PYTHONPATH=. /Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 --mode fidelity --session-num 1 --overwrite
env PYTHONPATH=. /Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 --mode pi --session-num 1 --ollama-model gpt-oss:120b
env PYTHONPATH=. /Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/python scripts/aggregate_cycle_outputs.py
```

Observed output shape for that narrow session-1 rebuild:

- `fidelity_summary.csv`: 2 rows
- `session_fidelity_summary.csv`: 1 row
- `pi_question_answers.csv`: 6 rows
- `pi_question_answers.json`: 6 items
- `topic_evidence.csv`: 50 rows

Interpretation:

- session 1 currently has 2 topics
- PI now uses 3 questions per topic
- so 6 PI answer rows is the expected shape for a successful session-1-only rebuild

Important caveat:

- after a narrow rerun, the cycle folder may intentionally represent only a partial slice of the cycle
- in the app, use:
  - `Cycle = PMHCycle1`
  - `Session number = 1`
- do not interpret `All sessions` for that cycle as complete until a full cycle rerun has been done

## Aggregation Behavior

`scripts/aggregate_cycle_outputs.py` does the following:

- scans every `PMHCycle*` directory under `data/derived/cycle_analysis`
- loads the per-cycle CSV outputs if present
- concatenates them into summary-level “table” CSVs under:
  - `data/derived/cycle_analysis/summary/`
- computes grouped summary CSVs for:
  - fidelity by cycle
  - fidelity by topic
  - session-fidelity by cycle
  - session-fidelity by manual session
  - PI by cycle
  - PI by question type
  - PI by cycle and question type
  - evidence by cycle

Important UI note:

- the Streamlit app primarily reads the per-cycle files directly from each `PMHCycleX` folder for interactive filtering
- the aggregate script is still useful for refreshing exported summary tables, but it does not replace the need for valid per-cycle files

## Streamlit Restart Note

After narrow reruns and aggregation, restarting Streamlit can help ensure the app is reading the refreshed files.

Validated Yale command:

```bash
cd /Users/rhea.chatterjeeyale.edu/rag-audio-analysis
nohup /Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/rag_audio_streamlit.log 2>&1 &
```

Observed runtime details on April 6, 2026:

- Streamlit was confirmed listening on port `8501`
- app log path:
  - `/tmp/rag_audio_streamlit.log`
- after the session-1 narrow rebuild, the most useful inspection setting was:
  - `Cycle = PMHCycle1`
  - `Session number = 1`

## Known Methodological Note

The PI question:

- `How often do facilitators refer to this topic?`

is not a true frequency question under the current top-`k` retrieval design.

## Session-fidelity generation ("Generation Grade") — quick runbook

Use this section as the canonical quick-run instructions for the Generation Grade flow (fidelity retrieval + optional generation adjudication). These snippets assume you are in the repository root and using the repo venv (`.venv`).

- Run fidelity only for a set of cycles (merge-mode, preserves unrelated rows) and log to `run_cycle_fidelity.log`:

```zsh
export PYTHONPATH="$PWD"
set -o pipefail
.venv/bin/python scripts/run_cycle_analysis.py \
  --cycles 2 3 4 5 \
  --mode fidelity \
  --fidelity-ollama-model gpt-oss:120b \
  2>&1 | tee -a run_cycle_fidelity.log
```

- Run targeted PI question(s) (merge-mode) for sessions 1 & 2 and append logs to `run_cycle_pi.log`:

```zsh
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_cycle_analysis.py \
  --cycles 1 --session-num 1 2 --mode pi \
  --question-id facilitator_delivery participant_practice \
  --ollama-model gpt-oss:120b |& tee -a run_cycle_pi.log
```

- Force-rewrite a cycle's PI CSV (use only when you intend to replace the file):

```zsh
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_cycle_analysis.py \
  --cycles 1 --session-num 1 2 --mode pi \
  --question-id participant_child_home \
  --ollama-model gpt-oss:120b --overwrite |& tee -a run_cycle_pi_overwrite.log
```

- After any run, rebuild aggregated summary tables (safe, quick):

```zsh
export PYTHONPATH="$PWD"
.venv/bin/python scripts/aggregate_cycle_outputs.py
```

- Restart Streamlit (repo venv) so the app reads refreshed summary files:

```zsh
export PYTHONPATH="$PWD"
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Notes
- Merge-mode (no `--overwrite`) preserves unrelated rows: the script removes rows matching the targeted filters and appends new rows, then writes the file with the existing rows intact.
- `--overwrite` writes only the rows produced by the current run and will drop previously saved rows that you did not regenerate.
- Use the log files (`run_cycle_fidelity.log`, `run_cycle_pi.log`, `run_cycle_pi_overwrite.log`) and `tail -f` to monitor progress in real time.
- Per-session loops give clearer per-session start/end markers but reload embeddings/models per invocation and are much slower than single-shot runs that process many sessions/cycles at once.

Quick verification
- Check the PI CSV header contains the new `confidence_explanation` column:

```zsh
head -n1 data/derived/cycle_analysis/PMHCycle1/pi_question_answers.csv
```

- Inspect any PI rows for a given session:

```zsh
grep -E 'participant_child_home|facilitator_delivery|participant_practice' data/derived/cycle_analysis/PMHCycle1/pi_question_answers.csv | grep ',1,' | sed -n '1,20p'
```

If anything here looks stale or you want the runbook exported elsewhere, say where and I'll add it.

It currently behaves more like a qualitative evidence question than a count/rate estimate.

If revisited later, either:

- reword it to match retrieval-based evidence
- or build a separate frequency-style metric outside the current PI QA setup

## Session-fidelity generation ("Generation Grade") — quick runbook

When you want the generation-backed session-level adjudication (Generation Grade) alongside the existing retrieval metrics, follow this small runbook from the project root.

Steps

- Run a narrow session-level fidelity pass (adds retrieval-based session metrics).
- Optionally run PI passes if you want updated PI outputs.
- Run the aggregator to refresh summary tables.
- Restart or refresh the Streamlit app to see the Generation Grade fields in the UI.

Commands (recommended — use the repo root and the project's `.venv`):

```zsh
# 1) session-level fidelity with generation enabled for session 1
PYTHONPATH="$PWD" .venv/bin/python scripts/run_cycle_analysis.py \
  --cycles 1 --session-num 1 --mode fidelity --overwrite \
  --fidelity-ollama-model gpt-oss:120b

# 2) (optional) run PI passes for the same session
PYTHONPATH="$PWD" .venv/bin/python scripts/run_cycle_analysis.py \
  --cycles 1 --session-num 1 --mode pi --overwrite --ollama-model gpt-oss:120b

# 3) aggregate per-cycle outputs into summary tables
PYTHONPATH="$PWD" .venv/bin/python scripts/aggregate_cycle_outputs.py

# 4) restart Streamlit (foreground)
PYTHONPATH="$PWD" .venv/bin/python -m streamlit run app/streamlit_app.py

# 4b) or run headless on port 8501
PYTHONPATH="$PWD" nohup .venv/bin/python -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &> /tmp/rag_audio_streamlit.log &
```

Quick verification

- Confirm per-cycle outputs were written:

```zsh
ls -la data/derived/cycle_analysis/PMHCycle1
head -n 6 data/derived/cycle_analysis/PMHCycle1/session_fidelity_summary.csv
head -n 6 data/derived/cycle_analysis/PMHCycle1/session_fidelity_evidence.csv
```

- Confirm aggregated summary tables were written:

```zsh
ls -la data/derived/cycle_analysis/summary
head -n 6 data/derived/cycle_analysis/summary/table_cycle_manual_session_fidelity.csv
```

Notes

- The generation step is enabled by passing `--fidelity-ollama-model` to `run_cycle_analysis.py`. The generated fields are additive: the original retrieval-based scores remain and the new generation fields are saved in the same session-level CSVs under columns prefixed with `adjudication_` (now labeled in the UI as "Generation Grade").
- For session-level prompts we include the full manual-unit set for the session (no small-cap truncation).
- If imports or dependencies fail, run `.venv/bin/pip install -r requirements.txt` inside the project venv and retry.

If you want, I can extract this section into a separate `README_RUNBOOK.md` instead of editing `AGENTS.md`.

## If Work Resumes Later

If a future session needs to pick up where this one left off, the safest next steps are:

1. treat session-fidelity as the canonical fidelity mode
2. assume the rebuilt session-fidelity CSVs are current unless intentionally rerun later
3. use `INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md` as the basis for partial-rerun refactoring
4. keep PI queries topic-based, but preserve the new topic-definition enrichment
5. avoid reintroducing topic-helper semantics into session-fidelity evidence

## Correct Working Environment

Important:

- future execution-heavy work should be done from the Yale-attached VS Code environment, not from a separate local shell checkout
- the intended machine is:
  - `rhea.chatterjeeyale.edu@MEDLQ9WNJ5CXJ.med.yale.internal`
- intended repo path:
  - `/Users/rhea.chatterjeeyale.edu/rag-audio-analysis`

Why this matters:

- this session was able to edit the local checkout, but SSH auth from the tool shell to the Yale machine was not fully available
- that means local code changes can drift from the actual Yale runtime environment
- for reliable validation, Streamlit, embeddings, and long pipeline runs should be run from the Yale-hosted workspace

Practical handoff:

1. open the Yale-attached VS Code window or remote shell for:
   - `rhea.chatterjeeyale.edu@MEDLQ9WNJ5CXJ.med.yale.internal`
2. open the repo at:
   - `/Users/rhea.chatterjeeyale.edu/rag-audio-analysis`
3. confirm git status there before making new changes
4. continue from the current design state recorded in this file and in:
   - `INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md`
   - `SESSION_LEVEL_FIDELITY_REDESIGN.md`

If a new Codex session is started, it should preferably be launched from that Yale-attached VS Code context so the tool environment matches the real runtime environment.

## Suggested Next Work

Likely next useful engineering tasks:

- confirm completion of the current full Ollama-backed build on Yale
- rerun `scripts/aggregate_cycle_outputs.py` after that build finishes
- validate how the updated topic definitions and enriched PI queries feel in the live app
- optionally run a smaller dedicated PI-weighting evaluation later if topic-weighting is revisited
