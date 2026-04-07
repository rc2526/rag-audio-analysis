## Data aggregation, provenance, and statistics — Quick reference

This document explains where fidelity and PI data live, how they are produced (which scripts and transformations), and how the summary statistics are calculated.

## High-level layout

- Per-cycle (detailed) outputs:
  - `data/derived/cycle_analysis/PMHCycle{N}/` — one folder per cycle
    - `fidelity_summary.csv` — topic-level fidelity adjudication rows
    - `session_fidelity_summary.csv` — session-level fidelity rows
    - `session_fidelity_evidence.csv` — transcript evidence windows used for session fidelity
    - `pi_question_answers.csv` — PI (participant impact) question rows (CSV)
    - `pi_question_answers.json` — PI question full payloads (JSON with prompt, answer, evidence)
    - `topic_evidence.csv` — all evidence windows (both fidelity + PI) used in this cycle

- Aggregated / reporting tables:
  - `data/derived/cycle_analysis/summary/` — aggregator outputs
    - `table_cycle_manual_session_fidelity.csv` — flattened session-topic fidelity rows (used by analysis scripts)
    - `table_pi_question_answers.csv` — flattened PI answers across cycles
    - `summary_fidelity_by_cycle.csv` — per-cycle summary metrics (means, pct high/moderate/low)
    - `summary_session_fidelity_by_cycle.csv` / `summary_session_fidelity_by_manual_session.csv` — other aggregated views
    - `table_topic_evidence.csv`, `table_cycle_manual_session_evidence.csv`, etc.

- Logs
  - `run_cycle_pi.log`, `run_cycle_pi_overwrite.log`, `run_cycle_fidelity.log` — run logs created when you run `scripts/run_cycle_analysis.py` with `tee`.
  - `app/streamlit.log` — Streamlit app logs.

## How the per-cycle data is derived

Primary orchestrator: `scripts/run_cycle_analysis.py`.
Steps per cycle/topic/session:
1. Load topic entries, RAG index metadata, and manual-unit definitions.
2. Build a concise retrieval `query_text` (via `build_pi_query_text` or `build_session_fidelity_query`). This is the string used to query the vector store.
3. Retrieve evidence windows using `query_evidence(...)` inside `build_transcript_windows(...)`.
   - Parameters affecting retrieval: `--question-topk`, `--fidelity-topk`, `--fidelity-weight-doc`, `--fidelity-weight-topic`, `--question-weight-doc`, `--question-weight-topic`, and `--context-window`.
   - Fidelity top-k uses `resolve_fidelity_topk(...)`: by default `dynamic_fidelity_topk=True` so top-k is `max(len(manual_units),1)` unless `--fixed-fidelity-topk` is supplied.
4. Construct summaries:
   - Topic-level: `summarize_fidelity(...)` computes coverage and an `adherence_score` (see formulas below).
   - Session-level: `summarize_session_fidelity(...)` aggregates across topics for a session.
5. Optional generation (automated adjudication / PI answers):
   - PI: `build_question_prompt(...)` builds the full model prompt which contains metadata, manual-unit excerpts, and the retrieved evidence labeled E1..EN. The model is called via `call_ollama(...)`. The returned JSON is parsed with `parse_json_response(...)` and fields (answer_summary, evidence_refs, confidence, confidence_explanation, manual_unit_ids, raw_response) are saved.
   - Fidelity: `build_topic_fidelity_adjudication_prompt(...)` / `build_session_fidelity_adjudication_prompt(...)` are used similarly for generating adjudication summaries; outputs map into `adjudication_*` CSV columns.
6. Write per-cycle CSV/JSON files (overwritten or merged depending on `--overwrite` and targeted filters).

Merge vs overwrite behavior
- Default behaviour when rerunning with filters (e.g., `--session-num`) is to merge: the script reads existing per-cycle CSVs and filters out only the targeted rows, then appends newly generated rows. Use `--overwrite` to replace files completely.

## Aggregation (how the summary tables are produced)

- Script: `scripts/aggregate_cycle_outputs.py` reads all per-cycle CSVs under `data/derived/cycle_analysis/PMHCycle*/` and writes the `summary/` tables.
- Aggregations include:
  - counts (number of session-topic rows)
  - means and medians of numeric columns (e.g., `adherence_score`, `manual_unit_coverage`, `subsection_coverage`, `retrieved_evidence_count`)
  - percentages of adherence labels: `pct_high_adherence`, `pct_moderate_adherence`, `pct_low_adherence` computed as (count with label)/(total count) * 100
  - grouped aggregations by cycle, topic, question, or manual-session as appropriate.

## Exact formulas and key fields

- manual_unit_coverage = matched_manual_unit_count / expected_manual_unit_count
  - matched_manual_unit_count is computed by checking which retrieved windows map to expected manual unit IDs.

- subsection_coverage = matched_subsection_count / expected_subsection_count

- evidence_density = retrieved_evidence_count / expected_manual_unit_count

- adherence_score = manual_weight * manual_cov + subsection_weight * subsection_cov
  - `manual_weight` and `subsection_weight` come from settings (keys under the `fidelity` section).
  - Cutoffs for labeling adherence:
    - `adherence_high_cutoff` (default from settings) -> label `high` if adherence_score >= high_cutoff
    - `adherence_moderate_cutoff` -> label `moderate` if adherence_score >= moderate_cutoff
    - otherwise label `low`

- Percentile/summary stats in `summary_fidelity_by_cycle.csv` are computed from the set of topic-level rows for each cycle (mean, median, etc.).

## PI-specific fields

- `query_text` — stored in CSV; used only for retrieval.
- `prompt_text` — the full prompt passed to the model; saved in `pi_question_answers.csv` and JSON.
- `answer_summary`, `confidence`, `confidence_explanation`, `evidence_refs`, `manual_unit_ids`, `raw_response` — parsed from model output and saved for auditing.

## Where to run / reproduce

- Run per-cycle (merge-mode, PI only, cycles 2–5 example):
  ```bash
  export PYTHONPATH="$PWD"
  .venv/bin/python scripts/run_cycle_analysis.py --cycles 2 3 4 5 --mode pi --ollama-model gpt-oss:120b 2>&1 | tee -a run_cycle_pi.log
  ```
- Recompute aggregations:
  ```bash
  export PYTHONPATH="$PWD"
  .venv/bin/python scripts/aggregate_cycle_outputs.py
  ```
- Run the analysis script I added:
  ```bash
  export PYTHONPATH="$PWD"
  .venv/bin/python scripts/fidelity_stats.py
  # writes: data/derived/cycle_analysis/summary/fidelity_analysis.txt
  ```

## Notes & caveats

- Provenance: per-cycle CSVs and `pi_question_answers.json` together provide full provenance: retrieval query, evidence windows used (E1..EN), the exact prompt, and raw model output.
- Settings: many thresholds and weights are configurable in `settings.ini` and accessed via `rag_audio_analysis.settings.get_float/get_int/get_str`.
- Retrieval is stochastic only insofar as the embedding index or ranking implementation is non-deterministic; prompts and model outputs are non-deterministic.
- If you need per-session live progress markers or more granular stats (e.g., per-topic histograms), I can add an exporter or a small notebook.



