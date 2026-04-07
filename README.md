# rag-audio-analysis

Research-oriented analysis layer built on top of the existing `rag-audio` project.

This repository treats the original `rag-audio` repo as the source of truth for:
- de-identified transcripts
- `manual.txt`
- `manual_topics.csv`
- `rag_index/` retrieval artifacts
- the sentence-transformer / RAG runtime used by the original pipeline

The current version of this project is centered on an automated session-topic evidence pipeline for:
- transcript-content analysis
- manual alignment / fidelity analysis
- PI-question summaries
- retrieval-backed browsing and chat in Streamlit

## What This Project Does

At a high level, this repo:

1. Reuses the existing `rag-audio` index and source data
2. Builds structured manual units from `manual.txt`
3. Runs session-topic retrieval against the indexed transcripts
4. Matches retrieved transcript evidence to manual units using embedding similarity
5. Generates cycle-level fidelity summaries and PI-question outputs
6. Aggregates those outputs into publication-friendly summary tables
7. Exposes the results in a Streamlit UI, including a small RAG-enabled chat tab

This repo does **not** replace `rag-audio`. It depends on it.

## Repository Structure

- `rag_audio_analysis/`
  - shared config, settings loader, source bridge, and chat runner
- `scripts/`
  - export, bootstrap, cycle-analysis, aggregation, and utility scripts
- `data/derived/`
  - generated CSV outputs and cycle-analysis results
- `app/`
  - Streamlit UI
- `settings.ini`
  - central settings file for active pipeline behavior
- `CONFIGURABLE_VALUES.md`
  - inventory of configurable values and remaining hardcoded logic

## Key Files

- `rag_audio_analysis/config.py`
  - project paths and source-repo path resolution
- `rag_audio_analysis/settings.py`
  - reads `settings.ini`
- `rag_audio_analysis/source_bridge.py`
  - bridge to the original `rag-audio` retrieval/index code
  - builds structured manual units
  - embedding-based topic matching
  - embedding-based manual-unit matching
- `rag_audio_analysis/chat_runner.py`
  - shared backend for the Streamlit RAG chat and CLI chat worker
- `scripts/bootstrap_analysis_data.py`
  - builds the basic derived CSVs
- `scripts/run_cycle_analysis.py`
  - runs the automated session-topic evidence pipeline across one or more cycles
- `scripts/aggregate_cycle_outputs.py`
  - combines per-cycle outputs into summary tables
- `scripts/chat_query.py`
  - CLI entry point for retrieval-backed chat
- `app/streamlit_app.py`
  - Streamlit interface for browsing outputs and running chat

## Derived Outputs

The bootstrap and analysis scripts write to `data/derived/`.

Core derived files include:
- `data/derived/topic_catalog.csv`
- `data/derived/manual_units.csv`
- `data/derived/transcript_spans.csv`
- `data/derived/coded_evidence.csv`
- `data/derived/quote_bank.csv`
- `data/derived/content_review_queue.csv`
- `data/derived/topic_content_summary.csv`
- `data/derived/topic_session_summary.csv`
- `data/derived/manual_fidelity_summary.csv`

Cycle-analysis outputs are written under:
- `data/derived/cycle_analysis/<cycle_id>/`

Those per-cycle folders contain files such as:
- `fidelity_summary.csv`
- `pi_question_answers.csv`
- `pi_question_answers.json`
- `topic_evidence.csv`

Aggregated publication-oriented summary tables are written under:
- `data/derived/cycle_analysis/summary/`

## Manual Units

The manual is currently parsed into structured units using this hierarchy:

1. `Session X`
2. session subsection
   - `Homework`
   - `Handouts`
   - `Discussion`
   - `Activity`
   - `Breathing Exercise`
   - `Physical Activity and Nutrition`
   - `Other Instructions`
3. fallback size cap within each subsection

Manual units are then assigned:
- `topic_id`
- `topic_match_score`
- session and subsection metadata

`manual_function` and `manual_skill_type` were intentionally removed from the current schema because they were heuristic labels and not part of the embedding-based matching workflow.

## Matching and Retrieval

### Topic assignment

Topic assignment now uses embedding similarity rather than the older token-scoring heuristic.

### Manual-unit assignment

Best matching manual units are now assigned using embedding similarity rather than raw token overlap.

### Retrieval modes

The cycle-analysis pipeline uses two retrieval modes:

- **Fidelity mode**
  - session-aware query:
    - `Session {n} {topic_label}`
  - intended to estimate manual alignment for a session-topic pair

- **PI-question mode**
  - question-specific retrieval prompts for:
    - facilitator reference
    - facilitator demonstration
    - participant practice
    - participant-child home practice

### Weighted retrieval defaults

Current active defaults (driven by `settings.ini` under `[cycle_analysis]` and `[chat]`):

- Fidelity retrieval (per-cycle/session-topic runs)
  - `fidelity_weight_doc = 1.0`
  - `fidelity_weight_topic = 0.0`
  - `fidelity_topk = 12` (how many retrieved windows to consider)

- PI-question / exploratory retrieval (question-driven runs)
  - `question_weight_doc = 1.0`
  - `question_weight_topic = 0.0`
  - `question_topk = 8`

These live in `settings.ini` and can be tuned per-run or per-environment.

## Settings

The active pipeline reads from `settings.ini`. Most runtime defaults for retrieval, adjudication, and UI behavior are configured there. Below are the most commonly tuned keys (section/key and current default):

Configurable settings (examples)

- [cycle_analysis]
  - `fidelity_topk = 12` (int) — number of windows to retrieve for fidelity adjudication
  - `question_topk = 8` (int) — number of windows to retrieve for PI questions
  - `fidelity_weight_doc = 1.0` (float) — document weight when building retrieval scores for fidelity runs
  - `fidelity_weight_topic = 0.0` (float) — topic weight when building retrieval scores for fidelity runs
  - `question_weight_doc = 1.0` (float) — document weight for PI-question retrieval
  - `question_weight_topic = 0.0` (float)

- [fidelity]
  - `manual_coverage_weight = 0.6` (float) — weight given to matched manual units in `adherence_score`
  - `subsection_coverage_weight = 0.4` (float)
  - `adherence_high_cutoff = 0.66` (float)
  - `adherence_moderate_cutoff = 0.33` (float)

- [prompting]
  - `manual_units_in_prompt = 6` (int) — how many manual-unit excerpts to include in the model prompt
  - `manual_excerpt_chars`, `evidence_excerpt_chars`, `display_excerpt_chars`, `ui_excerpt_chars` — truncation lengths used when composing prompts and UI snippets

- [ollama]
  - `default_model = gpt-oss:120b` — model used for optional generation-backed adjudication
  - `ssh_host`, `ssh_key`, `remote_bin` — remote runtime hooks for Ollama when using a remote host

- [ui] and [chat]
  - `streamlit_port`, `app_title`, `topk`, `weight_doc`, `weight_topic`, `include_manual`, `answer_with_model` — UI / chat defaults used by `app/streamlit_app.py`

If you change these values, re-run the relevant scripts (or restart Streamlit for UI changes). See `CONFIGURABLE_VALUES.md` for a longer inventory of tunables.

## Typical Workflow

### 1. Make sure the source repo is ready

The source `rag-audio` repo should already contain:
- de-identified transcripts
- `manual.txt`
- `manual_topics.csv`
- `rag_index/`

### 2. Bootstrap the base analysis data

From this repo root:

```bash
PYTHONPATH=. python3 scripts/bootstrap_analysis_data.py
```

### 3. Run cycle analysis

Example:

```bash
PYTHONPATH=. /Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 2 3 4 5
```

If using the remote Ollama setup for `gpt-oss:120b`, the current workflow runs through the existing SSH-backed Ollama route used by the original project.

### 4. Aggregate completed cycle outputs

After the cycle run finishes:

```bash
PYTHONPATH=. /Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/python scripts/aggregate_cycle_outputs.py
```

### 5. Launch the Streamlit app

Because the app depends on the same environment as the source retrieval stack, launch it with the source repo environment:

```bash
/Users/rhea.chatterjeeyale.edu/rag-audio/.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Streamlit UI

The current UI includes:
- `Overview`
- `Fidelity`
- `PI Questions`
- `Evidence Browser`
- `Manual Units`
- `RAG Chat`

### RAG Chat

The chat tab supports:
- transcript-only retrieval
- transcript + manual retrieval
- evidence-only mode
- grounded answer mode using `gpt-oss:120b`

Recommended defaults in the app:
- transcript-only exploratory questions:
  - manual off
  - model answer on
  - `document weight = 1.0`
  - `topic weight = 0.0`
- manual + transcript fidelity questions:
  - manual on
  - model answer on
  - `document weight = 0.5`
  - `topic weight = 0.5`

## Fidelity Outputs

The current fidelity summary is an automated, retrieval-based alignment estimate.

For each session-topic pair, it reports:
- retrieved evidence count
- expected manual-unit count
- matched manual-unit count
- manual-unit coverage
- expected subsection count
- matched subsection count
- subsection coverage
- adherence score
- adherence label

Current adherence formula:

- `manual_unit_coverage = matched_manual_units / expected_manual_units`
- `subsection_coverage = matched_subsections / expected_subsections`
- `adherence_score = 0.6 * manual_unit_coverage + 0.4 * subsection_coverage`

Current labels:
- `high` if score `>= 0.66`
- `moderate` if score `>= 0.33`
- `low` otherwise

This should be described as a computational / automated fidelity proxy rather than a human-validated fidelity rating.

## PI-Question Outputs

PI-question rows currently include:
- query text
- retrieved evidence count
- `gpt-oss:120b` answer summary
- evidence refs
- manual units cited
- model-reported confidence
- full saved prompt
- raw JSON response

These outputs are intended to be evidence-constrained summaries, not independent ground-truth coding.

## Git / Repo Notes

- `data/derived/` is ignored
- generated logs such as `app/streamlit.log` are ignored

So the repo is focused on code, settings, and documentation rather than generated analysis artifacts.

Where to change behavior (quick map)
----------------------------------

If you're changing runtime behavior, here is a quick map of common changes and the files to edit.

- Retrieval, manual-unit parsing, and matching heuristics
  - Edit: `rag_audio_analysis/source_bridge.py`
  - Also check: `settings.ini` (`transcript_export`, `topic_matching`, `manual_parsing`) for tunable thresholds.

- High-level orchestration, output schema, and run flags
  - Edit: `scripts/run_cycle_analysis.py`
  - Also check: `settings.ini` (`cycle_analysis`) and the CLI flags (for example `--fidelity-topk`, `--fixed-fidelity-topk`).

- Fidelity scoring and labels
  - Edit: `scripts/run_cycle_analysis.py` (`summarize_fidelity(...)`) for the implementation; change weights/cutoffs in `settings.ini` under `[fidelity]`.

- PI prompt templates and model-backed fields
  - Edit: `scripts/run_cycle_analysis.py` for how prompts are composed and parsed; change prompt content in `settings.ini` under `[pi_questions]`.

- Streamlit UI labels, column display, and interactive defaults
  - Edit: `app/streamlit_app.py`; many UI defaults are read from `settings.ini` under `[ui]` and `[chat]`.

- Aggregation and summary table generation
  - Edit: `scripts/aggregate_cycle_outputs.py` (controls how per-cycle CSVs are combined into `summary/` tables).

- Small analyses and notebook-driven visuals
  - Edit: `analysis/*.ipynb` and scripts under `scripts/` (for reproducible, headless runs use the `.venv` and `nbconvert`).

This map is intentionally short; see `CONFIGURABLE_VALUES.md` and `DATA_AGGREGATION.md` for a longer inventory and provenance notes.
