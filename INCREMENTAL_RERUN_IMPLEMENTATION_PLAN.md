# Incremental Rerun Implementation Plan

This document describes a post-full-run refactor plan for making the analysis pipeline rerunnable at smaller scopes.

It is updated to match the current fidelity design:

- topic-level fidelity is being retired from the incremental rerun design
- cycle-level manual-session fidelity uses direct retrieval from `session_summaries.csv`
- session-fidelity evidence is no longer built from merged topic-helper windows

The goal is to support small iteration loops without rerunning the entire all-cycle pipeline.

## Goal

Make `scripts/run_cycle_analysis.py` support rerunning:

- one cycle
- one manual session
- one topic
- one PI question
- one mode (`session_fidelity` or `pi`)
- retrieval only vs LLM answer only

without touching unrelated outputs.

## Design Principle

Use different units of work for the two remaining analysis modes.

### Session-fidelity unit

- `cycle_id`
- `manual_session_num`

This reflects the new fidelity structure:

- retrieval uses the manual-session summary
- transcript evidence is pulled from the full cycle corpus
- matching is restricted to manual units from that manual session

### PI-question unit

- `cycle_id`
- `session_num`
- `topic_id`
- `question_id`

## New CLI Interface

Add the following flags to `scripts/run_cycle_analysis.py`.

### Target selection

- `--cycles 1 2 3`
- `--session-num 9`
- `--manual-session-num 9`
- `--topic-id mindful-parenting-skills`
- `--question-id participant_child_home`

Notes:

- `--session-num` applies to PI-question tasks
- `--manual-session-num` applies to session-fidelity tasks
- if only `--session-num` is supplied, it can also be used to constrain session-fidelity rows when the numeric session labels align

### Mode selection

- `--mode all`
- `--mode session-fidelity`
- `--mode pi`

Default:

- `--mode all`

### Execution behavior

- `--no-llm`
  - run retrieval and evidence generation only
- `--reuse-evidence`
  - reuse saved PI evidence when possible
- `--overwrite`
  - replace matching rows/files in scope
- `--skip-existing`
  - default behavior for already-completed scoped outputs

Optional later:

- `--dry-run`

## Task-Building Layer

Refactor the script so it first builds explicit task lists before any retrieval or model calls.

Suggested functions:

- `build_session_fidelity_tasks(...)`
- `build_pi_tasks(...)`

### Session-fidelity task

One task per:

- `cycle_id`
- `manual_session_num`

Inputs for each task:

- session summary from `data/inputs/session_summaries.csv`
- topic labels for that manual session
- manual units for that manual session

### PI task

One task per:

- `cycle_id`
- `session_num`
- `topic_id`
- `question_id`

## Split the Runner Into Explicit Pipelines

### 1. Session-fidelity pipeline

For each session-fidelity task:

- build the session-fidelity query from the new wrapper
- include:
  - `Manual Session {n}` anchor
  - session-specific priority cues
  - the saved session summary
- retrieve transcript evidence once across the full cycle corpus
- expand transcript windows
- match windows only against manual units from that manual session
- compute:
  - manual-unit coverage
  - subsection coverage
  - evidence density
  - adherence score
- write or replace one row in `session_fidelity_summary.csv`
- write or replace matching rows in `session_fidelity_evidence.csv`

Important:

- this pipeline should not depend on topic-level fidelity windows
- this pipeline should not inherit `source_topic_id` or `source_topic_label` semantics
- session-fidelity retrieval depth should remain:
  - `topk = expected_manual_unit_count`
  - where `expected_manual_unit_count = len(session_manual_units)`
- do not revert session-fidelity to a flat `topk = 12`

### 2. PI-question pipeline

For each PI task:

- build the question-specific retrieval query
- retrieve transcript evidence
- match manual units for that `session_num + topic_id`
- optionally call the LLM unless `--no-llm`
- write or replace one row in `pi_question_answers.csv`
- write or replace one entry in `pi_question_answers.json`
- write or replace matching `analysis_mode = pi_question` rows in `topic_evidence.csv`

## Output Replacement Rules

This is the most important implementation detail.

### `session_fidelity_summary.csv`

Primary key:

- `cycle_id`
- `manual_session_num`

When rerunning one session-fidelity target:

- remove any existing row with that key
- append the new row

### `pi_question_answers.csv`

Primary key:

- `cycle_id`
- `session_num`
- `topic_id`
- `question_id`

When rerunning one PI task:

- remove any existing row with that key
- append the new row

### `pi_question_answers.json`

Primary key:

- `cycle_id`
- `session_num`
- `topic_id`
- `question_id`

When rerunning one PI task:

- remove matching JSON object
- append the new one

### `topic_evidence.csv`

For PI rows, key should include:

- `analysis_mode = pi_question`
- `cycle_id`
- `session_num`
- `topic_id`
- `question_id`

When rerunning:

- delete matching rows for that scope
- append fresh rows

### `session_fidelity_evidence.csv`

Key should include:

- `analysis_mode = session_fidelity`
- `cycle_id`
- `manual_session_num`

When rerunning:

- delete matching rows for that scope
- append fresh rows

## Evidence Reuse for PI Questions

This is the biggest time-saver for prompt iteration.

If only PI prompt wording changes, retrieval should not have to rerun.

### Behavior with `--reuse-evidence`

For a PI task:

- first look for existing matching `topic_evidence.csv` rows
- require:
  - `analysis_mode = pi_question`
  - matching `cycle_id`
  - matching `session_num`
  - matching `topic_id`
  - matching `question_id`
- if found:
  - reuse those rows
  - skip retrieval
  - rebuild prompt
  - rerun only LLM answer generation

This enables a much faster prompt-iteration loop.

## Recommended Commands

### Full run

```bash
python scripts/run_cycle_analysis.py --cycles 1 2 3 4 5 --mode all
```

### One cycle only

```bash
python scripts/run_cycle_analysis.py --cycles 3 --mode all
```

### Session-fidelity only for one cycle

```bash
python scripts/run_cycle_analysis.py --cycles 3 --mode session-fidelity --overwrite
```

### PI questions only for one cycle

```bash
python scripts/run_cycle_analysis.py --cycles 3 --mode pi --overwrite
```

### One PI question across one cycle

```bash
python scripts/run_cycle_analysis.py \
  --cycles 3 \
  --mode pi \
  --question-id participant_child_home \
  --overwrite
```

### One topic in one session for PI questions

```bash
python scripts/run_cycle_analysis.py \
  --cycles 3 \
  --session-num 9 \
  --topic-id mindful-parenting-skills \
  --mode pi \
  --overwrite
```

### One manual session for session-fidelity only

```bash
python scripts/run_cycle_analysis.py \
  --cycles 3 \
  --manual-session-num 9 \
  --mode session-fidelity \
  --overwrite
```

### Retrieval-only PI debugging

```bash
python scripts/run_cycle_analysis.py \
  --cycles 1 \
  --session-num 9 \
  --topic-id mindful-parenting-skills \
  --mode pi \
  --question-id participant_child_home \
  --no-llm \
  --overwrite
```

### Prompt-only PI rerun from saved evidence

```bash
python scripts/run_cycle_analysis.py \
  --cycles 1 \
  --session-num 9 \
  --topic-id mindful-parenting-skills \
  --mode pi \
  --question-id participant_child_home \
  --reuse-evidence \
  --overwrite
```

## Recommended Implementation Order

### Phase 1

Add:

- `--mode`
- `--session-num`
- `--manual-session-num`
- `--topic-id`
- `--question-id`

and implement scoped task generation.

### Phase 2

Add row-level overwrite behavior for:

- `fidelity_summary.csv`
- `session_fidelity_summary.csv`
- `pi_question_answers.csv`
- `pi_question_answers.json`
- `topic_evidence.csv`
- `session_fidelity_evidence.csv`

### Phase 3

Add:

- `--no-llm`
- `--reuse-evidence`

### Phase 4

Add:

- `--dry-run`
- clearer progress logging
- optional per-mode timing output

### Phase 5

Add build-performance improvements that do not change retrieval semantics.

Priority optimizations:

- cache the embedding model once per run
- cache transcript metadata and transcript path lookup once per run
- cache the full manual-unit index once per run
- cache per-session manual-unit subsets once per run
- cache per-session manual-unit embeddings once per run
- reuse already-loaded transcript embedding arrays rather than reloading them unnecessarily

Important:

- do not add dedupe logic as part of this optimization phase
- do not reduce session-fidelity retrieval depth below `expected_manual_unit_count`
- the preferred speed strategy is better caching, not narrower retrieval

## Minimal Validation Plan

After implementation, test these cases:

1. rerun one session-fidelity target and confirm only one manual-session row and its evidence rows change
2. rerun one PI question and confirm only that question row changes
3. rerun with `--no-llm` and confirm evidence is written but answer fields are blank
4. rerun with `--reuse-evidence` and confirm no new retrieval is needed

## Main Benefit

With this plan in place:

- changing one PI question will not require rerunning all cycles
- changing one manual-session fidelity query will not require rerunning unrelated topics or PI outputs
- changing one topic/session PI question target will not require rerunning unrelated outputs
- prompt iteration becomes much faster once `--reuse-evidence` exists
- session-fidelity can stay methodologically broader with `topk = expected_manual_unit_count` while still becoming faster through caching

## Performance Notes

The desired performance strategy is:

- keep session-fidelity retrieval breadth aligned to the denominator
- avoid dedupe-focused redesign work for now
- improve speed by reducing repeated construction/loading work

Specific guidance:

- session-fidelity should continue to use:
  - `topk = len(session_manual_units)`
- PI questions should keep their own separate retrieval depth
- optimization effort should focus on:
  - caching manual-unit subsets
  - caching manual-unit embeddings
  - caching model and index artifacts
- optimization effort should not focus on:
  - transcript-window dedupe
  - retrieval pruning heuristics that narrow session-fidelity coverage

## Important Note

This plan is intended for implementation after the current long-running full-cycle refresh completes.
