# AGENTS.md

## Project State

This repository is `rag-audio-analysis`.

The current working direction is:

- keep session-level manual-session fidelity
- retire topic-level fidelity as the intended future design
- keep PI questions topic-based
- enrich PI queries with topic glosses derived from session summaries

## Current Fidelity Design

Session-level fidelity is the primary fidelity mode.

It should work like this:

- one row per `cycle_id + manual_session_num`
- retrieval is run directly from the session summary in `data/inputs/session_summaries.csv`
- retrieval is against the full cycle transcript corpus
- retrieved transcript windows are matched only against manual units from that manual session
- outputs are written to:
  - `data/derived/cycle_analysis/PMHCycleX/session_fidelity_summary.csv`
  - `data/derived/cycle_analysis/PMHCycleX/session_fidelity_evidence.csv`

The old design that merged topic-level fidelity windows into session fidelity should not be reintroduced.

## Session-Fidelity Query Update

The session-fidelity query builder was updated to use a wrapped query with:

- `Manual Session {n}` anchor
- session-specific priority cues
- an "Avoid generic themes from other sessions" instruction
- the full session summary

Code locations:

- [rag_audio_analysis/source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py)
- [scripts/run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py)
- [scripts/backfill_session_fidelity_outputs.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/backfill_session_fidelity_outputs.py)

## Fidelity Rebuild Status

The session-fidelity rebuild completed successfully on April 6, 2026.

Updated outputs now exist for all five cycles:

- `PMHCycle1`
- `PMHCycle2`
- `PMHCycle3`
- `PMHCycle4`
- `PMHCycle5`

Aggregate summaries were also regenerated:

- [data/derived/cycle_analysis/summary/summary_session_fidelity_by_cycle.csv](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/data/derived/cycle_analysis/summary/summary_session_fidelity_by_cycle.csv)

Spot check result:

- saved `fidelity_query` in session-fidelity summaries now contains the tightened wrapper
- saved `query_text` in session-fidelity evidence matches that wrapper
- `source_topic_label` is blank in session-fidelity evidence, which is expected

## PI Question Direction

PI questions should remain topic-based.

Do not convert PI retrieval to full session-summary retrieval by default.

Instead, PI queries now include:

- session number
- topic label
- a gloss-style topic definition derived from the session summary
- the question label
- the original topic-based retrieval focus
- short session summary context

This keeps PI retrieval topic-anchored while adding richer context.

Current PI question set now uses a single combined facilitator item instead of separate
`facilitator_reference` and `facilitator_demonstration` questions:

- `facilitator_delivery`
  - "How do facilitators introduce, reinforce, or demonstrate this topic or skill?"
- `participant_practice`
- `participant_child_home`

This change was made because the old facilitator-reference wording implied frequency/counting
that the retrieval method does not support well, and the old two-question facilitator split
had avoidable overlap.

Main code location:

- [scripts/run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py)

## Topic Definitions

Approved topic definitions are now stored in:

- [data/derived/topic_catalog.csv](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/data/derived/topic_catalog.csv)

The shared source of truth for those definitions is:

- [rag_audio_analysis/source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py)

A preview artifact is also kept at:

- [data/derived/topic_definition_preview_v3.csv](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/data/derived/topic_definition_preview_v3.csv)

These definitions were promoted from preview into the stored topic catalog on April 6, 2026 and should now be reused by:

- the Overview tab topic map
- PI-query enrichment
- future topic-catalog exports

## Topic Definitions In UI

The Overview tab’s “Topic map by session” table reads and displays the stored `topic_definition` column from `topic_catalog.csv`.

Important:

- this is no longer UI-only
- the UI should reflect the stored topic catalog definitions directly
- if definitions look wrong in the app, regenerate the topic catalog rather than patching UI-only display logic

UI file:

- [app/streamlit_app.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/app/streamlit_app.py)

## Incremental Rerun Plan

There is now a saved implementation note for the future rerun refactor:

- [INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md)

Important design constraint:

- future rerun support should assume session-level fidelity only
- topic-level fidelity should not be part of the intended long-term rerun architecture

Implementation status update:

- `scripts/run_cycle_analysis.py` now supports targeted reruns with:
  - `--mode all|fidelity|pi`
  - `--session-num`
  - `--topic-id`
  - `--question-id`
  - merge-safe partial rewrites by default
  - `--overwrite` when a full replacement is intended
- fidelity now uses dynamic top-k by default:
  - `topk = expected_manual_unit_count`
  - use `--fixed-fidelity-topk` only if the older fixed-value behavior is intentionally needed

## Retrieval Weighting Defaults

Session fidelity now defaults to pure document similarity:

- `fidelity_weight_doc = 1.0`
- `fidelity_weight_topic = 0.0`

This is intentional. Session fidelity no longer uses topic-based fidelity logic, so the old
`0.5 / 0.5` blend was conceptually stale.

PI questions currently remain:

- `question_weight_doc = 1.0`
- `question_weight_topic = 0.0`

Rationale:

- PI queries are already enriched with session number, topic label, stored topic definition,
  and short session-summary context
- for now, that richer query text is preferred over adding generic topic-embedding weighting

## Remote Ollama Runtime

Important runtime note:

- the analysis repo on Yale does **not** have a local `ollama` binary on PATH
- model-backed PI answering is intended to run by SSHing from the Yale analysis host to the
  separate Ollama host configured in [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini)

Current configured remote Ollama settings:

- `ssh_host = rc2526@10.168.224.148`
- `ssh_key = ~/.ssh/ollama_remote`
- `remote_bin = /usr/local/bin/ollama`
- `default_model = gpt-oss:120b`

Validated on April 6, 2026:

- Yale can SSH to the configured Ollama host with the configured key
- `/usr/local/bin/ollama` exists on the remote host
- a tiny `ollama run gpt-oss:120b` call completed successfully from Yale

One observed behavior:

- `gpt-oss:120b` emits visible "thinking" text in raw CLI output before the final answer
- the pipeline still successfully parsed a JSON answer in the smoke test, but this is worth
  remembering if model-output parsing is revisited later

## Ollama Smoke Test Result

A one-row end-to-end PI smoke test completed successfully on Yale using:

- `--cycles 1`
- `--session-num 1`
- `--topic-id role-of-stress-in-body-weight-and-food-intake`
- `--question-id facilitator_delivery`
- `--mode pi`
- `--overwrite`
- `--ollama-model gpt-oss:120b`

What was confirmed:

- `query_text` saved correctly
- `prompt_text` saved correctly
- `answer_summary` saved correctly
- `raw_response` saved correctly
- the JSON companion file also contained the full prompt and parsed answer object

This means the remote-model-backed PI pipeline is working end to end.

## Full Build Status

As of the latest update in this session:

- the full Yale build was relaunched with:
  - `scripts/run_cycle_analysis.py --cycles 1 2 3 4 5 --ollama-model gpt-oss:120b`
- it is writing progress to:
  - [data/derived/cycle_analysis/full_build_ollama.log](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/data/derived/cycle_analysis/full_build_ollama.log)

Important operational note:

- the full Ollama-backed build is slow
- output progress is easier to verify by checking file timestamps and partial cycle outputs than by relying on the sparse log alone
- after the build completes, `scripts/aggregate_cycle_outputs.py` still needs to be rerun so the app reflects final outputs

## Runtime Performance Note

Important runtime clarification:

- running `--cycles 1` is still a full `PMHCycle1` pass, not a single-session run
- `PMHCycle1` currently spans 12 manual sessions and 46 topics
- under the current design, that implies roughly:
  - 46 topic-level fidelity retrieval passes
  - 12 session-level fidelity retrieval passes
  - 138 PI question passes (`46 topics x 3 questions`)
- cycle outputs are written at end-of-cycle, so long periods with no changed CSV timestamps do not necessarily mean the run is stuck

Observed April 6, 2026:

- `PMHCycle1` reruns remained slow even when launched alone because the cycle still contains many retrieval and PI-answer steps
- the process was observed alternating between:
  - high local CPU / MPS retrieval work on Yale
  - remote Ollama calls for PI answering

## Retrieval Optimization Applied

A low-risk optimization was applied to:

- [rag_audio_analysis/source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py)

Change:

- `query_evidence()` now skips loading topic embeddings and skips document-topic similarity computation when `weight_topic <= 0`

Why this is safe:

- current configured defaults are:
  - `fidelity_weight_doc = 1.0`
  - `fidelity_weight_topic = 0.0`
  - `question_weight_doc = 1.0`
  - `question_weight_topic = 0.0`
- so the skipped topic-score path was not contributing to ranking under the active settings

Sync status:

- the same patch was copied to the Yale-hosted repo at:
  - `/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/rag_audio_analysis/source_bridge.py`

## Ollama Confirmation During Cycle Runs

Confirmed during the Yale `PMHCycle1` rerun:

- PI answering is using the configured remote Ollama path
- the Yale analysis process spawned child commands of the form:
  - `ssh -i /Users/rhea.chatterjeeyale.edu/.ssh/ollama_remote -o BatchMode=yes rc2526@10.168.224.148 /usr/local/bin/ollama run gpt-oss:120b`

This means:

- retrieval is still performed locally on the Yale analysis host
- PI answer generation is performed by the remote Ollama host

## Practical Execution Guidance

For reliable iteration:

- do not assume `--cycles 1` is a quick smoke test; it is still a substantial full-cycle job
- if faster inspection is needed, prefer narrower targeted reruns such as:
  - `--session-num`
  - `--topic-id`
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

It currently behaves more like a qualitative evidence question than a count/rate estimate.

If revisited later, either:

- reword it to match retrieval-based evidence
- or build a separate frequency-style metric outside the current PI QA setup

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
