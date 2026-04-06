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

- [rag_audio_analysis/source_bridge.py](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/rag_audio_analysis/source_bridge.py)
- [scripts/run_cycle_analysis.py](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/scripts/run_cycle_analysis.py)
- [scripts/backfill_session_fidelity_outputs.py](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/scripts/backfill_session_fidelity_outputs.py)

## Fidelity Rebuild Status

The session-fidelity rebuild completed successfully on April 6, 2026.

Updated outputs now exist for all five cycles:

- `PMHCycle1`
- `PMHCycle2`
- `PMHCycle3`
- `PMHCycle4`
- `PMHCycle5`

Aggregate summaries were also regenerated:

- [data/derived/cycle_analysis/summary/summary_session_fidelity_by_cycle.csv](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/data/derived/cycle_analysis/summary/summary_session_fidelity_by_cycle.csv)

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

Main code location:

- [scripts/run_cycle_analysis.py](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/scripts/run_cycle_analysis.py)

## Topic Definitions In UI

The Overview tab now shows draft topic definitions even though `topic_catalog.csv` still has blank `topic_definition` values.

Important:

- this is currently UI-only
- the app derives gloss-style topic definitions at display time from `topic_label + session_summary`
- it does not rewrite `data/derived/topic_catalog.csv`

UI file:

- [app/streamlit_app.py](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/app/streamlit_app.py)

## Incremental Rerun Plan

There is now a saved implementation note for the future rerun refactor:

- [INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md](/Users/rhea.chatterjeeyale.edu/rag-audio-analysis/INCREMENTAL_RERUN_IMPLEMENTATION_PLAN.md)

Important design constraint:

- future rerun support should assume session-level fidelity only
- topic-level fidelity should not be part of the intended long-term rerun architecture

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

## Suggested Next Work

Likely next useful engineering tasks:

- implement the incremental rerun refactor in `scripts/run_cycle_analysis.py`
- optionally persist polished topic definitions into `topic_catalog.csv` instead of only generating them in the UI
- refine the facilitator-reference PI question wording so it matches the method
