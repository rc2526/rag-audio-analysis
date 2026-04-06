# Cycle-Level Manual-Session Fidelity Redesign

This document describes the recommended redesign of the fidelity workflow.

## Recommendation

If the main fidelity question is:

> How closely did facilitators follow the manual?

then the primary fidelity metric should be calculated at the level of:

- `cycle_id`
- `manual_session_num`

using manual units grouped by manual session, without requiring manual units to carry a single topic label.

This design fits the data better because the transcripts are **not reliably labeled by intervention session**. The manual is session-structured, but the transcript evidence is best treated as cycle-wide.

## Why the Current Topic-Linked Fidelity Is Brittle

The old fidelity pipeline is:

- session-topic based
- dependent on manual-unit topic assignment
- vulnerable to drift when manual-unit topic mapping changes

That creates problems like:

- a topic having only 1 expected manual unit
- a session-topic row showing 0 expected manual units after remapping
- fidelity denominators changing because of chunk-to-topic assignment rather than actual manual structure
- implied transcript-to-session matching that the source transcripts do not consistently support

## New Core Idea

Make fidelity compare:

- all manual units in a **manual session**
against
- transcript evidence retrieved from the **full cycle corpus**

This means:

- **expected manual units**
  - all manual units for `Session N`

- **observed manual units**
  - Session N manual units that are matched by transcript evidence retrieved from `PMHCycleX`

No manual-unit topic assignment is required for the main fidelity calculation.

## Revised Fidelity Workflow

For each:

- `cycle_id`
- `manual_session_num`

the revised workflow should:

1. look up the topics listed for that manual session in `manual_topics.csv`
2. run fidelity retrieval across the **whole cycle corpus** using those session-linked topics as helper queries
3. merge and deduplicate the retrieved transcript evidence rows
4. match each transcript evidence row to the best manual unit in that same manual session
5. compute coverage across manual units and subsections
6. write one cycle/manual-session fidelity row
7. preserve evidence-level rows for audit and review

## Retrieval Strategy

The fidelity query should remain session-aware, but transcripts should still be retrieved from the whole cycle rather than assumed to belong to one session-specific transcript file.

### Best practical version

- retrieve separately for each topic listed in `manual_topics.csv` for `Session N`
- merge the retrieved transcript evidence
- deduplicate evidence rows
- then score fidelity at the cycle/manual-session level

This keeps topic-aware retrieval as a helper while removing topic labels from the fidelity denominator.

## Manual Units

Under this redesign, `manual_units.csv` should primarily represent structure:

- `manual_unit_id`
- `manual_chunk_index`
- `source_file`
- `manual_week`
- `manual_section`
- `manual_subsection`
- `manual_text`
- `manual_text_short`
- `manual_home_practice`
- `manual_child_context`
- `manual_facilitator_instruction`
- `manual_participant_activity`
- `manual_quote_candidate`

The main fidelity metric should not require:

- `topic_id`
- `topic_match_score`

Those can remain optional exploratory fields if needed, but they should not drive the main fidelity denominator.

## Cycle-Level Manual-Session Fidelity Metrics

For each cycle/manual-session pair:

### 1. manual-unit coverage

```text
matched manual units / expected manual units
```

Where:

- expected manual units = all manual units for that manual session
- matched manual units = units matched by at least one retrieved transcript evidence row

### 2. subsection coverage

```text
matched subsections / expected subsections
```

Where subsections are things like:

- Homework
- Handouts
- Discussion
- Activity
- Breathing Exercise
- Physical Activity and Nutrition
- Other Instructions

### 3. evidence density

Useful supporting metric:

```text
retrieved evidence rows / expected manual units
```

This should not replace coverage, but it helps interpret sparse vs dense coverage across cycles.

### 4. optional facilitator-instruction coverage

Because some manual units are more central to protocol delivery than others, a future extension could separately report:

- coverage of units marked `manual_facilitator_instruction = 1`

## Recommended Adherence Score

Keep the current simple weighted score at first:

```text
adherence_score = 0.6 * manual_unit_coverage + 0.4 * subsection_coverage
```

Labels:

- `high` if `>= 0.66`
- `moderate` if `>= 0.33`
- `low` otherwise

Later, if desired, add:

- evidence density
- facilitator-instruction coverage

## New Output Tables

### 1. `session_fidelity_summary.csv`

One row per:

- `cycle_id`
- `manual_session_num`

Suggested columns:

- `cycle_id`
- `manual_session_num`
- `manual_session_label`
- `fidelity_query`
- `session_topic_ids`
- `session_topic_labels`
- `retrieved_evidence_count`
- `expected_manual_unit_count`
- `matched_manual_unit_count`
- `manual_unit_coverage`
- `expected_subsection_count`
- `matched_subsection_count`
- `subsection_coverage`
- `evidence_density`
- `adherence_score`
- `adherence_label`
- `matched_manual_unit_ids`
- `matched_subsections`
- `sample_session_ids`

### 2. `session_fidelity_evidence.csv`

One row per retrieved evidence item used in cycle/manual-session fidelity.

Suggested columns:

- `cycle_id`
- `manual_session_num`
- `manual_session_label`
- `analysis_mode`
  - `session_fidelity`
- `query_text`
- `source_topic_id`
- `source_topic_label`
- `retrieval_rank`
- `session_id`
- `speaker`
- `score_combined`
- `score_doc`
- `score_topic`
- `manual_unit_id_best_match`
- `manual_unit_match_score`
- `text`
- `excerpt`

### 3. Optional `session_fidelity_subsection_summary.csv`

One row per:

- `cycle_id`
- `manual_session_num`
- `manual_subsection`

Suggested columns:

- `cycle_id`
- `manual_session_num`
- `manual_subsection`
- `expected_manual_unit_count`
- `matched_manual_unit_count`
- `subsection_covered`

## Relationship to Topic Analysis

This redesign does **not** require removing topics from the project.

Topics should still be used for:

- transcript-content analysis
- PI-question analysis
- topic browsing in the UI
- optional exploratory views

The key change is:

- topics remain useful for retrieval and exploratory interpretation
- but the primary fidelity metric becomes a cycle-level manual-alignment measure

## UI Changes

The `Fidelity` tab should be refocused from session-topic rows to cycle/manual-session rows.

### Recommended Fidelity tab content

1. cycle/manual-session summary table
2. expected manual units for that manual session
3. matched manual units for that manual session
4. evidence snippets retrieved from the cycle corpus
5. optional subsection breakdown

## Transition Strategy

The safest implementation path is:

### Phase 1

Add the new cycle-level manual-session fidelity outputs **alongside** the current topic-level fidelity outputs.

This lets you compare:

- old topic-level fidelity
- new cycle-level manual-session fidelity

without deleting anything immediately.

### Phase 2

Update the UI so the new cycle/manual-session outputs are the primary fidelity view.

### Phase 3

If the new design works better, keep topic-level fidelity only as a legacy or exploratory layer.

## Why This Is Better

This redesign:

- removes dependence on brittle manual-unit topic assignment
- fits the actual transcript metadata better
- preserves auditability
- makes fidelity denominators more stable
- keeps topics available where they are actually most useful

## Immediate Implementation Path

1. generate `session_fidelity_summary.csv`
2. generate `session_fidelity_evidence.csv`
3. aggregate those outputs across cycles
4. make the `Fidelity` tab use the new files as the primary fidelity layer

That gives the project a more defensible answer to:

> How closely did facilitators follow the manual?
