# Methodology — Template

Use this document as a starting point for the Methodology section in a paper describing the rag-audio-analysis pipeline. Fill in bracketed placeholders (e.g., [N], [MODEL], [DATE]) with project-specific details.

## 1. Overview

This study uses an automated retrieval-augmented analysis pipeline to estimate fidelity of a manualized intervention and to generate short, evidence-grounded question summaries (PI questions) from de-identified session transcripts. The pipeline reuses the original `rag-audio` retrieval index and augments it with structured manual units, automated matching, and model-backed adjudication where appropriate.

Key components:
- Source transcripts and manual items (the `rag-audio` repo as source of truth).
- Embedding-backed retrieval (sentence-transformers RAG-style index).
- Session-topic fidelity estimation (retrieval + manual-unit matching).
- PI-question summarization (model-backed short answers constrained to retrieved evidence).
- Aggregation of per-cycle outputs into publication-ready summary tables.

## 2. Data

- Source: de-identified transcripts and manual materials maintained in the `rag-audio` repository. The pipeline assumes the source contains:
  - Full program transcripts (de-identified)
  - `manual.txt` and `manual_topics.csv`
  - Pre-built retrieval artifacts under `rag_index/`

- Derived artifacts (produced by this pipeline) are written to `data/derived/` and include:
  - `topic_catalog.csv`, `manual_units.csv`, `transcript_spans.csv`
  - Per-cycle outputs: `data/derived/cycle_analysis/PMHCycle<k>/fidelity_summary.csv`, `pi_question_answers.csv`, `pi_question_answers.json`, `topic_evidence.csv`
  - Aggregated summaries under `data/derived/cycle_analysis/summary/`

- Inclusion/exclusion: describe how transcripts/sessions were selected for analysis (e.g., cycles 1..5, sessions with complete transcripts). Specify any filtering by date, completeness, or participant characteristics.

## 3. Manual unit extraction

- The manual (text) is parsed into structured units by session and subsection (e.g., Homework, Handouts, Discussion, Activity, Breathing Exercise, Physical Activity and Nutrition, Other Instructions).
- Each manual unit is assigned identifiers (`manual_unit_id`) and metadata such as session number and subsection.
- These manual units are stored in `data/derived/manual_units.csv` and used to constrain and evaluate retrieval results.

Implementation notes / reproducibility
- The parser and unit construction code lives in `rag_audio_analysis/source_bridge.py` and is parameterized by `settings.ini` values under `[manual_parsing]`.

## 4. Retrieval and ranking

- Retrieval uses a pre-built RAG-style index (sentence-transformers embeddings) to return transcript windows most similar to a query.
- Two main retrieval modes:
  - Fidelity retrieval: session-aware query of the form `Session {n} {topic_label}`. Defaults: `fidelity_weight_doc = 1.0`, `fidelity_weight_topic = 0.0`, `fidelity_topk = 12` (see `settings.ini` under `[cycle_analysis]`).
  - PI-question retrieval: question-specific prompts enriched with session glosses and topic definitions. Defaults: `question_weight_doc = 1.0`, `question_weight_topic = 0.0`, `question_topk = 8`.

- Retrieval implementation details are in `rag_audio_analysis/source_bridge.py`. The index is expected at the `rag-audio` repo's `rag_index/` directory declared via the `source_root` setting.

## 5. Matching retrieved evidence to manual units

- Each retrieved transcript window is matched to the best manual unit by embedding similarity and heuristics implemented in `source_bridge.py`.
- Session-topic fidelity is estimated by counting matched manual units and matched manual subsections.
- Adherence score formula (default):
  - `adherence_score = 0.6 * manual_unit_coverage + 0.4 * subsection_coverage`
  - labels: `high` (>= 0.66), `moderate` (>= 0.33), `low` (otherwise)

## 6. PI-Question summarization and model usage

- For each session-topic pair the pipeline runs 3 PI questions (facilitator delivery, participant practice, participant-child home practice) and returns evidence-grounded JSON answers.
- By default model-backed PI summaries are generated via a remote Ollama host. The default model used in this project is `gpt-oss:120b` (see `settings.ini` under `[ollama]`).
- The code path for PI model calls uses the `--ollama-model` flag to `scripts/run_cycle_analysis.py`; fidelity adjudication (session-level generation grade) requires `--fidelity-ollama-model`.

Model runtime and remote execution
- Remote Ollama configuration (example values found in `settings.ini`):
  - `ssh_host = rc2526@10.168.224.148`
  - `ssh_key = ~/.ssh/ollama_remote`
  - `remote_bin = /usr/local/bin/ollama`
  - `default_model = gpt-oss:120b`
- The code executes remote calls via SSH and runs `/usr/local/bin/ollama run <model>` on the remote host. Ensure the key and host connectivity are working before executing production runs.

Safety, parsing, and fallback
- Model responses are first parsed as strict JSON. If parsing fails the pipeline attempts embedded-JSON extraction and fallback heuristics to create a short fallback answer. All raw responses are saved for audit in `pi_question_answers.json`.

## 7. Fidelity adjudication (generation grade)

- Session-level generation-grade adjudication (auto-generated summary and grade) is optional. It is triggered when `--fidelity-ollama-model` is supplied to `scripts/run_cycle_analysis.py`.
- The adjudication prompt asks the model to judge whether retrieved evidence reflects delivery of the manual session scope, returning `adjudication_summary`, `adjudication_label` and `adjudication_confidence` fields.
- If no fidelity-ollama-model is provided, the `adjudication_*` columns remain empty (this explains missing generation grades in prior runs).

## 8. Heuristic recovery and human-in-the-loop review

- The pipeline includes support for recovering non-JSON or malformed model outputs via a multi-pass heuristic extraction (embedded JSON extraction, tolerant parsing, and short-answer refinement). Recovered proposals are stored in a review artifact prior to application.
- The repository contains scripts to produce a review CSV/JSON, apply fixes with timestamped backups, and undo automated fixes if necessary.

## 9. Aggregation and downstream analysis

- After per-cycle runs complete, `scripts/aggregate_cycle_outputs.py` concatenates per-cycle CSVs and produces summary tables under `data/derived/cycle_analysis/summary/`.
- Notebooks in `analysis/` (for example `analysis/pi_confidence_by_topic.ipynb` and `analysis/pi_confidence_by_question.ipynb`) generate publication-ready charts (PNG/SVG) and CSVs for per-topic and per-question confidence summaries.

## 10. Reproducible run commands

Minimal reproducible sequence (assuming a project venv and source repo present):

```bash
# from repository root
export PYTHONPATH="$PWD"
.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 --mode fidelity --session-num 1 --overwrite
# optionally run PI generation (uses --ollama-model)
.venv/bin/python scripts/run_cycle_analysis.py --cycles 1 --mode pi --session-num 1 --ollama-model gpt-oss:120b
# aggregate results
.venv/bin/python scripts/aggregate_cycle_outputs.py
# launch the Streamlit app (uses source venv)
.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Notes:
- To produce session-level adjudication (Generation Grade), include `--fidelity-ollama-model gpt-oss:120b` in the fidelity run command.
- For targeted runs use flags `--session-num`, `--topic-id`, `--question-id`; the system supports merge-safe partial rewrites by default unless `--overwrite` is passed.

## 11. Evaluation metrics and output tables

- Report the per-session-topic metrics produced by `fidelity_summary.csv` (manual-unit coverage, subsection coverage, adherence_score, adherence_label) and include aggregated summaries from `summary/` (e.g., `summary_session_fidelity_by_cycle.csv`).
- For PI questions report: proportion of rows with an answer, proportion with evidence refs, and model-reported confidence buckets (high/medium/low); these are produced by `aggregate_cycle_outputs.py`.

## 12. Reproducibility checklist (for the paper)

- list source repo commit hash / dataset snapshot used
- state the `settings.ini` values (include appendix table or a copy of the relevant sections)
- include the exact model name and remote host used for generation (e.g., `gpt-oss:120b` via Ollama host `rc2526@10.168.224.148`)
- include the exact commands used to run the pipeline (as above)
- include notes about which fields are auto-generated vs human-coded (e.g., `adjudication_*` are auto-generated when the fidelity model is provided)

## 13. Limitations and ethics

- The automated alignment is a proxy and does not replace human manual fidelity coding; report it as a computational estimate and discuss potential false positives/negatives due to retrieval errors or ambiguous language in transcripts.
- Discuss privacy and data governance for de-identified transcripts and any institutional approvals.

## 14. Example text snippet for a Methodology subsection

"We built a retrieval-augmented analysis pipeline that reuses a sentence-transformer-based index from the original `rag-audio` project. We parsed the program manual into session-level manual units and used embedding similarity to match retrieved transcript windows to those units. For each session-topic pair we computed an adherence score that mixes manual-unit coverage and subsection coverage (0.6/0.4 weighting). For evidence summaries of three targeted PI questions per topic we ran a remote Ollama-backed model (`gpt-oss:120b`) and saved raw responses and parsed JSON. Aggregated tables and visualization code are available in `data/derived/cycle_analysis/summary/` and `analysis/` notebooks to reproduce reported results."

---

If you want, I can:
- produce a condensed 1-page Methodology suitable for submission word limits;
- generate a short Appendix YAML with the exact `settings.ini` keys used in the reported run;
- or fill in placeholders with concrete values (commit hash, run date, cycles/sessions analyzed) if you provide them.

## Supplement (templates & appendix)

Use the templates below as a supplement or appendix for a manuscript. Replace bracketed tokens with project-specific values.

1) Topic list and definitions by session (table template)

| Session | Topic ID | Topic label | Short definition |
|---:|---|---|---|
| 1 | TOP_0001 | Mindful Breathing | Brief definition / learning objectives for this topic. |
| 1 | TOP_0002 | Parenting Praise | Brief definition / learning objectives for this topic. |
| 2 | TOP_0003 | Positive Routines | ... |

Notes: export the CSV used by the pipeline from `data/derived/topic_catalog.csv` and include a small representative selection in the appendix.

2) Manual unit examples (representative excerpt)

```
manual_unit_id: MAN_0001
session: 1
subsection: Homework
text: "Practice mindful breathing for 3 minutes each day with your child. Encourage noticing breath and counting breaths."
topic_id: TOP_0001

manual_unit_id: MAN_0002
session: 1
subsection: Activity
text: "Demonstrate praising child behavior immediately and specifically using 'I like when you...'
topic_id: TOP_0002
```

3) PI prompt (model prompt template)

```
You are analyzing intervention transcripts using only the retrieved evidence and the short manual-unit excerpts provided.
Return valid JSON only.

Cycle: {cycle_id}
Session: Session {session_num}
Topic: {topic_label}
Question: {question_label}

Relevant manual units:
- {manual_unit_id} | {manual_subsection} | {manual_excerpt}

Retrieved evidence windows (E1..En):
- E1 | session_id={session_id} | {evidence_excerpt}

Respond with JSON exactly:
{
  "answer_summary": "short paragraph",
  "evidence_count": 0,
  "evidence_refs": ["E1"],
  "manual_unit_ids": ["MAN_0001"],
  "confidence": "low|medium|high",
  "confidence_explanation": "short sentence"
}

Base the answer only on the evidence shown above. If evidence is weak, say so.
```

4) PI query text (how queries are constructed)

Template:

```
Session {session_num} {topic_label}. {question_template}

Example (facilitator delivery):
Session 1 Mindful Breathing. Facilitator introducing, teaching, reviewing, cueing, modeling, guiding, demonstrating, or leading practice related to this topic or skill.
```

5) Fidelity query text (session-aware fidelity retrieval)

Template:

```
Session {session_num} {topic_label}

This short, session-aware query is used to retrieve transcript windows that are likely to contain delivery of the manual topic in the target session.
```

6) Fidelity adjudication prompt (model prompt template for adjudication)

```
You are adjudicating manual adherence from retrieved transcript evidence only.
Return valid JSON only.

Cycle: {cycle_id}
Session: Session {session_num}
Scope: {scope_label}
Task: {scope_description}

Expected manual units:
- {manual_unit_id} | {manual_subsection} | {manual_excerpt}

Retrieved evidence windows (E1..En):
- E1 | session_id={session_id} | {evidence_excerpt}

Respond with JSON exactly:
{
  "adjudication_summary": "short paragraph",
  "adherence_label": "high|moderate|low",
  "evidence_refs": ["E1"],
  "manual_unit_ids": ["MAN_0001"],
  "confidence": "low|medium|high"
}

Base the judgment only on the retrieved evidence and the expected manual units shown above.
```

7) `settings.ini` keys (appendix excerpt)

Include the following keys and their values in an appendix so reviewers can reproduce runs. Fill in values used in the reported run.

[cycle_analysis]
fidelity_topk = 12  # number of windows to retrieve for fidelity
question_topk = 8   # number of windows to retrieve for PI questions
fidelity_weight_doc = 1.0
fidelity_weight_topic = 0.0
question_weight_doc = 1.0
question_weight_topic = 0.0

[fidelity]
manual_coverage_weight = 0.6
subsection_coverage_weight = 0.4
adherence_high_cutoff = 0.66
adherence_moderate_cutoff = 0.33

[prompting]
manual_units_in_prompt = 6
manual_excerpt_chars = 220
evidence_excerpt_chars = 400
display_excerpt_chars = 500

[ollama]
default_model = gpt-oss:120b
ssh_host = rc2526@10.168.224.148
ssh_key = ~/.ssh/ollama_remote
remote_bin = /usr/local/bin/ollama

Appendix note: provide the commit hash and the exact `settings.ini` used (or include the `settings.ini` excerpt) so the run can be reproduced exactly.


