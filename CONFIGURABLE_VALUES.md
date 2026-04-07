# Configurable Values Inventory

This file is an inventory of the values that currently drive the `rag-audio-analysis` codebase.

Important note:
- This is a **single audit/reference file**.
- The active pipeline now reads many of these values from:
  - [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini)
- Some legacy values and rule-based logic are **still hardcoded in Python** and are called out explicitly below.
- If we want to fully remove hardcoding, the next step would be to refactor the code to load these values from a YAML or TOML settings file.

## Active Pipeline

These are the values used by the current session-topic evidence pipeline.
For the active pipeline, the shared source of truth is now:
- [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini)

### Source paths
Loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [config.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/config.py).

- `SOURCE_ROOT`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio`
- `SOURCE_RAG_INDEX`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/rag_index`
- `SOURCE_BUILD_AND_QUERY`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/tools/build_and_query_rag.py`
- `SOURCE_META`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/rag_index/meta.json`
- `SOURCE_MANUAL`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/manual.txt`
- `SOURCE_TOPICS_CSV`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/manual_topics.csv`
- `SOURCE_TOPIC_LIST`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/rag_index/topic_list.json`
- `SOURCE_MANUAL_DOC_INDICES`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/rag_index/manual_doc_indices.json`
- `SOURCE_MANUAL_DOC_TOPIC_MAP`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/rag_index/manual_doc_topic_map.json`
- `SOURCE_TRANSCRIPTS_GLOB`
  - `/Users/rhea.chatterjeeyale.edu/rag-audio/audio/de-identified_transcripts/**/*.txt`

### Local project paths
Defined in [config.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/config.py).

- `PROJECT_ROOT`
  - `/Users/rheachatterjee/Documents/Playground/rag-audio-analysis`
- `DATA_DIR`
  - `data/`
- `TEMPLATES_DIR`
  - `data/templates/`
- `DERIVED_DIR`
  - `data/derived/`
- `APP_DIR`
  - `app/`
- `CYCLE_ANALYSIS_DIR`
  - `data/derived/cycle_analysis/`

### Cycle-analysis CLI defaults
Loaded from `settings.ini` through `run_cycle_analysis.py`.

- `cycles`
  - default: `1,2,3,4,5`
- `fidelity_topk`
  - default: `12` (used as explicit fallback when dynamic top-k is disabled)
- `question_topk`
  - default: `8`
- `fidelity_weight_doc`
  - default: `1.0` (session-fidelity currently prefers document weight only)
- `fidelity_weight_topic`
  - default: `0.0`
- `question_weight_doc`
  - default: `1.0`
- `question_weight_topic`
  - default: `0.0`
- `context_window`
  - default: `2`
- `ollama_model`
  - default: empty string (set to `gpt-oss:120b` to enable generation-backed adjudication)
- `ollama_ssh_host`
  - default: `rc2526@10.168.224.148`
- `ollama_ssh_key`
  - default: `~/.ssh/ollama_remote`
- `ollama_remote_bin`
  - default: `/usr/local/bin/ollama`
- `limit_topics`
  - default: `0`

### Fidelity query pattern
Loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py).

- query format:
  - `Session {session_num} {topic_label}`

### PI-question query patterns
Loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) into `PI_QUESTION_SPECS` in [run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py).

- `facilitator_reference`
  - `Session {session_num} {topic_label}. Facilitator introducing, teaching, reviewing, or cueing this topic.`
- `facilitator_demonstration`
  - `Session {session_num} {topic_label}. Facilitator modeling, guiding, demonstrating, or leading practice of this skill.`
- `participant_practice`
  - `Session {session_num} {topic_label}. Participant describing practicing this skill in session or individually at home.`
- `participant_child_home`
  - `Session {session_num} {topic_label}. Participant describing using this skill with their child at home.`

### Fidelity scoring formula
Weights and cutoffs are loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini); the formula is implemented in `summarize_fidelity(...)` in [run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py).

- `manual_unit_coverage`
  - `matched_manual_units / expected_manual_units`
- `subsection_coverage`
  - `matched_subsections / expected_subsections`
- `adherence_score`
  - `0.6 * manual_unit_coverage + 0.4 * subsection_coverage`
- `adherence_label`
  - `high` if score `>= 0.66`
  - `moderate` if score `>= 0.33`
  - `low` if score `< 0.33`

### Prompt-building limits
Loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py).

- manual units included in prompt
  - first `6`
- manual unit excerpt length in prompt
  - `220` chars
- evidence excerpt length in prompt
  - `400` chars
- generic display excerpt fallback
  - `500` chars

### Manual parsing and chunking
`max_words_per_unit` is loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini); the section-detection logic remains in [source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py).

- `get_structured_manual_units(max_words=220)`
  - manual is split by:
    - `Session X`
    - subsection headers
    - fallback size cap within section
- fallback section chunk size
  - `220` words

### Manual subsection rules
Defined in `MANUAL_SUBSECTION_RULES` in [source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py).

Recognized subsection patterns currently include:
- `Handouts`
- `Homework`
- `Audio Recordings`
- `Stress Discussion`
- `Discussion Points`
- `Review / Discussion`
- `Breathing Exercise`
- `Nutrition and Physical Activity`
- `Physical Activity and Nutrition`
- `Nutrition`
- `Physical Activity`
- `Welcome / Rules`
- `Program Information`
- `Rules of the Group`
- `Group Content / Structure`

Generic short header fallback currently classifies unlabeled headers into:
- `breathing_exercise`
- `physical_activity_and_nutrition`
- `discussion`
- `activity`
- `other_instructions`

### Topic retrieval defaults in source bridge
Defaults are loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py).

- `query_topic_evidence(...)`
  - default `topk=120`
  - default `weight_doc=0.3`
  - default `weight_topic=0.7`
- `query_evidence(...)`
  - default `topk=25`
  - default `weight_doc=0.5`
  - default `weight_topic=0.5`
  - default `transcript_only=True`
  - default `model_name=None`

### Transcript evidence export defaults
Loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [export_transcript_spans.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/export_transcript_spans.py).

- per-topic retrieval:
  - `topk=120`
  - `weight_doc=0.3`
  - `weight_topic=0.7`
- context expansion:
  - local transcript window around retrieved hit

### Topic matching thresholds
Loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py).

- no topic returned if best score `< 2.0`
- topic confidence:
  - `high` if score `>= 6.0`
  - `medium` if score `>= 3.5`
  - `low` otherwise

### Manual-unit matching logic
`manual_unit_min_overlap` is loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini); the token-overlap matching algorithm still lives in `infer_manual_unit_for_text(...)` in [source_bridge.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/source_bridge.py).

- topic-constrained candidate set is used first
- fallback to all manual units if no topic-matched candidates exist
- score:
  - number of overlapping tokens between transcript evidence window and manual unit token set
- no match returned if best overlap score is `0`

### Ollama execution
Defaults are loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through `call_ollama(...)` in [run_cycle_analysis.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/run_cycle_analysis.py).

- local mode:
  - `ollama run {model}`
- SSH mode:
  - `ssh -i {key} -o BatchMode=yes {host} "{remote_bin} run {model}"`

### Streamlit UI values
Core UI values are now loaded from [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) through [streamlit_app.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/app/streamlit_app.py).

- app title
  - `rag-audio-analysis`
- tabs
  - `Overview`
  - `Fidelity`
  - `PI Questions`
  - `Evidence Browser`
  - `Manual Units`
- evidence excerpt preview length in UI
  - `280` chars
- recognized cycle folder prefix
  - `PMHCycle`

### Streamlit launch values
These are currently used operationally, but they are not stored in Python config.

- host launch pattern
  - `.venv/bin/streamlit run app/streamlit_app.py --server.headless true --server.port 8501`
- port
  - `8501`

## Legacy / Older Scaffolding Still Present

These files still exist in the repo, but they are not the primary workflow anymore.

- [bootstrap_analysis_data.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/bootstrap_analysis_data.py)
- [generate_analysis_summaries.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/generate_analysis_summaries.py)
- [generate_coded_evidence.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/generate_coded_evidence.py)
- [generate_content_review_queue.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/generate_content_review_queue.py)
- [export_speaker_role_map.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/scripts/export_speaker_role_map.py)
- [coding_rules.py](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/rag_audio_analysis/coding_rules.py)

These older components still contain their own values and defaults, but they are no longer the main UI/pipeline path.

## Values That Are Still Hardcoded

The following important values are still hardcoded in code rather than fully loaded from one central settings file:

- manual subsection detection rules and generic header classification
- topic scoring algorithm itself
- manual-unit token-overlap matching algorithm itself
- Streamlit launch command used operationally on the Yale host
- legacy script defaults in older non-primary scripts

## Recommended Next Refactor

If the goal is to make the codebase truly “not hardcoded,” the next step should be:

1. extend [settings.ini](/Users/rheachatterjee/Documents/Playground/rag-audio-analysis/settings.ini) as new parameters appear
2. migrate remaining rule-based constants out of code where that still makes sense
3. decide whether the older legacy scripts should also be moved onto the shared settings file

The active pipeline is now settings-driven; the remaining work is mainly cleanup and deeper de-hardcoding of rule logic.
