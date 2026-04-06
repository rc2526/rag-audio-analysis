# rag-audio-analysis

Research-oriented analysis scaffold built on top of the existing `rag-audio` project.

This project treats the original `rag-audio` repository as the source of truth for:
- de-identified transcripts
- `manual.txt`
- `manual_topics.csv`
- `rag_index/meta.json` and related retrieval artifacts

The first version of this project focuses on:
- exporting structured CSV datasets for downstream coding and analysis
- preserving topic / manual / transcript provenance
- browsing those datasets in a lightweight Streamlit UI

## Project layout

- `rag_audio_analysis/`
  - shared config and bridge code for reusing `rag-audio`
- `scripts/`
  - dataset export and bootstrap scripts
- `data/templates/`
  - CSV header templates
- `data/derived/`
  - generated CSV datasets
- `app/`
  - minimal Streamlit browser

## Typical workflow

1. Ensure the source `rag-audio` repo has the transcripts and `rag_index` artifacts you want to analyze.
2. From this project root, run:
   - `python3 scripts/bootstrap_analysis_data.py`
3. Launch the UI:
   - `streamlit run app/streamlit_app.py`

## Current outputs

The bootstrap script currently creates:
- `data/derived/topic_catalog.csv`
- `data/derived/manual_units.csv`
- `data/derived/speaker_role_map.csv`
- `data/derived/transcript_spans.csv`
- `data/derived/coded_evidence.csv`
- `data/derived/quote_bank.csv`

The last two are initialized as empty coding-ready files with headers.
