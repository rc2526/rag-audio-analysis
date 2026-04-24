# Design: Interactive Top-k Sparse Heatmap (Transcript Windows x Structured Manual Units)

## Goal
Implement an interactive visualization (Streamlit) that shows how each transcript *window* matches to its top-k structured manual units by embedding similarity, with click-through inspection of:
- the transcript window text (re-embedded)
- the manual unit text (structured by session/subsection)
- similarity score, rank, and metadata

This is primarily for interpretability and audit: it should be easy to see which manual subsection a window aligns to, and to spot systematic mismatches.

## Key Requirements
- Implement in `rag-audio-analysis` (not `rag-audio`).
- Use *structured manual units* (session/subsection) for interpretability.
- Re-embed transcript windows (do not approximate by averaging per-turn embeddings).
- Keep the display readable at scale via a *top-k sparse heatmap* (k=20 initially).
- No pipeline behavior changes; this is an analysis/QA view.

## Non-goals (initial version)
- Changing how the upstream `rag-audio` index is built.
- Making the heatmap a "single source of truth" metric; it is an exploration / QA tool.
- Building a persistent vector store for windows; embed on demand with caching.

## Current State (relevant code in `rag-audio-analysis`)
- Transcript window expansion exists:
  - `rag_audio_analysis/source_bridge.py` `expand_transcript_context()`
  - controlled by `context_window` in `settings.ini`
- Structured manual units exist:
  - `rag_audio_analysis/source_bridge.py` builds manual units with `manual_week`, `manual_section`, `manual_subsection`, `manual_category`, `text`, and `matching_text`
  - `build_manual_unit_index()` / `get_manual_units_for_session()`
- Manual unit embedding index exists and is cached:
  - `rag_audio_analysis/source_bridge.py` `get_manual_unit_embedding_index()` embeds `matching_text` (or `text`)
- Transcript evidence retrieval exists and already feeds windows for fidelity/PI runs:
  - `scripts/run_cycle_analysis.py` `build_transcript_windows()` (retrieves, then expands context)

## Definitions
- Transcript turn: a single row in the upstream RAG index `meta.json` (one speaker segment).
- Transcript window: concatenation of the center turn plus +/- `context_window` neighboring turns from the same transcript file (`expand_transcript_context()`).
- Structured manual unit: one unit from `build_manual_unit_index()` / `get_manual_units_for_session()`, session/subsection aware.

## Design Overview
Add a new interactive view to `rag-audio-analysis` that:
1) Selects a *candidate set* of transcript windows to search over (configurable modes below).
2) Builds each candidate window's text using `expand_transcript_context()`.
3) Re-embeds each window text using the same embedding model used for manual units.
4) Selects the structured manual units to score against (typically session-scoped for interpretability).
5) Computes cosine similarity and keeps top-k transcript windows per manual unit (k=20).
6) Renders the resulting sparse matches as a heatmap-like view; click shows window + manual text + metadata.

## Candidate Transcript Window Modes (what we search over)
This heatmap is **manual-unit-driven**: for each structured manual unit, we pick the top-k matching transcript windows from a chosen candidate set.

Candidate sets (modes):

1) Entire corpus windows
- Candidate windows are built from all transcript turns in `SOURCE_RAG_INDEX` (transcript-only), across all cycles/sessions.

2) Cycle-only windows
- Candidate windows are built only from transcript turns whose `path` contains the selected `cycle_id`.
- This matches the previous analysis constraint where retrieval was restricted by `cycle_id` at the path level.

3) Cycle-only + session-aware query prefix (bias, not a hard filter)
- Candidate set is the same as mode (2).
- Similarity scoring uses a session-aware prefix on the *manual-unit side* to mimic prior retrieval queries, e.g. embed:
  - `f"Session {session_num} " + manual_unit.matching_text`
- This nudges matches toward the target session without forbidding cross-session hits.

4) Cycle + strict session-only windows (auditing)
- Candidate windows are built only from transcript turns in the selected `cycle_id` **and** whose inferred `session_id` matches `session_num`.
- Use this when you want a strict “Session N only” view (no cross-session leakage).

Common controls:
- `context_window` (default from `settings.ini`)
- `max_windows` (default 200) + sampling (first N or random sample with seed)

## Window Construction Visibility (UI requirement)
To avoid ambiguity about what is being embedded and scored, the UI should make the candidate window construction explicit:

- Controls
  - `context_window` (turns before/after the center turn)
  - Display derived target size: `turns_per_window_target = 2*context_window + 1` (edge windows can be smaller)
  - Optional safety cap: `max_window_chars` (truncate window text before embedding) if needed for performance

- Always-visible diagnostics (after building candidates)
  - `candidate_centers_count` (eligible center turns after mode filters)
  - `candidate_windows_count` (windows actually built after caps/sampling)
  - `min/avg/max_turn_count_per_window` (realized)
  - `avg/p95_chars_per_window` (or a simple token estimate) to understand embedding cost

- Explain window panel
  - Show one sample window with `center_doc_index`, `path`, included turn indices, and the exact concatenated `speaker: text` format that is fed into the embedder.

## Re-Embedding Transcript Windows (hard requirement)
Embed each `window_text` (multi-turn) using the same embedding model used elsewhere in `rag-audio-analysis`:
- Prefer using `rag_audio_analysis/source_bridge.py` helpers:
  - `get_embedding_model()` and `model.encode([...])`, or `encode_texts([...])`
- Ensure embeddings are L2-normalized so cosine similarity is a dot product.

Rationale:
- Window text is contextual; averaging per-turn vectors can blur the signal and is not acceptable for this view.

## Manual Unit Axis (structured, interpretable)
Columns are structured manual units (not upstream manual chunks). Two scopes:

A) Session-scoped (recommended default)
- If user selects `session_num`, restrict manual units to that session via `get_manual_units_for_session(session_num, topic_id=...)`.
- Benefit: smaller, interpretable axis; similarities are more meaningful.

B) Global manual units
- Use all structured manual units across sessions.
- Benefit: diagnose cross-session confusion.
- Risk: larger and harder to read.

Display metadata per manual unit:
- `manual_unit_id` (e.g. `MAN_0001`)
- `manual_week`, `manual_section`, `manual_subsection`, `manual_category`

## Top-k Matching
Given:
- window embeddings matrix `W` shape `(n_windows, d)`
- manual unit embeddings matrix `M` shape `(n_units, d)` (normalized)

Compute:
- similarities: `S = W @ M.T`
- for each window i, take `topk=20` indices of `S[i]`

Store a long-form table for plotting:
- window fields: `window_id`, `center_doc_index`, `path`, `turn_count`, `cycle_id`, `session_num`
- manual fields: `manual_unit_id`, `manual_week`, `manual_section`, `manual_subsection`, `manual_category`
- match fields: `score`, `rank`

## Visualization (Interactive Sparse Heatmap)
Preferred representation: Plotly scatter (sparse heatmap).
- x-axis: manual unit order (within scope) or categorical manual unit id
- y-axis: window row index (or center_doc_index)
- color: similarity score
- hover: rank, score, manual_subsection, identifiers

Interaction:
- Capture click selection to populate a detail panel.
- If robust click capture is needed, add `streamlit-plotly-events` as an optional dependency.
- Fallback: window selector + table of that window's top-k matches.

Detail panel shows:
- transcript window metadata + full window text
- manual unit metadata + full manual unit text (+ matching_text if distinct)

## Caching / Performance
Re-embedding windows is expensive; cache aggressively.

Caching strategy:
- Manual unit embeddings are already cached (`get_manual_unit_embedding_index()`).
- Cache window expansion keyed by `(cycle_id, session_num, context_window, selection_mode, seed, max_windows)`.
- Cache window embeddings keyed by `(window_text_hash, embedding_model)`.

Operational limits:
- default `max_windows=200`
- warn when increasing substantially

## Implementation (file locations)

### Logic module (keeps `source_bridge.py` stable)
- `rag_audio_analysis/window_manual_heatmap.py`
  - `HeatmapConfig` (all configurable inputs; no hard-coded analysis constants)
  - `build_topk_window_manual_heatmap(...)` (end-to-end orchestration)
  - Helpers:
    - `select_window_centers(...)`
    - `build_windows_from_centers(...)` (calls `expand_transcript_context()`)
    - `embed_texts_with_cache(...)` (re-embeds windows with a simple in-memory cache)
    - `topk_sparse_matches(...)` (manual-unit-driven top-k windows per unit)

### Streamlit entrypoint (separate app; does not modify the main UI)
- `app/streamlit_topk_window_manual_heatmap.py`
  - Run: `streamlit run app/streamlit_topk_window_manual_heatmap.py`
  - Uses Plotly for the sparse heatmap scatter and provides a non-click fallback selector when `streamlit-plotly-events` is not installed.

### Prebuilt heatmap support (new)
- `scripts/build_prebuilt_heatmaps.py` — CLI runner that builds and saves precomputed heatmaps for the canonical modes. By default it writes to:
  - `data/derived/topk_window_manual_heatmaps/{mode}.parquet` (or CSV fallback)
- Purpose: let users load a ready-made heatmap without re-embedding windows interactively.
- Controlled by existing settings defaults; use `--force` to rebuild existing files.

Additional: per-cycle prebuilt generation
- The build script now discovers cycles from the RAG index and can produce per-cycle prebuilt files for cycle-scoped modes.
- Per-cycle output naming:
  - `data/derived/topk_window_manual_heatmaps/cycle_only_{cycle}.parquet`
- CLI flags added:
  - `--cycles` — comma-separated cycle ids to build (overrides discovery)
  - `--modes` — optional subset of modes to build (comma-separated)
  - `--force` — rebuild existing files
- Discovery logic: script infers cycles from `get_rag_index_rows()` + `infer_cycle_id(path)` and sorts them naturally; `--cycles` overrides discovery.

### Save/load helpers (new)
- `rag_audio_analysis/window_manual_heatmap.py` now contains `save_matches()` and `load_matches()` helpers to persist/restore the long-form matches table and a small diagnostics JSON next to it.

### Streamlit UI changes (new)
- The Streamlit view now supports a compact "View" selector in the sidebar with options:
  - `Prebuilt: cycle_only` and `Build fresh`.
- Prebuilt views are loaded from `data/derived/topk_window_manual_heatmaps/` into `st.session_state` to avoid rebuilding on every UI interaction. This prevents accidental rebuilds when selecting rows or changing UI selectors.
- Hover content on the Plotly scatter has been reduced to only show:
  - `window_text` (the transcript window), `manual_unit_matching_text` (the canonical manual chunk), and `score` (similarity).

### Storage / format
- Primary storage for prebuilt matches is Parquet (`.parquet`); if Parquet/pyarrow is not available the build script falls back to CSV. A diagnostics file `{mode}_diagnostics.json` is saved alongside the matches file.


### Settings (all UI defaults + tunables)
- `settings.ini` `[topk_window_manual_heatmap]`
  - selection mode, scope, defaults, window/sampling caps, plotting parameters, and prefix-bias template

### Dependencies
- Declared in `requirements.txt`:
  - `numpy`, `plotly` (in addition to existing `pandas`, `streamlit`, `altair`)
- Optional (not required): `streamlit-plotly-events` for point-click capture.

## Testing / Verification (lightweight)
- Verify window expansion is deterministic for a fixed `(cycle_id, session_num, context_window)`.
- Verify similarity scores are within [-1, 1] and top-k ranks are sorted.
- Manual spot-check: clicking a few high-score cells should show semantically aligned text/subsection.

## Future Enhancements
- Facet/group manual axis by `manual_subsection` / `manual_category`.
- Add per-session normalization (z-score within session) toggle.
- Export CSV/Parquet of the long-form top-k table for downstream analysis.

## Recent changes (2026-04-22)
This project has received several small, focused updates to make the interactive view practical and reproducible. Key implemented changes are listed below so the design doc stays in sync with the codebase.

- Prebuilt heatmap support
  - `scripts/build_prebuilt_heatmaps.py` was added to precompute and persist heatmaps to `data/derived/topk_window_manual_heatmaps/`.
  - Prebuilt outputs are written as Parquet (primary) with a CSV fallback; a `{mode}_diagnostics.json` file is saved alongside each matches file.
  - CLI flags: `--modes` (comma list), `--cycles` (comma list, overrides discovery), and `--force` (overwrite existing files).

- Per-cycle prebuilt generation
  - The build script discovers cycles via `get_rag_index_rows()` + `infer_cycle_id(path)` and can emit per-cycle files for cycle-scoped modes. Examples:
    - `cycle_only_PMHCycle1.parquet` (actual filenames follow the pattern `{mode}_{cycle}.parquet`).
  - This was added to avoid the sampling artifact where an `entire_corpus` prebuilt file could contain only one cycle due to first-N deterministic selection.

- Save / load helpers
  - `rag_audio_analysis/window_manual_heatmap.py` now exposes `save_matches()` and `load_matches()` helpers to persist/restore the long-form matches table and diagnostics.

- Streamlit UI improvements
  - `app/streamlit_topk_window_manual_heatmap.py` gained a compact sidebar selector for prebuilt modes and a `Build fresh` flow.
  - Prebuilt matches are loaded into `st.session_state` to avoid accidental rebuilds on UI interactions.
  - Plotly hover content has been simplified to show exactly: `window_id`, `manual_unit_matching_text`, and `score` (formatted). Full texts remain available in the detail panel on click or selection.

- Sampling / reproducibility note
  - `entire_corpus` mode collects all non-manual RAG index rows and then applies the `max_windows` cap. If `random_sample=False` (default in many configs) it takes the first N centers in the order returned by `get_rag_index_rows()`, which can bias prebuilt outputs toward the earliest cycle(s) represented in the index.
  - Recommended options to avoid unintended bias:
    - Use `random_sample=True` with `sample_seed` for reproducible corpus-wide sampling.
    - Generate per-cycle prebuilt files and combine/analyze them.

New: per-cycle × per-session outputs
- The build script now accepts a `--sessions` flag to produce per-cycle × per-session prebuilt files where the session argument only limits manual units (columns), not transcript windows (rows).
- Filenames follow the pattern: `cycle_only_{cycle}_session{session}.parquet`.
- Behavior:
  - If both `--cycles` and `--sessions` are provided, the script emits per-cycle × per-session files (e.g. `cycle_only_PMHCycle2_session1.parquet`) — windows come from the cycle, manual units from the session.
  - If `--cycles` is provided but `--sessions` is omitted, the script emits per-cycle files as before: `cycle_only_{cycle}.parquet`.
  - If `--cycles` is omitted but `--sessions` is provided, the script emits session-only prebuilt files: `cycle_only_session{session}.parquet`.
- CLI additions: `--sessions=1,2,3` and `--dryrun` (prints planned filenames).
- Example:
```zsh
# Dry-run preview for session 1 across two cycles
python [build_prebuilt_heatmaps.py](http://_vscodecontentref_/3) \
  --cycles=PMHCycle1,PMHCycle2 \
  --sessions=1 \
  --dryrun

# Build for real:
python [build_prebuilt_heatmaps.py](http://_vscodecontentref_/4) \
  --cycles=PMHCycle1,PMHCycle2 \
  --sessions=1 \
  --force

### Recipe: build all per-cycle prebuilt files
Run these from the repo root (where `scripts/` is located). First discover cycle ids, then pass them to the build script.

1) Discover cycles (prints a comma-separated list):
```zsh
python - <<'PY'
from rag_audio_analysis.source_bridge import get_rag_index_rows, infer_cycle_id
rows = get_rag_index_rows()
cycles = sorted({infer_cycle_id(r.get('path','')) for r in rows if infer_cycle_id(r.get('path',''))})
print(','.join(cycles))
PY
```

2) Build prebuilt heatmaps for all cycles (replace the value for `--cycles` with the comma list printed above):
```zsh
# Example: replace PMHCycle1,PMHCycle2,... with the output from step (1)
python scripts/build_prebuilt_heatmaps.py \
  --cycles=PMHCycle1,PMHCycle2,PMHCycle3,PMHCycle4,PMHCycle5 \
  --force
```

### Additional implemented updates (2026-04-22)

The following features were added after the original notes above and are implemented in the current codebase:

- Manual × Manual dense similarity
  - A dense Manual×Manual similarity view was added to the Streamlit UI. Users can either load a precomputed similarity file from `data/derived/manual_unit_similarities/` or compute it on demand via the UI button `Compute manual×manual similarity (dense)`.
  - When computed the matrix is L2-normalized and saved (autosave) to `data/derived/manual_unit_similarities/manual_sim_autosave.parquet`. A CSV download option is available in the UI for portability.

- Streamlit UI: tab split and independent controls
  - The UI now exposes two explicit tabs: `Transcript × Manual` and `Manual × Manual`. Controls that only apply to the dense manual similarity (threshold) live in an independent slider `Min score to show (manual×manual)` in the sidebar so the two views are independent.
  - The previous checkbox that gated the Manual×Manual view was removed — the Manual tab is always visible.
  - The Build button was removed from the UI; the prebuilt discovery selector in the sidebar loads persisted prebuilt files from `data/derived/topk_window_manual_heatmaps/` into `st.session_state` instead of rebuilding interactively.

- Session-state caching and keys
  - To avoid re-computation and to make UI interactions snappy, the app caches several objects in `st.session_state`:
    - `matches` — loaded long-form matches table (prebuilt)
    - `diag` — diagnostics dict saved alongside prebuilt matches
    - `mode_loaded` — the loaded prebuilt key
    - `manual_sim_full` — cached full manual×manual DataFrame
    - `manual_sim_loaded_key` — filename key for the loaded/saved manual sim

- Fixes and housekeeping
  - A subtle indentation/nesting bug was fixed: the transcript render helper (`render_transcript_tab`) was accidentally defined nested inside the detail render function which caused transcript UI pieces (Matches table, selection) to leak into the Manual tab. Those helpers are now top-level functions and the tab isolation works correctly.
  - Hover content remains trimmed for readability; the detail panel still exposes full texts and metadata.

### How to compute / load manual×manual similarity

- Compute from the Streamlit UI: open the `Manual × Manual` tab and press `Compute manual×manual similarity (dense)` — the result will be cached in `st.session_state` and autosaved to `data/derived/manual_unit_similarities/manual_sim_autosave.parquet`.
- Load a precomputed matrix: place a `.parquet` or `.csv` matrix (square, indexed by `manual_unit_id`) in `data/derived/manual_unit_similarities/` and use the `Load precomputed similarity file` selector inside the Manual tab.

These additions keep the heavy compute offline or on-demand, make the two views independent, and improve reproducibility when exploring both the sparse Transcript×Manual matches and the dense Manual×Manual similarity space.

Optional: to avoid cycle-bias for a global sample, run a randomized entire-corpus build separately (not covered by this simplified script) using `random_sample=true` and a fixed seed.
