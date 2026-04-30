## Design: Embeddings map / scatter for topic_window_heatmaps

Purpose
- Capture a short, actionable design for producing 2D/3D embeddings maps or scatter plots from the data in `data/derived/topic_window_heatmaps` so we can visually explore window-level similarity and manual-unit clustering.

When this is useful
- Quick exploratory analysis of how transcript windows and manual units cluster by semantic similarity.
- Validate that manual-unit matches are reflected as nearby points in embedding space.
- Produce interactive visuals for investigators to inspect clusters, outliers and evidence snippets.

Data preconditions (what you must confirm before plotting)
- Files in `data/derived/topic_window_heatmaps` are generally Parquet/CSV. Confirm one file contains either:
  - Precomputed embeddings for each window (columns like `embedding`, `vec`, `vector`, `embedding_*`), or
  - Raw `text`/`matching_text`/`excerpt` fields that can be encoded with the project encoder (`encode_texts()` in `rag_audio_analysis/source_bridge.py`).
- Also useful columns: `transcript_id`, `window_id`, `manual_unit_id`, `topic_id`, `score`, `path`, `session_num`.

High-level pipeline
1. Inspect: open one representative file in the folder and confirm presence of embeddings or text.
2. Extract points: build a table of rows to visualize with columns: `id`, `text`, `embedding` (numpy array), plus metadata (topic/manual/session/score).
3. (Optional) Compute embeddings: if only texts present, use the project's encoder (`encode_texts()`) to compute embeddings so visualization matches retrieval semantics.
4. Normalize: L2-normalize embeddings if you want cosine similarities to map to Euclidean distances.
5. Dimensionality reduction:
   - PCA (2 components) to get a baseline.
   - UMAP (2D/3D) for interactive clustering (recommended for local+global structure).
   - t-SNE for local structure; consider after a PCA pre-reduction.
6. Plot:
   - Static: matplotlib/seaborn for quick checks.
   - Interactive: Plotly/Altair for hover tooltips showing `text`, `manual_unit_id`, `score`, `transcript_id`.
   - Streamlit: embed the interactive chart with controls (color by `topic_id` / `manual_unit_id`, filter by `session_num`, adjust UMAP params).

Visualization choices and UX
- Color encoding: use `manual_unit_id` (categorical) or `score` (continuous) or `topic_id`.
- Size: emphasize high-score points with larger markers.
- Tooltips: show snippet text, full path, session label, manual unit id.
- Sampling: for very large datasets (tens of thousands of points) use sampling, density aggregation (datashader) or WebGL-backed scatter (Plotly `scattergl`).

Parameters to tune
- UMAP: `n_neighbors` (locality), `min_dist` (compactness), `metric` (cosine preferred if embeddings normalized).
- t-SNE: perplexity and iteration count; pre-reduce with PCA to ~50 dims.
- PCA: quick global structure check; keeps runtime minimal.

Reproducibility and caching
- Cache embeddings and the reduced coordinates to disk (Parquet / npz) keyed by source file and reduction params so re-runs are fast.
- Record encoder version (model name + revision) and UMAP/t-SNE parameters in metadata.

Edge cases & gotchas
- Vector column parsing: vectors may be stringified ("[0.1,0.2]") — parse carefully and validate dtype/shape.
- Mixed encoders: embeddings computed with a different model may not match retrieval distance semantics; prefer the project's encoder.
- Very short or duplicate texts can distort clusters; consider deduping or collapsing duplicates with counts.

Next steps (implementation options)
- Minimal: provide a one-off Jupyter/Streamlit script that reads files, uses `encode_texts()` if needed, runs UMAP, and shows a Plotly scatter with hover.
- Production-ish: Streamlit page with interactive controls (file selector, color by, UMAP params, sample size) and a cache for reduced coords.

If you want, I can produce a compact runnable script (Jupyter/Streamlit) that implements the minimal pipeline and produces an interactive UMAP scatter from the specified folder. Specify whether you prefer precomputed embeddings or on-the-fly encoding and I will draft the script.

Document created for later implementation.
