## File-specific guidance: application-of-mindful-eating-skills parquet

File inspected:
`data/derived/topic_window_heatmaps/topic_only_application-of-mindful-eating-skills-for-choices-of-healthy-fast-foods-and-portion-sizes_full.parquet`

Recommended columns and plotting parameters

- Point sources:
  - Transcript windows (one point per window): `window_text` (compute embeddings from this)
  - Manual-unit points (optional): `manual_unit_matching_text` or `manual_unit_text` (collapse rows by `manual_unit_id` for one point per manual)

- Metadata for tooltips / grouping:
  - `window_id`, `manual_unit_id`, `score`, `session_num`, `cycle_id`, `path`, `center_doc_index`, `included_doc_indices`

- Embeddings:
  - This file has no precomputed embeddings. Use `encode_texts()` from `rag_audio_analysis/source_bridge.py` to compute embeddings for `window_text` and `manual_unit_matching_text`.
  - L2-normalize embeddings if you will use cosine-based UMAP.

- Small-file parameters (20 rows):
  - PCA(n_components=2) for quick plotting.
  - Plot windows as small grey points and manual units as larger colored markers.
  - Hover: first ~250 chars of `window_text`, `manual_unit_id`, and `score`.

- Scaling to larger datasets:
  - Pre-reduce with PCA to 50 dims, then UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').
  - Use Plotly `scattergl` or datashader for large point counts.

- Quick analysis ideas:
  - Centroid per manual: collapse manual rows and compute centroids; visualize centroid proximity to window points.
  - Coverage metric: nearest-neighbor distance from manual centroid to windows; rank manuals by min distance to detect low coverage.

This file is a compact reference for plotting this specific parquet dataset.
