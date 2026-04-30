## Embeddings scatter: interpretation guide

This short companion document captures the explanation of what an embeddings scatter plot shows and how to read it. It complements `DESIGN_EMBEDDINGS_VISUALIZATION.md`.

What the scatter plot shows
- Each point represents a transcript window (or manual unit if included).
- Coordinates are a 2D projection (PCA/UMAP/t-SNE) of high-dimensional semantic embeddings; nearby points are semantically similar.
- Clusters correspond to themes or repeated language; manual-unit points inside clusters indicate good coverage.
- Local proximity is most reliable; global distances are approximate and depend on reduction method.

How to interpret visual encodings
- Color by `manual_unit_id`, `topic_id`, or `session_num` to reveal grouping patterns.
- Size by `score` to emphasize confident matches.
- Tooltips should show snippet text, `transcript_id`/`session_num`, `manual_unit_id`, `score`, and `path` to allow quick audits.

Practical analyses enabled
- Coverage: check whether manual units are localized or scattered across the plot.
- Session patterns: do topics concentrate in certain sessions or cycles?
- Outlier detection: spot unusual windows for review (transcription errors, rare events).
- Cluster validation: inspect neighbors to confirm cluster semantics.

Caveats & recommendations
- Dimensionality reduction loses information — use plots for exploration, not exact distance judgments.
- Use the same encoder as retrieval to keep visual distances meaningful relative to matching thresholds.
- For large datasets, use sampling or WebGL/density rendering to avoid overplotting.

Next steps
- Link this doc from `DESIGN_EMBEDDINGS_VISUALIZATION.md` or merge the content there when edits are easier.

Document added as a companion to the main design doc.
