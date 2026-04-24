## Topic × Transcript — Design & Implementation Plan

This document describes a minimal, low-risk design and implementation plan to add a "Topic × Transcript" view to the Streamlit UI (`app/streamlit_app.py`) that matches topic definitions against the entire transcript corpus (no window max) across all cycles.

## Topic × Transcript — Design & Implementation Plan (updated)

This document captures the implemented decisions and the remaining work to support a Topic × Transcript view that can be explored interactively (fast, prebuilt top-k) and later reproduced as exact, full‑corpus aggregates (no top-k).

### Status summary
- Topic × Transcript UI tab added (in `app/streamlit_topk_window_manual_heatmap.py`).
- Prebuilt discovery: the UI discovers files in `data/derived/topic_window_heatmaps/` and allows multi-select.
- Transcript identifier derivation implemented (falls back to `cycle_id` + `session_num`, then `path` or `window_id`).
- Aggregation: UI supports aggregation modes (Max, Mean, Median, Count) computed over windows grouped by `topic_id` × `transcript_id`.
- Min-score slider: global `min_score` from the sidebar is applied as a pre-aggregation filter.
- Topic definitions and `topic_definition` are included in the combined table and CSV export.

### Checklist (requirements)
- [x] Add a Topic × Transcript tab and prebuilt discovery UI (implemented).
- [x] Derive `transcript_id` from match rows if not present (implemented).
- [x] Add min-score slider and apply it before aggregation (implemented; uses existing sidebar slider `min_score`).
- [x] Add aggregation selector (Max/Mean/Median/Count) and pivot to topics × transcripts (implemented).
- [x] Include topic labels and definitions in the displayed table and CSV (implemented).
- [ ] Run the full prebuild for all topics with topk=20 across the corpus (recommended next step).
- [ ] Produce a true, no-topk per-topic×transcript aggregate (offline build) when you choose a threshold to lock in.

### Data contract (current prebuilt flow)
- Input rows (prebuilt parquet): must include at least: topic_id, score, window_id, center_doc_index, cycle_id, session_num, path, window_text. The UI derives `transcript_id` when needed.
- Aggregation input: rows are filtered with `score >= min_score` (slider) before grouping.
- Aggregation output: table with index=topic_id (or topic_label) and columns=transcript_id; cell values are the selected aggregation (max/mean/median/count).
- CSV output: the pivoted matrix exported as CSV with `topic_label` / `topic_definition` included.

### Implementation notes & where code changed
- UI changes live in: `app/streamlit_topk_window_manual_heatmap.py`.
  - Multi-select prebuilt load -> concatenation of selected parquet(s).
  - `transcript_id` derivation function that prefers `cycle_id` + `session_num`, then `path`, then `window_id`.
  - Aggregation selector and pivot code that groups by (`topic_id`, `transcript_id`) and computes the requested statistic.
  - Heatmap rendering via Plotly `imshow` and a CSV download button for the aggregated matrix.
  - Topic labels/definitions are read from `TOPIC_CATALOG_CSV` when available and injected into the display frame.

### UX behavior and caveats
- By default the Topic×Transcript matrix in the UI is computed from selected prebuilt files (top-k). That is fast, but aggregates reflect only saved top-k windows per topic and may miss global maxima.
- The `min_score` slider filters rows before aggregation; for Count mode it functions as the count threshold.
- Large matrices (many topics × many transcripts) may be slow to render; the UI shows a compact heatmap and provides CSV download. Consider restricting selections to a subset when exploring.

### Recommended workflow (what you asked for)
1. Prebuild: run topic×transcript prebuilds across all topics with topk=20 (no `max_windows` cap). This creates per-topic files under `data/derived/topic_window_heatmaps/` you can explore interactively. Example command:

```bash
PYTHONPATH=. ./.venv/bin/python3 scripts/build_topic_window_heatmaps.py --all --topk 20 --outdir data/derived/topic_window_heatmaps --max-windows 0
```

2. Explore in the UI: open `app/streamlit_topk_window_manual_heatmap.py`, select the prebuilt files (one or many), use the `min_score` slider and `Aggregation` selector (Max/Mean/Median/Count) to find a sensible threshold and aggregation behavior. Download CSVs as needed.

3. True aggregate (no top-k): once you pick a min threshold you want to lock in, run an offline true-aggregate build that computes per-topic×transcript statistics across all windows (no topk). Two approaches:
   - Compact true-aggregate: compute and persist a compact parquet with (topic_id, transcript_id, max_score, mean_score, median_score, n_windows). UI can load this file instantly.
   - Embedding-backed counts: persist window embeddings and topic embeddings; at UI time compute exact counts > threshold from vectorized dot-products (flexible but requires embeddings storage and some compute).

Suggested command to produce compact true-aggregates (script to add/extend):

```bash
PYTHONPATH=. ./.venv/bin/python3 scripts/build_topic_window_heatmaps.py --compute-compact-aggregates --outdir data/derived/topic_window_heatmaps
```

### Next steps I can run for you
- Quick (recommended now): run `python -m py_compile app/streamlit_topk_window_manual_heatmap.py` and a smoke load of one sample prebuilt to verify aggregation/pivot code.
- Medium: run the prebuild for all topics with topk=20 (the command above). This can take time but you already ran similar builds earlier.
- Larger: implement the true-aggregate build mode (compute+persist compact aggregates and/or embeddings) and wire the UI toggle to load those compact files when present.

If you want, I will run the compile + smoke test now and then (optionally) kick off the full prebuild of topk=20 for all topics. Tell me which to run.

---

Document updated: reflects implemented UI features and the two‑phase workflow you described (explore with prebuilt top‑k; then run exact full‑corpus aggregates once a threshold is chosen).
