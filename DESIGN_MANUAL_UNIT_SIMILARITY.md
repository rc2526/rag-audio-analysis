Design: Manual-Unit ↔ Transcript-Window Similarity Retrieval

## Purpose

Describe the design and implementation plan to change session-level fidelity retrieval to rank and filter evidence exclusively by the cosine similarity between manual-unit embeddings and transcript-window embeddings.

This doc documents the helper API, CLI wiring, CSV schema changes, caching and performance notes, tests, and rollout steps for a similarity-first retrieval mode.

## Goals

- Rank transcript windows by manual-unit ↔ window cosine similarity (not by doc/topic retrieval score).
- Return evidence rows only for pairs meeting the configured `topic_matching.manual_unit_min_similarity` threshold (or include a diagnostic mode for near-misses).
- Keep per-unit top-k and optional global caps to bound output size.
- Reuse existing embedding model and encoding helpers; centralize vector logic in `rag_audio_analysis/source_bridge.py`.
- Make results explainable and easy to audit: include query attribution and similarity score in CSV outputs.

## Requirements (checked during implementation)

- Inputs:
  - session_manual_units (list of canonical MAN units for a manual session)
  - cycle_id (e.g., `PMHCycle1`)
  - meta_rows, path_lookup, window (context expansion items)
  - topk_per_unit (int, optional)
  - min_similarity (float, optional; default from settings.ini)
- Outputs:
  - Evidence rows: one row per (manual unit, transcript window) pair that meets acceptance criteria. Each row must include the manual-unit id used as the query, the cosine similarity, and a boolean accepted flag.
- CLI: `--manual-query-mode` is now `similarity` only.

## Where to implement

- Primary implementation: `rag_audio_analysis/source_bridge.py`
  - Add a vectorized helper that:
    - loads/encodes manual-unit texts into matrix M (m×d)
    - gathers/encodes candidate transcript windows for the cycle into matrix W (n×d)
    - computes S = M @ W.T (cosine, since encode_texts() returns normalized rows)
    - selects entries S[i,j] >= min_similarity and returns structured rows sorted by similarity (per unit)
  - Reason: `source_bridge` already owns embedding model, caching, and index-to-path expansion logic.

- Orchestration: `scripts/run_cycle_analysis.py`
  - Call the similarity helper and format CSV rows accordingly.

## Helper API (recommended)

Signature (pseudo-Python):

- get_manual_unit_embeddings(session_manual_units) -> (manual_ids, manual_texts, manual_embs)
- get_cycle_window_embeddings(cycle_id, meta_rows, path_lookup, window) -> (doc_indices, window_texts, window_embs, window_meta)
  - query_evidence_by_manual_unit_similarity(
    session_manual_units: list[dict],
    cycle_id: str,
    meta_rows: list[dict],
    path_lookup: dict,
    window: int,
    topk_per_unit: int = 0,
    min_similarity: float | None = None,
    global_cap: int | None = None,
  ) -> list[dict]

Returned row fields (each row):
- doc_index
- retrieval_rank (rank by similarity for that manual unit)
- session_id
- cycle_id
- speaker
- score_combined (set to similarity for clarity)
- score_doc (empty)
- score_topic (empty)
- text
- manual_unit_id_best_match (the query manual id)
- manual_unit_match_score (cosine)
- query_manual_unit_id
- query_manual_unit_subsection
- query_text
- mapped_manual_unit_match_score
- mapped_manual_unit_accepted (True)

Notes:
- Use `encode_texts()` existing helper which returns normalized rows, so dot product = cosine.

## CSV / Output schema changes

- `session_fidelity_evidence.csv` add columns:
  - query_manual_unit_id
  - query_manual_unit_subsection
  - mapped_manual_unit_match_score
  - mapped_manual_unit_accepted

- Populate `mapped_manual_unit_match_score` with cosine in [0..1] and `mapped_manual_unit_accepted` boolean (or string) set by threshold.
- `manual_unit_id_best_match` should reflect the query unit id in this flow; keep both fields for compatibility.

## Acceptance & coverage rules

- Binary rule (default): a manual unit U is "covered" if there exists at least one evidence row for U with `mapped_manual_unit_match_score >= min_similarity`.
- Optional weighted rule: coverage contribution = max(mapped_manual_unit_match_score across U) (useful to compute soft adherence scores).

## Performance and caching

- Encoding cost is the dominant expense. Strategies:
  - Cache manual-unit embeddings (compute once per session).
  - Cache cycle window embeddings keyed by (cycle_id, doc_index, window_span). Persist to disk between runs for repeated analysis.
  - Batch-encode texts via existing model API to amortize overhead.
  - Use vectorized numpy matmul for computing S = M @ W.T. If n or m is large, compute in blocks (e.g., batch manual units or windows) to bound memory.
- Memory: if W is very large, iterate manual units in batches and compute partial dot products.

## Edge cases

- Paraphrase mismatch: embeddings may under-score valid paraphrases. Provide a diagnostic mode to emit top-k candidates below threshold for inspection.
- Duplicate windows: the same window may match multiple manual units. This is expected and allowed. Summary metrics should count coverage per unit (not unique-window counts) unless otherwise required.
- Empty manual units or no windows: return empty lists gracefully.

## Tests & validation

- Unit test: small synthetic session with 2 manual units and 4 windows; assert the similarity matrix S, accepted indices, and returned rows shape.
- Smoke test: run `python3 scripts/run_cycle_analysis.py --cycles 1 --manual-query-mode similarity --manual-query-topk 6` and inspect outputs in `CYCLE_ANALYSIS_DIR/PMHCycle1/`.
- Manual QA: compare a small set of (unit, window) pairs with expected embeddings using an interactive REPL.

## Rollout plan

1. Implement helper and wire to `run_cycle_analysis.py` to run the similarity-first flow (done).
2. Run smoke test on 1 cycle (small topk) and inspect `session_fidelity_evidence.csv` (mapped scores and accepted flags).
3. Add optional on-disk caching for window embeddings if initial runs are slow.
4. Add diagnostic flag `--include-nearmisses` to emit below-threshold top-N candidates for tuning.
5. Iterate thresholds and optionally add re-ranking hybrids (alpha * retrieval_score + beta * similarity).

## Example commands

- Smoke run (per-unit topk 6):

```bash
python3 scripts/run_cycle_analysis.py --cycles 1 --manual-query-mode similarity --manual-query-topk 6
```

- Run a single session (if supported) and produce only fidelity outputs:

```bash
python3 scripts/run_cycle_analysis.py --cycles 1 --manual-query-mode similarity --manual-query-topk 6 --mode fidelity --session-num 3
```

## Exact headers for new files

When producing the similarity-first outputs we recommend the following exact headers and helper names (already added as constants and wrappers in `scripts/run_cycle_analysis.py`):

- `session_manual_similarity_evidence.csv` header (writer helper: `write_similarity_evidence_csv`):

  - cycle_id
  - manual_session_num
  - query_manual_unit_id
  - query_manual_unit_subsection
  - query_text_excerpt
  - doc_index
  - retrieval_rank
  - global_rank
  - session_id
  - speaker
  - mapped_manual_unit_match_score
  - mapped_manual_unit_accepted
  - text_excerpt
  - text_length
  - doc_path
  - note

- `session_manual_similarity_diagnostics.csv` header (writer helper: `write_similarity_diagnostic_csv`):

  - cycle_id
  - manual_session_num
  - query_manual_unit_id
  - candidate_type
  - doc_index
  - retrieval_rank
  - global_rank
  - session_id
  - speaker
  - mapped_manual_unit_match_score
  - similarity_distance_to_threshold
  - retrieval_score_doc
  - window_embedding_hash
  - text_excerpt
  - include_full_text_flag
  - doc_path
  - note

These functions simply call the existing `write_csv` helper with the canonical headers so future wiring is a one-line change.

## Future improvements

- Persist window embeddings per cycle to disk for repeated analyses.
- Add a hybrid ranking option (linear combination of retrieval and similarity) for higher recall.
- Add a small web UI to browse matched pairs and tune thresholds interactively.

## Recent implementation changes (2026-04-27)

This project made a few small but important changes during implementation and debugging; record them here so future readers understand the current behaviour and where to tune thresholds/top-k.

- Acceptance logic centralized in the helper
  - `query_evidence_by_manual_unit_similarity(...)` now sets `mapped_manual_unit_accepted` based on the computed score vs. the effective `min_similarity` used by the helper (i.e. `mapped_manual_unit_accepted = score >= min_similarity`).
  - Rationale: the helper is the single place that knows the active `min_similarity` (either passed in or read from settings) so acceptance is decided where rows are produced.

- Top-k semantics clarified
  - `topk_per_unit` is applied after thresholding: the helper first filters candidate windows by `sims[i,j] >= min_similarity` then sorts by similarity and finally truncates to `topk_per_unit` if > 0.
  - Effect: `topk_per_unit` reduces number of returned candidates per manual unit but cannot create matches below the threshold; increase `min_similarity` to reduce weak matches.

- Generator utility: non-destructive, per-cycle appender
  - To avoid repeatedly editing the main orchestrator, a small utility `scripts/generate_cycle_similarity.py` was added. It iterates topic-discovered sessions and appends per-session similarity evidence into the cycle CSVs (`session_manual_similarity_evidence.csv` and `manual_unit_coverage_summary.csv`) rather than overwriting.

- Coverage computation and UI caching
  - The viewer `app/view_similarity.py` was added to inspect the evidence CSVs interactively.
  - The session-coverage UI originally computed `total_units` from the coverage CSV (which undercounts units that had zero returned candidates). The UI was changed to compute `total_units` from `get_manual_units_for_session(session)` (the canonical source) and to cache those counts on disk per-cycle in `manual_unit_counts.json` so totals are not recomputed on every page render.
  - A sidebar control was added to force recompute the cached counts when manual units change.

- Default behaviours and fixes applied
  - Earlier runs defaulted `mapped_manual_unit_accepted` to True in some code paths; that was fixed so acceptance respects the configured threshold.
  - The generator/utility uses `topk_per_unit` and `min_similarity` parameters so runs can be reconfigured without changing code.

## Operational notes

- If you change `topic_matching.manual_unit_min_similarity` in `settings.ini`, re-run the generator and (optionally) press "Recompute manual unit counts" in the viewer so cached totals align with any edits you made to manual units.
- To tighten coverage, raise `manual_unit_min_similarity` (e.g. 0.60–0.65) and/or reduce `topk_per_unit` to 1–3 to surface only the strongest matches per manual unit.


---

File created by the implementation task: `DESIGN_MANUAL_UNIT_SIMILARITY.md`.
