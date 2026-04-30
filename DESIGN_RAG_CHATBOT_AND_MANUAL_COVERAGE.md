# Design: RAG Chatbot Demo & Manual Coverage Analyses

Purpose
- Capture a concise design for two related deliverables:
  1. Paper-ready manual-coverage analyses using existing cycle/session CSVs and manual units.
  2. A demo and evaluation plan to compare a retrieval-augmented-generation (RAG) chatbot against a non-RAG baseline using transcripts + manual units.

Status
- Draft for review. No code changes are made by this document.

## High-level goals
- Produce reproducible, auditable analyses that quantify how well manual units are covered by transcript evidence across cycles and sessions.
- Build a lightweight interactive demo that shows how retrieval over transcripts (and manual units) improves grounding and reduces hallucination vs a pure LLM baseline.

## Data sources (in this repo)
- Canonical manual units: `data/derived/manual_units.csv` (built by `scripts/export_manual_units.py` / `build_manual_unit_index()`)
- Per-cycle evidence (generator outputs):
  - `data/derived/cycle_analysis/<cycle>/session_manual_similarity_evidence.csv` (per candidate match rows)
  - `data/derived/cycle_analysis/<cycle>/manual_unit_coverage_summary.csv` (per manual unit coverage summary)
  - `data/derived/cycle_analysis/<cycle>/manual_unit_counts.json` (cached canonical counts per session)
- Transcript windows and index helpers: `rag_audio_analysis/source_bridge.py` (`get_rag_index_rows()`, `expand_transcript_context()`)

## Part A — Manual coverage analyses (paper-ready)

Objectives
- Describe how many manual units are matched by transcript windows and how confident we are about those matches.
- Show per-session and per-cycle patterns and quantify uncertainty.

Core analyses
1. Descriptive statistics
   - Per-cycle and per-session totals: manual unit counts, matched counts, coverage_rate.
   - Distributions of `max_similarity` and `mapped_manual_unit_match_score`.
   - Top-k coverage (top1/top5 fractions).
2. Calibration & precision analysis
   - Stratified sampling by similarity score bins; human label each sampled evidence row as true/false match.
   - Compute empirical precision per bin and overall (Wilson 95% CI).
   - Produce calibration plot: score bin vs empirical precision.
3. Coverage visualization
   - Heatmap: cycles × sessions showing `coverage_rate`.
   - Line charts: per-session coverage across cycles.
   - Case-study panels: manually inspect 2–3 unit-level examples (manual text, top windows, acceptance decision).
4. Statistical testing & modeling
   - Test differences across cycles/sessions using two-proportion tests or Fisher exact when counts are small.
   - Fit logistic regression / GLMM (match ~ cycle + covariates + (1|session)) to estimate cycle effects and report odds ratios with CIs.
5. Reporting
   - Table: cycle-level summaries (n_units, n_matched, coverage_rate, mean_max_similarity).
   - Appendix: labeling protocol, sample sizes, inter-annotator agreement.

Suggested minimal labeling protocol (for calibration)
- Stratify by similarity bins (e.g., [0.28-0.45), [0.45-0.55), [0.55-0.65), [0.65+]) and by session coverage (low/high), sample ~30–100 rows per stratum.
- Labels: TrueMatch / FalseMatch + error reason (topic mismatch, generic content, excerpt too short, wrong session).

Quality gates (for claims in paper)
- Report precision with 95% Wilson CIs.
- Use conservative thresholds (choose min_similarity so lower CI for precision ≥ target, e.g., 0.7).

## Part B — RAG vs non‑RAG chatbot demo and evaluation

Objectives
- Build a demo where users can ask questions of transcripts (optionally restricted by session) and see grounded answers (RAG) vs LLM-only answers (non‑RAG).
- Evaluate empirically whether RAG yields more grounded answers and fewer hallucinations.

Minimal architecture (prototype)
- Frontend: Streamlit chat UI that supports session/cycle selection, a chat window, and per-answer evidence panes.
- Retriever: reuse `get_rag_index_rows()` + `expand_transcript_context()` to build transcript-window docs; embed with `get_embedding_model()`/`encode_texts()` and persist index per-cycle.
- RAG answerer: assemble a prompt with top-K retrieved snippets (labelled with doc index and metadata) and pass to the LLM with a strict grounding instruction.
- Non-RAG baseline: same LLM and prompt engineering but without retrieved snippets (optionally include brief session summary/manual unit headings).

Prompt policy (for RAG)
- System: enforce "Answer only using the provided snippets; cite each factual claim with bracketed source ids. If no evidence, say 'I don't know'."
- Evidence included inline as short labelled excerpts (limit tokens; provide full windows only on demand).

Follow-up support
- Maintain last N turns of chat in the prompt and a cached set of previously cited document ids to bias retrieval.
- Provide controls: restrict-to-session, top_k, min_similarity.

Evaluation plan
1. Build evaluation set (n=30–100 queries)
   - Mix of evidence-seeking questions, fact lookup, and follow-ups.
   - Include unanswerable items to measure hallucination.
2. Gold annotations
   - For each query collect gold supporting doc indices and a human-judged correct answer label.
3. Run both systems (RAG & non-RAG) on the eval set; collect:
   - Correctness (human binary), evidence precision (fraction of cited snippets that actually support), hallucination rate, rater preference, latency.
4. Statistical tests
   - Paired tests: McNemar for correctness, paired bootstrap for differences in evidence precision, sign test or Wilcoxon for preference.

UX to highlight grounding
- Side-by-side answer comparison view (blind A/B) with explicit citation lists for RAG answers.
- Per-answer evidence inspector: show full windows and manual unit text.

Follow-up / multi-turn conversational behavior
- The demo supports multi-turn chat: the UI retains recent turns (configurable N) and sends them with each new query so the retriever and LLM can use context. Follow-ups are handled by re-running retrieval with the current utterance plus a compact representation of prior turns (either verbatim short turns or a generated summary). The prompt instructs the model to cite evidence ids (E1, E2) for factual claims and to prefer "I don't know" when no supporting evidence is present.
- Evidence continuity: the system preserves previously cited evidence ids in the conversation state and can bias retrieval to re-surface or avoid repeating them. Assistant responses include the cited evidence ids so the UI can expand the exact windows on-demand.
- Token-budget strategy: keep only the last 1-3 turns verbatim in the prompt and include earlier history as short summaries when needed. Retrieved snippets are limited (top_k, excerpt length) to stay within token limits.
- Clarification behavior: when context is ambiguous or insufficient, the prompt favors asking a clarifying question rather than hallucinating an answer.

## Evaluation metrics and visuals for demo
- Table: RAG vs non-RAG — accuracy, evidence precision, hallucination rate, mean latency.
- Bar chart with 95% CIs of accuracy/evidence precision difference.
- Example transcripts that show clear grounding benefits (positive/negative examples).

## Implementation notes and reuse
- Reuse helper functions in `rag_audio_analysis/source_bridge.py`:
  - `get_embedding_model()`, `encode_texts()`, `get_rag_index_rows()`, `expand_transcript_context()`, `build_manual_unit_index()`.
- Reuse CSVs for audit and to seed the eval set.
- Cache embeddings and index per-cycle to avoid repeated model loads.

## Deliverables (pickable)
- Scripts for: sampling and exporting evaluation examples, computing coverage & calibration plots, running RAG vs non-RAG baseline comparisons.
- Streamlit demo skeleton that implements the chat UI + retrieval pipeline (no external infra required).
- Evaluation harness to run both systems and produce tables/plots and significance tests.

## Timeline (rough estimate)
- Fast prototype (one day): index transcript windows, simple Streamlit UI, single-turn RAG answers.
- Labeling & calibration (1–3 days): produce and label stratified sample for precision estimates.
- Demo & eval (2–4 days): implement non-RAG baseline, collect human ratings, compute metrics and plots.

## Next steps
- Choose the next deliverable I should prepare (no changes are made until you ask):
  - (A) Sampling + labeling export script for calibration.
  - (B) Coverage analysis scripts + plotting templates.
  - (C) Streamlit skeleton for RAG vs non-RAG demo.
  - (D) Evaluation harness to run both systems and compute stats.


Appendix: quick pointers to repo locations
- Manual units builder: `rag_audio_analysis/source_bridge.py::get_structured_manual_units()` and `build_manual_unit_index()`.
- Evidence generator: `scripts/generate_cycle_similarity.py`.
- Viewer/UI: `app/view_similarity.py` (current evidence inspector) and `scripts/precompute_manual_unit_counts.py` (counts cache).


---

*Document created for review. Reply with which deliverable to prepare and I will implement the scripts or skeleton next.*


## Messaging-style chat UI (design addendum)

Goal
- Make the Streamlit demo behave like a messaging app: linear chat history, assistant replies as single message bubbles, and retrieved evidence attached to assistant messages as expandable attachments. Preserve prompt and retrieval plumbing for audit but hide them by default.

Message contract
- Each turn is stored as a dict with these keys:
   - role: "user" | "assistant"
   - text: the message text
   - time: ISO timestamp
   - retrieval: optional list of evidence items: { source: "window"|"manual", id, score, path|manual_unit_id, excerpt }
   - prompt: optional prompt text sent to the model (hidden by default)
   - model_raw: optional raw model output (for debugging)

UI composition
- Replace the current multi-column answer panes with a single chat stream:
   - Use `st.chat_message(role)` (if Streamlit supports it) or a repeating `st.container()` per message.
   - Render user bubbles to the right and assistant bubbles to the left.
   - Assistant bubble footer: show brief provenance ("RAG · N evidence · best_score") and a small "Show evidence" expander.
   - The evidence expander lists retrieved items with source badge (MAN/WIN), score, snippet, and buttons to view full window or copy the evidence id.
   - Add a per-message "Show prompt" toggle to reveal the exact prompt for auditing.

Retrieval & conversational context
- Context window: include the last N turns (default N=2 user turns or last 512 tokens) when constructing the retrieval query or prompt.
- Retrieval flow per turn:
   1. Compose a short retrieval query from the current user message, optionally prefixed by a compact representation of recent turns.
   2. Call the retriever (windows + manuals). Merge manuals into evidence rows (existing approach) and return top-K.
   3. Build the model prompt from system instruction + recent turns + current user message + evidence_rows and call the model.
   4. Append assistant message to history including retrieval list and prompt text for audit.

UX affordances
- Collapse evidence by default; show counts and best-score in the message footer.
- Allow pinning an evidence item to the next prompt (user action) so it becomes forced context.
- Provide a "Regenerate" button on each assistant message to re-run the same turn (optionally with re-run retrieval).
- Offer a compact summary view for long histories (summarize older turns) to control token budget.

Edge cases and mitigations
- Long prompt / token limits: keep only recent turns verbatim, summarize older history, and limit evidence excerpts.
- Duplicate evidence across turns: dedupe by normalized text hash or exact match when rendering attachments.
- Provenance clarity: always display evidence source (manual vs window), cycle/session, and a clickable id that opens the full transcript window.
- Latency: show an animated spinner during retrieval and model calls; optionally stream partial results.

Testing checklist
- Happy path: single-turn QA returns assistant bubble with evidence expander listing manuals and windows.
- Context use: follow-up question uses prior turns and yields consistent answers.
- Audit features: toggle prompt visibility and confirm prompt contains the same evidence items listed in the message attachments.
- Regenerate: clicking regenerate produces a new assistant message and does not break history order.

Rollout approach
- Implement UI wiring first (history store, chat rendering, evidence attachments) using existing retrieval API.
- Enable the audit toggle to reveal prompts and retrieved evidence for early testing.
- Keep previous demo view accessible behind a feature flag until validation is complete.


## Recent implementation delta (what's already been applied in the repo)

Summary
- The repository now contains a working messaging-style Streamlit demo and several backend changes to improve manual-unit usage and multi-turn history handling. These edits are conservative and preserve the legacy two-column RAG vs non-RAG view (feature-flagged by the Messaging-style UI toggle).

Files changed (high level)
- `app/rag_demo.py`
   - Messaging-style UI added (sidebar toggle). Single-line composer + send button with a safe on-click callback to avoid Streamlit session_state widget mutation errors.
   - Composer now appends history entries in the role/text contract: `{"role":"user","text":...}` and appends an assistant entry `{"role":"assistant","text":...,"rag": ...}` after a RAG response.
   - Evidence rendering updated to show manual-unit session/transcript identifiers when available (prefers `session_num`, then `transcript_id`, then `manual_session`).
   - "All cycles" option added to the Cycle selectbox (maps to empty cycle param to allow global retrieval across cycles).
   - Messaging UI intentionally omits non-RAG rendering; the legacy two-column view still shows non-RAG for side-by-side comparison.

- `rag_audio_analysis/rag_service.py`
   - `retrieve_for_question()` unchanged in API but `answer_rag()` now merges manual hits with transcript window hits into `evidence_rows` so the default prompt sees manual text.
   - Added a standardized `session_label` in the merged evidence items (prefers fields present on windows/manuals). `evidence_rows` includes `session_label` for downstream consumers.
   - `answer_rag()` now accepts `history` (list of turns) and `history_turns` and prepends recent turns to the question so prompts are history-aware.

Behavioral notes
- History contract: the UI and `answer_rag()` now agree on a minimal turn shape (`role` + `text`). The UI appends assistant turns containing the full RAG payload under a `rag` key so the renderer can display evidence and prompts. This resolves a bug where previously the UI used `{"user": ...}` and `answer_rag()` ignored history.
- Manual evidence: manual hits are merged into evidence rows and surfaced to the prompt. UI lines display a session label when available. The backend also exposes `session_label` in `evidence_rows` so other tools can rely on it.
- Defaults and controls: the demo uses the manual-unit similarity threshold from `settings.ini` by default so manual evidence is shown unless the user adjusts the slider.

Short verification steps
1. Start the Streamlit demo and enable the Messaging-style UI in the sidebar.
    - streamlit run app/rag_demo.py
2. Send a short question. Confirm the UI appends a user bubble and then an assistant bubble with an Evidence expander.
3. Inspect the Evidence expander — manual hits should show a session label when available and appear alongside transcript windows.
4. Send a follow-up question that depends on the previous turn. The model prompt (expandable via "Show prompt") should include the recent turns you sent.

Known gaps / recommended next steps
- Composer placement: the message composer is currently in the main content area; moving it to a sticky/bottom position with scroll-to-bottom behavior remains to be implemented.
- Session label normalization: the current `session_label` uses fields present on items. Consider normalizing into a canonical `PMHCycleX_sessionY` format (derive from `path` when missing) for consistency across data sources.
- Tests: add an automated test that asserts `answer_rag()` includes history-prefixed questions when given a sample history list.
- Audit UX: expose a per-message "Regenerate" and "Pin evidence" affordance to reproduce and lock context.

If you want, I can now implement one of the recommended next steps (normalize `session_label`, move composer to sticky bottom, or add a unit test for history handling). Please tell me which to prioritize.

