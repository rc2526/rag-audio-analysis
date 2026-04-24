Design: Transcript × Transcript similarity

Goal
- Provide a dense transcript×transcript similarity heatmap to help discover similar sessions or documents across the corpus.

Approach (default)
- Build transcript-level text by concatenating turns from each transcript file (group by path).
- Use the project's embedding model to encode each transcript into a fixed-size vector using existing helper `encode_texts` from `rag_audio_analysis.source_bridge`.
- Compute pairwise similarity between transcripts:
  - Default metric: Cosine similarity (compute dot product of normalized embedding vectors).
  - Optional: Pearson correlation across embedding dimensions via numpy.corrcoef.

UI
- New tab: "Transcript × Transcript"
- Controls:
  - Similarity metric: Cosine (default) or Pearson
  - Max transcripts to include (default 200, min 20, cap 2000)
  - Clustering toggle: reorder rows/cols by mean similarity (simple)
  - Compute button to run embedding and similarity compute
- Outputs:
  - Plotly heatmap with transcripts on both axes
  - CSV download of the full transcript×transcript similarity matrix
  - Tip text to reduce max transcripts if compute is slow

Performance & Safety
- O(N^2) memory/time for N transcripts — default cap 200 to avoid blow-ups.
- Warn the user when more transcripts are discovered than the cap and limit to the first N unless the cap is increased.

Implementation notes
- Reuse `get_transcript_turns()` to read and group transcript texts.
- Reuse `encode_texts()` to get embeddings (handles model loading and normalization).
- Cache embeddings or the sim DataFrame in `st.session_state` if desired later.

Alternatives (future)
- Use aggregated manual-unit vectors (if available) as a different similarity space.
- Build transcript embeddings from weighted window-level embeddings (weight by score) for more nuanced similarities.

"""
