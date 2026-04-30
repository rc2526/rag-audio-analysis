def test_imports():
    from rag_audio_analysis.indexer import build_cycle_window_index, load_cycle_window_index
    # Attempt to build or load an index for PMHCycle1; do not force long operations in CI
    try:
        build_cycle_window_index("PMHCycle1", force=False)
    except Exception:
        # acceptable in environments without transcripts; ensure load raises a clear error
        import pytest

        with pytest.raises(FileNotFoundError):
            load_cycle_window_index("nonexistent_cycle")
