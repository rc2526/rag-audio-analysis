from pathlib import Path
from rag_audio_analysis.settings import get_str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
DERIVED_DIR = DATA_DIR / "derived"
APP_DIR = PROJECT_ROOT / "app"

SOURCE_ROOT = Path(get_str("paths", "source_root", "/Users/rhea.chatterjeeyale.edu/rag-audio"))
SOURCE_RAG_INDEX = SOURCE_ROOT / get_str("paths", "rag_index_subdir", "rag_index")
SOURCE_BUILD_AND_QUERY = SOURCE_ROOT / get_str("paths", "build_and_query_relpath", "tools/build_and_query_rag.py")
SOURCE_VENV_PYTHON = SOURCE_ROOT / ".venv" / "bin" / "python"
SOURCE_META = SOURCE_ROOT / get_str("paths", "meta_relpath", "rag_index/meta.json")
SOURCE_MANUAL = SOURCE_ROOT / get_str("paths", "manual_relpath", "manual.txt")
SOURCE_TOPICS_CSV = SOURCE_ROOT / get_str("paths", "topics_csv_relpath", "manual_topics.csv")
SOURCE_TOPIC_LIST = SOURCE_ROOT / get_str("paths", "topic_list_relpath", "rag_index/topic_list.json")
SOURCE_MANUAL_DOC_INDICES = SOURCE_ROOT / get_str("paths", "manual_doc_indices_relpath", "rag_index/manual_doc_indices.json")
SOURCE_MANUAL_DOC_TOPIC_MAP = SOURCE_ROOT / get_str("paths", "manual_doc_topic_map_relpath", "rag_index/manual_doc_topic_map.json")
SOURCE_TRANSCRIPTS_GLOB = str(SOURCE_ROOT / get_str("paths", "transcripts_glob", "audio/de-identified_transcripts/**/*.txt"))

TOPIC_CATALOG_CSV = DERIVED_DIR / "topic_catalog.csv"
MANUAL_UNITS_CSV = DERIVED_DIR / "manual_units.csv"
TRANSCRIPT_SPANS_CSV = DERIVED_DIR / "transcript_spans.csv"
CODED_EVIDENCE_CSV = DERIVED_DIR / "coded_evidence.csv"
QUOTE_BANK_CSV = DERIVED_DIR / "quote_bank.csv"
SPEAKER_ROLE_MAP_CSV = DERIVED_DIR / "speaker_role_map.csv"
CONTENT_REVIEW_QUEUE_CSV = DERIVED_DIR / "content_review_queue.csv"
TOPIC_CONTENT_SUMMARY_CSV = DERIVED_DIR / "topic_content_summary.csv"
TOPIC_SESSION_SUMMARY_CSV = DERIVED_DIR / "topic_session_summary.csv"
MANUAL_FIDELITY_SUMMARY_CSV = DERIVED_DIR / "manual_fidelity_summary.csv"
CYCLE_ANALYSIS_DIR = DERIVED_DIR / "cycle_analysis"

for path in (DATA_DIR, TEMPLATES_DIR, DERIVED_DIR, APP_DIR, CYCLE_ANALYSIS_DIR):
    path.mkdir(parents=True, exist_ok=True)
