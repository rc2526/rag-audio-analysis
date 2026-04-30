import argparse, json, os, sys, hashlib, tempfile, shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# --- adjust imports to use project modules ---
from rag_audio_analysis.config import CYCLE_ANALYSIS_DIR, DERIVED_DIR
from rag_audio_analysis.indexer import build_cycle_window_index
from rag_audio_analysis.source_bridge import build_manual_unit_index, get_manual_unit_embedding_index, encode_texts

def atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)

def atomic_npy_save(path: Path, arr: np.ndarray):
    tmp = path.with_suffix(path.suffix + ".tmp")
    # use numpy.save to a temp file path
    np.save(str(tmp), arr.astype("float32"))
    os.replace(str(tmp), str(path))

def sha256_of_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def persist_manual_units(out_dir: Path):
    units, emb = get_manual_unit_embedding_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "manual_unit_index_meta.json"
    emb_path = out_dir / "manual_unit_embeddings.npy"
    info_path = out_dir / "manual_unit_index_info.json"

    # ensure embeddings exist / normalize rows
    if emb is None or getattr(emb, "size", 0) == 0:
        # fallback: re-encode matching_text explicitly
        texts = [str(u.get("matching_text") or u.get("text") or "") for u in units]
        emb = encode_texts(texts)

    if getattr(emb, "ndim", 0) == 2 and emb.shape[0] == len(units):
        # normalize rows defensively
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_norm = (emb / norms).astype("float32")
    else:
        emb_norm = np.zeros((len(units), 0), dtype="float32")

    # atomic writes
    atomic_write_bytes(meta_path, json.dumps(units, ensure_ascii=False, indent=2).encode("utf-8"))
    atomic_npy_save(emb_path, emb_norm)

    info = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "unit_count": len(units),
        "embedding_shape": list(emb_norm.shape),
        "dtype": str(emb_norm.dtype),
        "model": None,
    }
    info["embeddings_sha256"] = sha256_of_file(emb_path)
    atomic_write_bytes(info_path, json.dumps(info, indent=2).encode("utf-8"))
    return meta_path, emb_path, info_path

def build_indexes_for_cycles(cycles, window, force):
    built = []
    failed = []
    for c in cycles:
        try:
            emb_path, meta_path = build_cycle_window_index(c, window=window, force=force)
            built.append((c, emb_path, meta_path))
        except Exception as e:
            failed.append((c, str(e)))
    return built, failed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="*", help="Cycle names to process (default: all)")
    parser.add_argument("--window", type=int, default=2, help="Context window size")
    parser.add_argument("--force", action="store_true", help="Force rebuild indexes")
    parser.add_argument("--skip-manual-persist", action="store_true", help="Don't persist manual-unit artifacts")
    args = parser.parse_args()

    base = Path(CYCLE_ANALYSIS_DIR)
    if not base.exists():
        print("CYCLE_ANALYSIS_DIR missing:", base, file=sys.stderr)
        sys.exit(1)

    if args.cycles:
        cycles = args.cycles
    else:
        cycles = [p.name for p in sorted(base.iterdir()) if p.is_dir()]

    print("Cycles to process:", cycles)
    print("Window:", args.window, "Force:", args.force)

    built, failed = build_indexes_for_cycles(cycles, args.window, args.force)
    print("\nIndex build summary:")
    for c, emb, meta in built:
        print("  built:", c, "->", emb, meta)
    for c, err in failed:
        print("  failed:", c, "-", err)

    # persist manual units
    if not args.skip_manual_persist:
        print("\nPersisting manual-unit metadata and embeddings to", Path(DERIVED_DIR))
        try:
            meta_path, emb_path, info_path = persist_manual_units(Path(DERIVED_DIR))
            print("  manual meta:", meta_path)
            print("  manual embeddings:", emb_path)
            print("  manual info:", info_path)
            # quick verification
            import json
            m = json.load(open(meta_path, encoding="utf-8"))
            arr = np.load(str(emb_path))
            ok = (len(m) == arr.shape[0])
            print("  verification: meta rows =", len(m), "emb rows =", arr.shape[0], "match=", ok)
        except Exception as e:
            print("  failed to persist manual units:", e)

    # quick index verification: ensure files exist for first few cycles
    print("\nQuick verification of index files:")
    for c, emb, meta in built[:5]:
        e = Path(emb)
        m = Path(meta)
        print(f"  {c} -> emb exists={e.exists()}, meta exists={m.exists()}")
    print("\nDone.")

if __name__ == '__main__':
    main()
