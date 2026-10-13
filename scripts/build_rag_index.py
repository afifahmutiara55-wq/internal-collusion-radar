"""Bangun ulang indeks RAG dari knowledge_base/*.md (chunking → embedding MiniLM → FAISS + BM25).

Pemakaian: python scripts/build_rag_index.py [--force]
Tanpa --force, indeks hanya dibangun kalau belum ada atau masih memakai embedding pengganti (LSA).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from radar_app.config import SETTINGS  # noqa: E402
from radar_app.genai.knowledge_base import chunk_documents, load_documents, save_chunks, tokenizer_name  # noqa: E402
from radar_app.genai.retriever import HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    meta_path = SETTINGS.rag_index_dir / "windows.json"
    if meta_path.exists() and not args.force:
        embedder = json.loads(meta_path.read_text()).get("embedder", "")
        try:
            import sentence_transformers  # noqa: F401
            real_available = True
        except ImportError:
            real_available = False
        if "LSA" not in embedder or not real_available:
            print(f"Indeks sudah ada ({embedder}); lewati. Gunakan --force untuk membangun ulang.")
            return 0
    documents = load_documents(SETTINGS.knowledge_base_dir)
    chunks, splitter = chunk_documents(documents, SETTINGS.chunk_size_tokens, SETTINGS.chunk_overlap_tokens, SETTINGS.chunk_max_tokens)
    save_chunks(chunks, SETTINGS.knowledge_base_dir / "chunks.jsonl")
    retriever = HybridRetriever.build(chunks, SETTINGS.embedding_model, SETTINGS.embedding_window_tokens)
    retriever.save(SETTINGS.rag_index_dir)
    tokens = sorted(c["n_tokens"] for c in chunks)
    print(f"{len(documents)} dokumen → {len(chunks)} chunk ({splitter}; {tokenizer_name()}), token {tokens[0]}–{tokens[-1]}, "
          f"median {tokens[len(tokens) // 2]}")
    print("Indeks:", retriever.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
