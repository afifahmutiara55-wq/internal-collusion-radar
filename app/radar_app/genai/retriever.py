"""Hybrid retrieval: dense (FAISS) + sparse (BM25), digabung dengan Reciprocal Rank Fusion, hapus duplikat, top 4.

- Dense  : embedding MiniLM per jendela kalimat → FAISS IndexFlatIP (inner product = cosine karena vektor ternormalisasi).
           IndexFlat = pencarian eksak; untuk knowledge base puluhan–ratusan chunk, ini paling akurat dan tetap cepat.
- Sparse : BM25 (Robertson & Zaragoza, 2009) atas teks chunk; bagus untuk istilah persis (nomor pasal, "POJK 12").
- Fusion : RRF (Cormack dkk., 2009): skor = Σ 1 / (60 + peringkat). Tidak perlu menyamakan skala skor dense vs BM25,
           dan chunk yang muncul di kedua daftar otomatis menjadi satu (deduplikasi) dengan skor lebih tinggi.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .embeddings import LsaEmbedder, make_embedder, sentence_windows
from .knowledge_base import load_chunks, save_chunks

STOPWORDS = set("""yang dan di ke dari untuk pada dengan dalam atau ini itu adalah sebagai oleh juga akan tidak bisa
dapat ada apa siapa berapa bagaimana kapan mengapa kenapa saja serta para tersebut karena jika bila maka agar supaya
the of and to in is are a an for on by with be as at it this that""".split())
PREFIXES = ("meng", "meny", "mem", "men", "me", "peng", "peny", "pem", "pen", "pe", "ber", "ter", "di", "ke", "se")
SUFFIXES = ("nya", "kan", "an", "i")


def light_stem(word: str) -> str:
    """Stemming ringan bahasa Indonesia (hanya untuk kata > 6 huruf) supaya 'persetujuan' ≈ 'setuju'."""
    if len(word) <= 6 or word.isdigit():
        return word
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 5:
            word = word[: -len(suffix)]
            break
    for prefix in PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 4:
            return word[len(prefix):]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", text.lower())
    return [light_stem(w) for w in words if w not in STOPWORDS]


class SimpleBM25:
    """BM25Okapi (k1 = 1,5; b = 0,75) — rumus yang sama dengan paket rank_bm25."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.documents = [Counter(tokens) for tokens in corpus_tokens]
        self.lengths = np.array([len(tokens) for tokens in corpus_tokens], dtype=float)
        self.average_length = self.lengths.mean() if len(self.lengths) else 0.0
        document_frequency = Counter(token for tokens in corpus_tokens for token in set(tokens))
        n = len(corpus_tokens)
        self.idf = {token: math.log((n - df + 0.5) / (df + 0.5) + 1) for token, df in document_frequency.items()}

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.documents))
        for i, counts in enumerate(self.documents):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.average_length, 1e-9))
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf:
                    scores[i] += self.idf.get(token, 0.0) * tf * (self.k1 + 1) / (tf + norm)
        return scores


def make_bm25(corpus_tokens):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(corpus_tokens), "rank_bm25.BM25Okapi"
    except Exception:
        return SimpleBM25(corpus_tokens), "SimpleBM25 (pengganti, rumus sama)"


class HybridRetriever:
    def __init__(self, chunks: list[dict], embedder, window_texts: list[str], window_chunk_index: list[int],
                 window_vectors: np.ndarray):
        self.chunks, self.embedder = chunks, embedder
        self.window_texts, self.window_chunk_index = window_texts, np.asarray(window_chunk_index)
        self.window_vectors = window_vectors.astype("float32")
        self.bm25, self.bm25_name = make_bm25([tokenize(c["text"]) for c in chunks])
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.window_vectors.shape[1])
            self.index.add(self.window_vectors)
            self.dense_backend = "FAISS IndexFlatIP"
        except Exception:
            self.index = None
            self.dense_backend = "numpy inner product (pengganti FAISS, hasil identik)"

    # ------------------------------------------------------------------ build / save / load
    @classmethod
    def build(cls, chunks: list[dict], model_name: str, window_tokens: int = 110) -> "HybridRetriever":
        window_texts, window_chunk_index = [], []
        for chunk_number, chunk in enumerate(chunks):
            for window in sentence_windows(chunk["text"], window_tokens):
                window_texts.append(window)
                window_chunk_index.append(chunk_number)
        embedder = make_embedder(model_name, window_texts)
        return cls(chunks, embedder, window_texts, window_chunk_index, embedder.encode(window_texts))

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, folder / "chunks.jsonl")
        np.save(folder / "window_vectors.npy", self.window_vectors)
        (folder / "windows.json").write_text(json.dumps({"texts": self.window_texts,
                                                          "chunk_index": self.window_chunk_index.tolist(),
                                                          "embedder": self.embedder.name}, ensure_ascii=False))
        if isinstance(self.embedder, LsaEmbedder):
            import joblib
            joblib.dump(self.embedder, folder / "lsa_embedder.joblib")

    @classmethod
    def load(cls, folder: Path, model_name: str) -> "HybridRetriever":
        folder = Path(folder)
        meta = json.loads((folder / "windows.json").read_text())
        real_embedder_available = importlib.util.find_spec("sentence_transformers") is not None
        if "LSA" in meta["embedder"] and real_embedder_available:      # indeks lama dibuat dengan pengganti → bangun ulang
            return cls.build(load_chunks(folder / "chunks.jsonl"), model_name)
        if (folder / "lsa_embedder.joblib").exists() and "LSA" in meta["embedder"]:
            import joblib
            embedder = joblib.load(folder / "lsa_embedder.joblib")
        else:
            embedder = make_embedder(model_name, meta["texts"])
            if isinstance(embedder, LsaEmbedder):          # indeks dibuat dengan MiniLM tapi model tidak ada → bangun ulang
                return cls.build(load_chunks(folder / "chunks.jsonl"), model_name)
        return cls(load_chunks(folder / "chunks.jsonl"), embedder, meta["texts"], meta["chunk_index"],
                   np.load(folder / "window_vectors.npy"))

    # ------------------------------------------------------------------ search
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """k chunk teratas menurut cosine; skor chunk = jendela terbaiknya (max-sim)."""
        query_vector = self.embedder.encode([query])
        n_windows = min(len(self.window_texts), max(k * 6, 30))
        if self.index is not None:
            scores, window_ids = self.index.search(query_vector, n_windows)
            scores, window_ids = scores[0], window_ids[0]
        else:
            all_scores = self.window_vectors @ query_vector[0]
            window_ids = np.argsort(-all_scores)[:n_windows]
            scores = all_scores[window_ids]
        best = {}
        for window_id, score in zip(window_ids, scores):
            chunk_number = int(self.window_chunk_index[window_id])
            best[chunk_number] = max(best.get(chunk_number, -1.0), float(score))
        return sorted(best.items(), key=lambda item: -item[1])[:k]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def hybrid_search(self, query: str, dense_k: int = 8, sparse_k: int = 8, top_n: int = 4, rrf_k: int = 60) -> list[dict]:
        dense, sparse = self.dense_search(query, dense_k), self.sparse_search(query, sparse_k)
        fused: dict[int, dict] = {}
        for rank, (chunk_number, score) in enumerate(dense, start=1):
            fused.setdefault(chunk_number, {"rrf": 0.0})
            fused[chunk_number].update(rrf=fused[chunk_number]["rrf"] + 1 / (rrf_k + rank), dense_rank=rank, dense_score=score)
        for rank, (chunk_number, score) in enumerate(sparse, start=1):
            fused.setdefault(chunk_number, {"rrf": 0.0})
            fused[chunk_number].update(rrf=fused[chunk_number]["rrf"] + 1 / (rrf_k + rank), sparse_rank=rank, bm25_score=score)
        ranked = sorted(fused.items(), key=lambda item: -item[1]["rrf"])[:top_n]
        return [{**self.chunks[chunk_number], "rrf_score": info["rrf"], "dense_rank": info.get("dense_rank"),
                 "sparse_rank": info.get("sparse_rank"), "dense_score": info.get("dense_score"),
                 "bm25_score": info.get("bm25_score")} for chunk_number, info in ranked]

    def describe(self) -> dict:
        return {"chunks": len(self.chunks), "windows": len(self.window_texts), "embedder": self.embedder.name,
                "dimension": int(self.window_vectors.shape[1]), "dense": self.dense_backend, "sparse": self.bm25_name}


def load_or_build_retriever(index_dir: Path, kb_dir: Path, model_name: str, chunk_kwargs: dict | None = None):
    """Muat indeks dari disk; kalau belum ada, bangun dari knowledge base (chunk_documents) lalu simpan."""
    from .knowledge_base import chunk_documents, load_documents
    index_dir = Path(index_dir)
    if (index_dir / "windows.json").exists():
        try:
            return HybridRetriever.load(index_dir, model_name)
        except Exception as error:
            print(f"[retriever] indeks tidak bisa dimuat ({error}); membangun ulang.")
    chunks, _ = chunk_documents(load_documents(kb_dir), **(chunk_kwargs or {}))
    retriever = HybridRetriever.build(chunks, model_name)
    retriever.save(index_dir)
    return retriever
