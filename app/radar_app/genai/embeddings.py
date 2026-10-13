"""Embedding untuk dense retrieval.

Model: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensi, mendukung bahasa Indonesia).

Catatan penting: model ini **memotong input di 128 token** (`max_seq_length = 128`). Chunk RAG berukuran hingga ±600
token, jadi kalau satu chunk di-embed utuh, hanya ±128 token pertamanya yang "dibaca". Solusinya: setiap chunk dipecah
menjadi beberapa **jendela kalimat ≤ 110 token**, setiap jendela di-embed, dan skor chunk = skor jendela terbaiknya
(multi-vector / max-sim). Chunk utuh tetap yang dikirim ke LLM sebagai konteks.
"""
from __future__ import annotations

import re

import numpy as np

from .knowledge_base import count_tokens


def sentence_windows(text: str, max_tokens: int = 110) -> list[str]:
    """Pecah teks menjadi jendela berisi kalimat utuh, masing-masing ≤ max_tokens (kalimat panjang dipotong per kata)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?:])\s+|\n+", text) if s.strip()]
    windows, current = [], []
    for sentence in sentences:
        while count_tokens(sentence) > max_tokens:
            words = sentence.split()
            cut = max(1, int(len(words) * max_tokens / max(count_tokens(sentence), 1)))
            windows.append(" ".join(words[:cut]))
            sentence = " ".join(words[cut:])
        if current and count_tokens(" ".join(current + [sentence])) > max_tokens:
            windows.append(" ".join(current))
            current = current[-1:] if count_tokens(current[-1] + " " + sentence) <= max_tokens else []
        current.append(sentence)
    if current:
        windows.append(" ".join(current))
    return windows


class SentenceTransformerEmbedder:
    """Embedding asli dengan sentence-transformers."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=32,
                                            show_progress_bar=False), dtype="float32")


class LsaEmbedder:
    """Pengganti untuk uji alur saja: TF-IDF (kata + karakter) → SVD → vektor ternormalisasi."""

    def __init__(self, corpus: list[str], dimension: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True))])
        matrix = self.vectorizer.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=max(2, min(dimension, matrix.shape[0] - 1, matrix.shape[1] - 1)),
                                random_state=42).fit(matrix)
        self.name = f"LSA TF-IDF+SVD ({self.svd.n_components} dimensi, pengganti)"
        self.dimension = self.svd.n_components
        self.max_seq_length = None

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.svd.transform(self.vectorizer.transform(list(texts)))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, 1e-12, None)).astype("float32")


def make_embedder(model_name: str, fallback_corpus: list[str]):
    """Coba model asli; kalau sentence-transformers / model tidak tersedia, pakai LSA."""
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as error:
        print(f"[embeddings] {model_name} tidak tersedia ({error.__class__.__name__}); memakai LSA pengganti.")
        return LsaEmbedder(fallback_corpus)
