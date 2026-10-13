"""Knowledge base RAG: memuat dokumen markdown dan memecahnya menjadi chunk.

Chunking memakai `RecursiveCharacterTextSplitter` (LangChain) dengan panjang diukur dalam TOKEN tokenizer model
embedding. Dokumen dipecah dulu per bagian (judul `##`) supaya satu chunk tidak mencampur dua topik; bagian yang panjang
dipecah rekursif (paragraf → baris → kalimat → kata). Setiap chunk diberi "header konteks" (judul dokumen › bagian),
yang terbukti membantu retrieval karena chunk tetap membawa topiknya.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


# --------------------------------------------------------------------------- tokenizer
@lru_cache(maxsize=1)
def _hf_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None


def count_tokens(text: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> int:
    """Jumlah token menurut tokenizer model embedding; kalau tokenizer tidak tersedia, pakai perkiraan.

    Perkiraan: tokenizer XLM-R (dipakai MiniLM multilingual) memecah kata Indonesia rata-rata ±1,35 sub-kata.
    """
    tokenizer = _hf_tokenizer(model_name)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(round(len(re.findall(r"\w+|[^\w\s]", text)) * 1.35))


def tokenizer_name(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> str:
    return "tokenizer " + model_name.split("/")[-1] if _hf_tokenizer(model_name) is not None else "perkiraan (kata × 1,35)"


# --------------------------------------------------------------------------- dokumen
def parse_document(path: Path) -> dict:
    """Pisahkan front matter (--- key: value ---) dari isi markdown."""
    raw = path.read_text(encoding="utf-8")
    metadata, body = {}, raw
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    metadata.setdefault("doc_id", path.stem)
    metadata.setdefault("title", path.stem)
    return {**metadata, "text": body.strip(), "path": str(path)}


def load_documents(kb_dir: Path) -> list[dict]:
    return [parse_document(path) for path in sorted(Path(kb_dir).glob("*.md"))]


def split_sections(document: dict) -> list[tuple[str, str]]:
    """[(judul bagian, teks)] berdasarkan heading '## '. Judul H1 dibuang (sudah ada di metadata)."""
    sections, current_title, buffer = [], "Pengantar", []
    for line in document["text"].splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            continue
        if line.startswith("## "):
            if "".join(buffer).strip():
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title, buffer = line[3:].strip(), []
        else:
            buffer.append(line)
    if "".join(buffer).strip():
        sections.append((current_title, "\n".join(buffer).strip()))
    return sections


# --------------------------------------------------------------------------- splitter
class SimpleRecursiveSplitter:
    """Pengganti RecursiveCharacterTextSplitter dengan algoritma yang sama (dipakai kalau LangChain tidak terpasang)."""

    def __init__(self, chunk_size: int, chunk_overlap: int, length_function, separators=SEPARATORS):
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.length, self.separators = length_function, separators

    def split_text(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator = next((s for s in separators if s == "" or s in text), "")
        remaining = separators[separators.index(separator) + 1:] if separator in separators else []
        pieces = list(text) if separator == "" else [p for p in text.split(separator) if p.strip()]
        chunks, current = [], []
        for piece in pieces:
            if self.length(piece) > self.chunk_size and remaining:
                if current:
                    chunks.append(separator.join(current))
                    current = []
                chunks.extend(self._split(piece, remaining))
                continue
            candidate = separator.join(current + [piece])
            if current and self.length(candidate) > self.chunk_size:
                chunks.append(separator.join(current))
                while current and self.length(separator.join(current)) > self.chunk_overlap:
                    current.pop(0)
            current.append(piece)
        if current:
            chunks.append(separator.join(current))
        return [c.strip() for c in chunks if c.strip()]


def make_splitter(chunk_size: int, chunk_overlap: int, length_function):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              length_function=length_function, separators=SEPARATORS,
                                              keep_separator=False), "LangChain RecursiveCharacterTextSplitter"
    except Exception:
        return SimpleRecursiveSplitter(chunk_size, chunk_overlap, length_function), "SimpleRecursiveSplitter (pengganti)"


def chunk_documents(documents: list[dict], chunk_size: int = 480, chunk_overlap: int = 64, max_tokens: int = 600,
                    min_section_tokens: int = 150) -> tuple[list[dict], str]:
    """Chunk per bagian; bagian pendek digabung dengan bagian berikutnya; tiap chunk ≤ max_tokens."""
    splitter, splitter_name = make_splitter(chunk_size, chunk_overlap, count_tokens)
    chunks = []
    for document in documents:
        merged_sections, pending_title, pending_text = [], None, ""
        for title, text in split_sections(document):
            if pending_title is not None:
                title, text = f"{pending_title} · {title}", f"{pending_text}\n\n{text}"
                pending_title = None
            if count_tokens(text) < min_section_tokens:
                pending_title, pending_text = title, text
                continue
            merged_sections.append((title, text))
        if pending_title is not None:
            if merged_sections:
                last_title, last_text = merged_sections.pop()
                merged_sections.append((f"{last_title} · {pending_title}", f"{last_text}\n\n{pending_text}"))
            else:
                merged_sections.append((pending_title, pending_text))
        number = 0
        for section_title, section_text in merged_sections:
            pieces = [section_text] if count_tokens(section_text) <= chunk_size else splitter.split_text(section_text)
            for piece in pieces:
                header = f"[{document['title']} › {section_title}]"
                text = f"{header}\n{piece}"
                while count_tokens(text) > max_tokens:            # pengaman: potong kalimat terakhir
                    piece = piece.rsplit(". ", 1)[0] if ". " in piece else piece[: int(len(piece) * 0.9)]
                    text = f"{header}\n{piece}"
                number += 1
                chunks.append({"chunk_id": f"{document['doc_id']}#{number:02d}", "doc_id": document["doc_id"],
                               "title": document["title"], "section": section_title, "category": document.get("category", ""),
                               "source": document.get("source", ""), "url": document.get("url", ""),
                               "text": text, "n_tokens": count_tokens(text)})
    return chunks, splitter_name


def save_chunks(chunks: list[dict], path: Path) -> None:
    Path(path).write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8")


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
