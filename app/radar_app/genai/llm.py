"""Klien LLM lokal (Ollama) + generator cadangan ekstraktif.

Generator utama: Gemma 4 E4B-it (`gemma4:e4b` di Ollama), temperature 0,1 → jawaban konsisten dan minim "kreativitas".
Kalau Ollama/model tidak tersedia, sistem TIDAK berhenti: jawaban disusun secara ekstraktif dari kalimat konteks yang
paling relevan, lengkap dengan sitasi. Ini mekanisme *fallback* yang diminta guardrail lapis monitoring.
"""
from __future__ import annotations

import re
import time

import requests

from .retriever import tokenize


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._models_cache: tuple[float, set] | None = None

    def available_models(self) -> set:
        if self._models_cache and time.time() - self._models_cache[0] < 30:
            return self._models_cache[1]
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            models = {m["name"] for m in response.json().get("models", [])}
            models |= {name.split(":")[0] + ":latest" for name in models if ":" not in name}
        except Exception:
            models = set()
        self._models_cache = (time.time(), models)
        return models

    def has_model(self, model: str) -> bool:
        models = self.available_models()
        return model in models or (":" not in model and f"{model}:latest" in models)

    def chat(self, model: str, messages: list[dict], temperature: float = 0.1, num_ctx: int = 8192,
             response_format: str | None = None, max_tokens: int = 900) -> str:
        if not self.has_model(model):
            raise LLMUnavailable(f"model {model} tidak tersedia di {self.base_url}")
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": max_tokens}}
        if response_format:
            payload["format"] = response_format
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except Exception as error:
            raise LLMUnavailable(str(error)) from error
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def context_units(text: str) -> list[str]:
    """Pecah isi chunk menjadi kalimat/baris (baris daftar tetap utuh)."""
    body = re.sub(r"^\[.*?\]\n", "", text)
    units = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        if raw_line.startswith((" ", "\t")) and units and re.match(r"^(\d+\.|[-*])\s", units[-1]):
            units[-1] = f"{units[-1]} {line}"          # lanjutan butir daftar yang terpotong baris
            continue
        if re.match(r"^(\d+\.|[-*])\s", line) or line.startswith("|"):
            units.append(line)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return [re.sub(r"\*\*|`", "", u).strip(" |") for u in units]


def extractive_answer(question: str, contexts: list[dict], max_units: int = 4) -> str:
    """Jawaban cadangan: kalimat/baris konteks dengan tumpang-tindih kata tertinggi terhadap pertanyaan, bersitasi [S#].
    Kalau kalimat terpilih diakhiri titik dua (pengantar daftar), butir daftar sesudahnya ikut disertakan."""
    query_terms = set(tokenize(question))
    candidates = []
    for source_number, context in enumerate(contexts, start=1):
        units = context_units(context["text"])
        for position, unit in enumerate(units):
            overlap = len(query_terms & set(tokenize(unit))) / (len(query_terms) + 1e-9)
            candidates.append((overlap - 0.01 * position - 0.03 * source_number, source_number, position, units))
    chosen = [c for c in sorted(candidates, key=lambda c: -c[0])[:max_units] if c[0] > 0.05]
    if not chosen:
        return "Maaf, informasi tersebut tidak ditemukan di knowledge base."
    lines, used = [], set()
    for _, source_number, position, units in sorted(chosen, key=lambda c: (c[1], c[2])):
        span = [position]
        if units[position].endswith(":"):
            span += [p for p in range(position + 1, min(position + 6, len(units))) if re.match(r"^(\d+\.|-)\s", units[p])]
        for p in span:
            if (source_number, p) not in used and len(units[p]) > 3:
                used.add((source_number, p))
                lines.append(f"{units[p]} [S{source_number}]")
    return "\n".join(lines)
