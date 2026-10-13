"""Pipeline RAG lengkap dengan guardrail dan audit log.

pertanyaan → [guardrail input] → hybrid retrieval (FAISS + BM25, RRF, top 4) → [cek kecukupan konteks]
          → Gemma 4 E4B (temperature 0,1; jawab hanya dari konteks + sitasi) atau jawaban ekstraktif cadangan
          → [guardrail output] → audit log → jawaban + sumber + status guardrail
"""
from __future__ import annotations

import json
import time

from ..config import SETTINGS
from .guardrails import ABSTAIN_MESSAGE, GuardCheck, validate_input, validate_output
from .llm import LLMUnavailable, extractive_answer
from .prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

MIN_DENSE_SCORE = 0.30        # cosine minimum (MiniLM) agar konteks dianggap relevan
MIN_BM25_SCORE = 1.0


def format_contexts(contexts: list[dict]) -> str:
    return "\n\n".join(f"[S{i}] ({c['title']} › {c['section']})\n{c['text'].split(chr(10), 1)[-1]}"
                       for i, c in enumerate(contexts, start=1))


class RagAssistant:
    def __init__(self, retriever, llm_client=None, store=None, settings=SETTINGS, known_entities: set | None = None):
        self.retriever, self.llm, self.store, self.settings = retriever, llm_client, store, settings
        self.known_entities = known_entities or set()

    def llm_ready(self) -> bool:
        return self.llm is not None and self.llm.has_model(self.settings.generator_model)

    def retrieve(self, question: str) -> list[dict]:
        return self.retriever.hybrid_search(question, self.settings.dense_k, self.settings.sparse_k,
                                            self.settings.retrieval_top_n)

    def context_is_sufficient(self, contexts: list[dict]) -> bool:
        if not contexts:
            return False
        best_dense = max((c.get("dense_score") or 0.0) for c in contexts)
        best_bm25 = max((c.get("bm25_score") or 0.0) for c in contexts)
        dense_threshold = MIN_DENSE_SCORE if "MiniLM" in self.retriever.embedder.name else 0.10
        return best_dense >= dense_threshold or best_bm25 >= MIN_BM25_SCORE

    def generate(self, question: str, contexts: list[dict], role: str) -> tuple[str, str]:
        if self.llm_ready():
            try:
                messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT.format(role=role)},
                            {"role": "user", "content": RAG_USER_TEMPLATE.format(contexts=format_contexts(contexts),
                                                                                 question=question)}]
                return self.llm.chat(self.settings.generator_model, messages, self.settings.generator_temperature), "llm"
            except LLMUnavailable:
                pass
        return "(Mode cadangan tanpa LLM) " + extractive_answer(question, contexts), "fallback"

    def answer(self, question: str, role: str = "investigator", username: str = "-") -> dict:
        started = time.time()
        guard_in = validate_input(question, role, self.settings.max_input_chars,
                                  embedder=self.retriever.embedder, llm_client=self.llm,
                                  guard_model=self.settings.guard_model, known_entities=self.known_entities)
        result = {"question": question, "question_masked": guard_in.text, "contexts": [], "mode": "blocked",
                  "checks": guard_in.flags(), "answer": guard_in.refusal}
        if guard_in.allowed:
            contexts = self.retrieve(guard_in.text)
            result["contexts"] = contexts
            if not self.context_is_sufficient(contexts):
                result.update(answer=ABSTAIN_MESSAGE, mode="abstain")
                result["checks"].append(GuardCheck("output", "konteks memadai", False, "skor retrieval rendah → menjawab tidak tahu").__dict__)
            else:
                draft, mode = self.generate(guard_in.text, contexts, role)
                guard_out = validate_output(draft, contexts, llm_client=self.llm, guard_model=self.settings.guard_model,
                                            question=guard_in.text)
                answer = guard_out.text
                if guard_out.refusal and "keamanan" in guard_out.refusal:
                    answer, mode = guard_out.refusal, "blocked"
                elif not guard_out.allowed and mode == "llm":
                    answer = ("(Jawaban LLM kurang ter-grounding; ditampilkan kutipan langsung dari sumber) "
                              + extractive_answer(guard_in.text, contexts))
                    mode = "fallback_grounding"
                result.update(answer=answer, mode=mode)
                result["checks"] += guard_out.flags()
        result["latency_ms"] = (time.time() - started) * 1000
        self._log(result, role, username)
        return result

    def _log(self, result: dict, role: str, username: str) -> None:
        if self.store is None:
            return
        self.store.log_ai_event({
            "username": username, "role": role, "mode": f"rag:{result['mode']}", "question_masked": result["question_masked"],
            "guard_flags": json.dumps(result["checks"], ensure_ascii=False),
            "retrieved": json.dumps([c["chunk_id"] for c in result["contexts"]]), "tool_calls": "[]",
            "answer": result["answer"], "latency_ms": result["latency_ms"], "model": self.settings.generator_model,
            "fallback_used": result["mode"] not in ("llm",), "blocked": result["mode"] == "blocked"})
