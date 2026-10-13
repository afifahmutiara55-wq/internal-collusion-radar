"""LLM-as-a-Judge untuk mengevaluasi kualitas jawaban RAG.

Desain juri yang "bijak" (mengurangi bias yang diketahui dari literatur LLM-as-a-judge, mis. Zheng dkk., 2023):
1. **Model juri berbeda keluarga dengan generator** (generator Gemma → juri default Qwen2.5-7B-Instruct) untuk mengurangi
   *self-preference bias*. Bisa diganti model yang lebih kuat lewat env JUDGE_MODEL.
2. **Satu kriteria per panggilan** (faithfulness, relevance, correctness, citation, safety_tone) supaya tidak terjadi
   *halo effect*; rubrik 1–5 dengan deskripsi tiap skor; instruksi untuk tidak menilai panjang jawaban (*verbosity bias*).
3. **Alasan dulu, baru skor** (JSON), temperature 0 → konsisten dan bisa diaudit.
4. **Dikalibrasi terhadap manusia**: kesepakatan dan Cohen's kappa dihitung terhadap label manusia
   (knowledge_base/eval/human_labels_template.csv) sebelum hasil juri dipercaya.
5. **Pemeriksaan deterministik** (sitasi valid, grounding leksikal, kata menuduh, PII) dijalankan berdampingan sebagai
   pembanding yang tidak bisa "berhalusinasi".

Mesin yang dipakai (otomatis memilih yang tersedia):
- `mlflow.genai.judges.make_judge` dengan model `ollama:/<JUDGE_MODEL>` (MLflow ≥ 3.4) → hasil tercatat di MLflow;
- panggilan langsung ke Ollama dengan rubrik yang sama;
- heuristik deterministik (bukan LLM) bila tidak ada LLM sama sekali — diberi label jelas.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .guardrails import ACCUSATORY_REPLACEMENTS, citation_check, groundedness_score, mask_pii
from .llm import LLMUnavailable
from .prompts import JUDGE_PROMPT, JUDGE_RUBRICS
from .retriever import tokenize

CRITERIA = list(JUDGE_RUBRICS)


def _contexts_text(contexts: list[dict]) -> str:
    return "\n\n".join(f"[S{i}] {c['text']}" for i, c in enumerate(contexts, start=1))[:6000]


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = tokenize(prediction), tokenize(reference)
    if not predicted or not expected:
        return 0.0
    common = sum(min(predicted.count(t), expected.count(t)) for t in set(predicted))
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def heuristic_judge(question: str, answer: str, contexts: list[dict], reference: str = "") -> dict:
    """Penilaian deterministik 1–5 per kriteria (proksi murah; BUKAN pengganti juri LLM)."""
    abstained = "tidak ada di knowledge base" in answer or "tidak ditemukan di knowledge base" in answer
    grounding = groundedness_score(answer, contexts) if contexts else 0.0
    question_terms = set(tokenize(question))
    relevance = len(question_terms & set(tokenize(answer))) / max(len(question_terms), 1)
    reference_terms = set(tokenize(reference)) if reference else set()
    correctness = (len(reference_terms & set(tokenize(answer))) / len(reference_terms)) if reference_terms else np.nan
    citations_ok, cited = citation_check(answer, len(contexts))
    accusatory = sum(bool(re.search(p, answer, flags=re.I)) for p, _ in ACCUSATORY_REPLACEMENTS)
    _, pii = mask_pii(answer)
    to_scale = lambda x: float(np.clip(1 + 4 * x, 1, 5))
    scores = {"faithfulness": 5.0 if abstained else to_scale(grounding),
              "relevance": 3.0 if abstained else to_scale(min(1.0, relevance * 1.6)),
              "correctness": np.nan if np.isnan(correctness) else (2.0 if abstained else to_scale(correctness)),
              "citation": 5.0 if (citations_ok and cited) else (4.0 if abstained else 1.0),
              "safety_tone": 1.0 if pii else (3.0 if accusatory else 5.0)}
    rationale = {"faithfulness": f"{grounding:.0%} kalimat didukung konteks (overlap kata)",
                 "relevance": f"{relevance:.0%} kata kunci pertanyaan muncul di jawaban",
                 "correctness": "tidak ada acuan" if np.isnan(correctness) else f"{correctness:.0%} kata penting jawaban acuan tercakup",
                 "citation": f"sitasi {cited} {'valid' if citations_ok else 'tidak valid'}",
                 "safety_tone": f"PII: {pii or '-'}; frasa menuduh: {accusatory}"}
    return {c: {"score": scores[c], "rationale": rationale[c]} for c in CRITERIA}


class LLMJudge:
    def __init__(self, llm_client=None, judge_model: str = "qwen2.5:7b-instruct", use_mlflow: bool = True):
        self.llm, self.judge_model, self.use_mlflow = llm_client, judge_model, use_mlflow
        self._mlflow_judges = None

    @property
    def engine(self) -> str:
        if self._mlflow_ready():
            return f"MLflow make_judge (ollama:/{self.judge_model})"
        if self.llm is not None and self.llm.has_model(self.judge_model):
            return f"Ollama langsung ({self.judge_model})"
        return "heuristik deterministik (LLM juri tidak tersedia)"

    def _mlflow_ready(self) -> bool:
        if not self.use_mlflow or self.llm is None or not self.llm.has_model(self.judge_model):
            return False
        if self._mlflow_judges is None:
            try:
                from mlflow.genai.judges import make_judge
                judges = {}
                for criterion, (name, rubric) in JUDGE_RUBRICS.items():
                    instructions = (f"Nilai SATU kriteria: {name}. Rubrik: {rubric}\n"
                                    "Pertanyaan dan konteks ada di {{ inputs }}; jawaban di {{ outputs }}; "
                                    "jawaban acuan di {{ expectations }}. Berikan alasan singkat lalu skor bulat 1-5.")
                    try:
                        judges[criterion] = make_judge(name=criterion, instructions=instructions,
                                                       model=f"ollama:/{self.judge_model}", feedback_value_type=int)
                    except TypeError:
                        judges[criterion] = make_judge(name=criterion, instructions=instructions,
                                                       model=f"ollama:/{self.judge_model}")
                self._mlflow_judges = judges
            except Exception:
                self._mlflow_judges = {}
        return bool(self._mlflow_judges)

    def judge(self, question: str, answer: str, contexts: list[dict], reference: str = "",
              criteria: list[str] | None = None) -> dict:
        criteria = criteria or CRITERIA
        if self._mlflow_ready():
            results = {}
            for criterion in criteria:
                try:
                    feedback = self._mlflow_judges[criterion](
                        inputs={"question": question, "contexts": _contexts_text(contexts)},
                        outputs={"answer": answer}, expectations={"reference_answer": reference or "-"})
                    results[criterion] = {"score": float(feedback.value), "rationale": str(feedback.rationale)[:500]}
                except Exception as error:
                    results[criterion] = {"score": np.nan, "rationale": f"gagal: {error}"}
            return results
        if self.llm is not None and self.llm.has_model(self.judge_model):
            results = {}
            for criterion in criteria:
                name, rubric = JUDGE_RUBRICS[criterion]
                prompt = JUDGE_PROMPT.format(criterion_name=name, rubric=rubric, question=question,
                                             contexts=_contexts_text(contexts), reference=reference or "-", answer=answer)
                try:
                    reply = self.llm.chat(self.judge_model, [{"role": "user", "content": prompt}], temperature=0.0,
                                          response_format="json", max_tokens=300)
                    parsed = json.loads(re.search(r"\{.*\}", reply, flags=re.S).group(0))
                    results[criterion] = {"score": float(parsed["skor"]), "rationale": str(parsed.get("alasan", ""))[:500]}
                except (LLMUnavailable, AttributeError, KeyError, ValueError, json.JSONDecodeError) as error:
                    results[criterion] = {"score": np.nan, "rationale": f"gagal parse: {error}"}
            return results
        heuristic = heuristic_judge(question, answer, contexts, reference)
        return {c: heuristic[c] for c in criteria}


def agreement_with_humans(judge_scores: list[float], human_scores: list[float]) -> dict:
    """Kesepakatan juri vs manusia: % sama persis, % selisih ≤ 1, dan Cohen's kappa berbobot kuadrat."""
    judge, human = np.asarray(judge_scores, dtype=float), np.asarray(human_scores, dtype=float)
    mask = ~(np.isnan(judge) | np.isnan(human))
    judge, human = np.rint(judge[mask]).astype(int), np.rint(human[mask]).astype(int)
    if len(judge) == 0:
        return {"n": 0}
    from sklearn.metrics import cohen_kappa_score
    return {"n": int(len(judge)), "sama_persis": float((judge == human).mean()),
            "selisih_maks_1": float((np.abs(judge - human) <= 1).mean()),
            "cohen_kappa_kuadratik": float(cohen_kappa_score(human, judge, weights="quadratic"))}
