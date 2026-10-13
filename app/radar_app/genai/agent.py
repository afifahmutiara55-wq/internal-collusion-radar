"""Agen investigasi: Plan → Act (tool) → Observe → Deliver, dengan batas wewenang per role.

- Planner LLM (Gemma 4 E4B lewat Ollama) menjawab dalam JSON: panggil tool atau beri jawaban akhir.
- Kalau LLM tidak tersedia, planner berbasis aturan (intent → urutan tool) dipakai sebagai cadangan.
- Guardrail proses: whitelist tool per role, validasi argumen, maksimal 4 langkah, tool hanya-baca,
  dan aksi (draf memo) selalu MENUNGGU_PERSETUJUAN manusia.
"""
from __future__ import annotations

import json
import re
import time

from ..config import SETTINGS
from .guardrails import GuardCheck, validate_input, validate_output
from .llm import LLMUnavailable
from .prompts import AGENT_FINAL_PROMPT, AGENT_SYSTEM_PROMPT
from .tools import ROLE_TOOLS, TOOL_SPECS, ToolError

OBSERVATION_CHAR_LIMIT = 2500


def parse_json_reply(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("tidak ada JSON")
    return json.loads(match.group(0))


class InvestigationAgent:
    def __init__(self, tools, llm_client=None, store=None, settings=SETTINGS, known_names: set | None = None,
                 known_units: set | None = None, embedder=None):
        self.tools, self.llm, self.store, self.settings = tools, llm_client, store, settings
        self.known_names, self.known_units = known_names or set(), known_units or set()
        self.known_entities, self.embedder = self.known_names | self.known_units, embedder

    def llm_ready(self) -> bool:
        return self.llm is not None and self.llm.has_model(self.settings.generator_model)

    # ------------------------------------------------------------------ planner cadangan (tanpa LLM)
    def rule_based_plan(self, question: str, role: str) -> list[tuple[str, dict]]:
        text = question.lower()
        request_id = re.search(r"\bR\d{7}\b", question.upper())
        employee_id = re.search(r"\bE\d{5}\b", question.upper())
        community_id = re.search(r"komunitas\s+(?:no\.?\s*|nomor\s+|id\s+)?(\d{1,3})\b", text)
        named_employee = next((e for e in sorted(self.known_names, key=len, reverse=True) if e.lower() in text), None)
        unit = next((u for u in sorted(self.known_units, key=len, reverse=True)
                     if re.search(rf"\b{re.escape(u.lower())}\b", text)), None)
        month = re.search(r"\b(20\d\d-\d\d)\b", text)
        flag = next((key for key, words in {"near_limit": ["near limit", "mendekati limit"], "after_hours": ["malam", "luar jam"],
                                            "fast_approval": ["kilat", "cepat"], "debtor_repeat": ["debitur berulang", "split"],
                                            "pair_dependency": ["ketergantungan", "bergantung"]}.items()
                     if any(w in text for w in words)), None)
        plan = []
        if request_id and any(w in text for w in ["memo", "draf", "draft", "laporan investigasi"]):
            plan = [("detail_alert", {"request_id": request_id.group(0)}),
                    ("cari_regulasi", {"pertanyaan": "struktur memo investigasi dan langkah klarifikasi"}),
                    ("draf_memo_investigasi", {"request_id": request_id.group(0)})]
        elif request_id:
            plan = [("detail_alert", {"request_id": request_id.group(0)})]
        elif any(w in text for w in ["kapasitas", "investigator", "berapa alert per bulan", "skor minimum"]):
            numbers = re.findall(r"\b(\d{1,2})\s*(?:orang|investigator)", text)
            plan = [("simulasi_kapasitas", {"investigator": int(numbers[0])} if numbers else {})]
        elif "komunitas" in text or "lingkaran" in text or "jaringan" in text:
            args = {"community_id": int(community_id.group(1))} if community_id else (
                {"nama_atau_id": (employee_id.group(0) if employee_id else named_employee)} if (employee_id or named_employee) else {"top": 5})
            plan = [("komunitas_risiko", args)]
        elif employee_id or named_employee:
            target = employee_id.group(0) if employee_id else named_employee
            plan = [("profil_pegawai", {"nama_atau_id": target})] if "alert" not in text else \
                [("cari_alert", {"nama_pegawai": target, "limit": 10})]
        elif any(w in text for w in ["unit mana", "unit paling", "per unit", "cabang paling", "wilayah"]):
            plan = [("risiko_unit", {})]
        elif any(w in text for w in ["kpi", "ringkas", "kondisi risiko", "ringkasan", "eksekutif"]):
            years = re.findall(r"\b(2025|2026)\b", text)
            plan = [("ringkasan_kpi", {"tahun": int(years[0])} if years else {})]
        elif any(w in text for w in ["kinerja model", "performa", "akurasi", "precision", "recall", "adil", "fairness"]):
            plan = [("performa_model", {})]
        elif "alert" in text:
            args = {"limit": int(re.findall(r"\b(\d{1,2})\s+alert", text)[0]) if re.findall(r"\b(\d{1,2})\s+alert", text) else 10}
            if unit:
                args["unit"] = unit
            if "tinggi" in text or "tertinggi" in text:
                args["level"] = "Tinggi"
            if flag:
                args["red_flag"] = flag
            if month:
                args["bulan"] = month.group(1)
            plan = [("cari_alert", args)]
        else:
            plan = [("cari_regulasi", {"pertanyaan": question})]
        return plan

    # ------------------------------------------------------------------ jalankan
    def run(self, question: str, role: str = "investigator", username: str = "-") -> dict:
        started = time.time()
        guard_in = validate_input(question, role, self.settings.max_input_chars, embedder=self.embedder,
                                  llm_client=self.llm, guard_model=self.settings.guard_model,
                                  known_entities=self.known_names)
        result = {"question": question, "question_masked": guard_in.text, "steps": [], "checks": guard_in.flags(),
                  "mode": "blocked", "final_answer": guard_in.refusal, "pending_action": None}
        if not guard_in.allowed:
            return self._finish(result, role, username, started)
        allowed_tools = ROLE_TOOLS[role]
        observations = []
        mode = "llm" if self.llm_ready() else "rule_based"
        if mode == "llm":
            try:
                observations = self._llm_loop(guard_in.text, role, allowed_tools, result)
            except LLMUnavailable:
                mode, observations = "rule_based", []
        if mode == "rule_based":
            for tool, args in self.rule_based_plan(guard_in.text, role)[: self.settings.max_agent_steps]:
                observations.append(self._execute(tool, args, role, result, thought="rencana berbasis aturan"))
        result["mode"] = mode
        result["final_answer"] = self._final_answer(guard_in.text, role, observations, mode, result)
        return self._finish(result, role, username, started)

    def _execute(self, tool: str, args: dict, role: str, result: dict, thought: str = "") -> dict:
        step = {"step": len(result["steps"]) + 1, "thought": thought, "tool": tool, "args": args}
        try:
            if tool not in ROLE_TOOLS[role]:
                raise ToolError(f"tool '{tool}' di luar wewenang role {role}")
            output = self.tools.call(tool, args, role)
            step.update(status="ok", observation=output)
            result["checks"].append(GuardCheck("proses", f"izin tool {tool}", True, f"diizinkan untuk {role}").__dict__)
            if tool == "draf_memo_investigasi":
                result["pending_action"] = output
        except ToolError as error:
            step.update(status="ditolak", observation={"error": str(error)})
            result["checks"].append(GuardCheck("proses", f"izin/argumen tool {tool}", False, str(error)).__dict__)
        result["steps"].append(step)
        return step

    def _llm_loop(self, question: str, role: str, allowed_tools: list, result: dict) -> list:
        tool_descriptions = "\n".join(f"- {name}: {TOOL_SPECS[name]['description']}" for name in allowed_tools)
        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT.format(role=role, tool_descriptions=tool_descriptions,
                                                                            max_steps=self.settings.max_agent_steps)},
                    {"role": "user", "content": question}]
        observations, seen = [], set()
        for _ in range(self.settings.max_agent_steps + 1):
            reply = self.llm.chat(self.settings.generator_model, messages, temperature=0.0, response_format="json")
            try:
                decision = parse_json_reply(reply)
            except (ValueError, json.JSONDecodeError):
                break
            if "final" in decision or len(observations) >= self.settings.max_agent_steps:
                result["llm_final_draft"] = decision.get("final")
                break
            tool, args = decision.get("tool"), decision.get("args") or {}
            signature = (tool, json.dumps(args, sort_keys=True))
            if signature in seen:
                break
            seen.add(signature)
            step = self._execute(tool, args, role, result, thought=decision.get("thought", ""))
            observations.append(step)
            messages += [{"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
                         {"role": "user", "content": "OBSERVASI: " + json.dumps(step["observation"], ensure_ascii=False,
                                                                               default=str)[:OBSERVATION_CHAR_LIMIT]}]
        if not observations:                                  # LLM gagal merencanakan → pakai rencana aturan
            for tool, args in self.rule_based_plan(question, role)[: self.settings.max_agent_steps]:
                observations.append(self._execute(tool, args, role, result, thought="cadangan: rencana berbasis aturan"))
        return observations

    def _final_answer(self, question: str, role: str, observations: list, mode: str, result: dict) -> str:
        successful = [o for o in observations if o["status"] == "ok"]
        if not successful:
            rejected = "; ".join(o["observation"]["error"] for o in observations) or "tidak ada tool yang cocok"
            return f"Permintaan tidak dapat dipenuhi: {rejected}."
        evidence_text = json.dumps([o["observation"] for o in successful], ensure_ascii=False, default=str)
        if mode == "llm":
            try:
                prompt = AGENT_FINAL_PROMPT.format(question=question, observations=evidence_text[:6000])
                draft = self.llm.chat(self.settings.generator_model, [{"role": "user", "content": prompt}],
                                      self.settings.generator_temperature)
                guard_out = validate_output(draft, evidence_text=evidence_text, llm_client=self.llm,
                                            guard_model=self.settings.guard_model, question=question)
                result["checks"] += guard_out.flags()
                return guard_out.text
            except LLMUnavailable:
                pass
        answer = self.template_answer(successful)
        guard_out = validate_output(answer, evidence_text=evidence_text)
        result["checks"] += guard_out.flags()
        return guard_out.text

    @staticmethod
    def template_answer(steps: list) -> str:
        """Jawaban akhir tanpa LLM: rangkuman terstruktur dari hasil setiap tool."""
        parts = []
        for step in steps:
            tool, output = step["tool"], step["observation"]
            if tool == "cari_alert":
                lines = [f"- {a['request_id']} · {a['event_time']} · {a['requester']} → {a['approver']} ({a['requester_unit']}) · "
                         f"skor {a['risk_score']:.0f} ({a['risk_band']}) · {a.get('top_reasons') or ''}" for a in output["alert"]]
                parts.append(f"Ditemukan {output['jumlah_ditemukan']} alert; {output['ditampilkan']} teratas (sumber: cari_alert):\n" + "\n".join(lines))
            elif tool == "detail_alert":
                reasons = "; ".join(r["alasan"] for r in output["alasan_shap"]) or "-"
                parts.append(f"Transaksi {output['request_id']} berskor {output['skor_risiko']:.0f}/100 (kategori {output['kategori']}). "
                             f"Requester {output['requester']}; approver {output['approver']}. Alasan utama: {reasons}. "
                             f"Red flag: {', '.join(output['red_flag']) or 'tidak ada'}. Skor anomali approver: "
                             f"{output['skor_anomali_approver'] if output['skor_anomali_approver'] is not None else '-'} (sumber: detail_alert). "
                             "Ini indikasi berbasis data yang perlu diklarifikasi, bukan kesimpulan.")
            elif tool == "profil_pegawai":
                parts.append(f"{output['nama']} ({output['employee_id']}) — {output['jabatan']}, unit {output['unit']}. Alert Tinggi 2026: "
                             f"{output['alert_tinggi_2026']}. Skor anomali terakhir: {output.get('skor_anomali_terakhir', '-')} "
                             f"({output.get('kategori_anomali', '-')}); {output.get('alasan_anomali') or ''}. Komunitas risiko: "
                             f"{output['komunitas_risiko'] if output['komunitas_risiko'] is not None else 'tidak ada'} (sumber: profil_pegawai).")
            elif tool == "komunitas_risiko":
                if "anggota" in output:
                    info = output["komunitas"]
                    members = ", ".join(f"{m['employee_name']} ({m['employee_id']}, {m['role']})" for m in output["anggota"])
                    parts.append(f"Komunitas {info['community_id']}: {info['anggota']} anggota, skor {info['skor_komunitas']:.0f}/100, "
                                 f"prioritas {info['prioritas']}, unit {info['unit']}. Anggota: {members} (sumber: komunitas_risiko).")
                elif "teratas" in output:
                    lines = [f"- Komunitas {c['community_id']} · skor {c['skor_komunitas']:.0f} · {c['anggota']} orang · {c['unit']} · "
                             f"approver: {c['approver_utama']}" for c in output["teratas"]]
                    parts.append(f"{output['jumlah_komunitas']} komunitas risiko aktif; teratas (sumber: komunitas_risiko):\n" + "\n".join(lines))
                else:
                    parts.append(output.get("pesan", ""))
            elif tool == "cari_regulasi":
                lines = [f"- [{s['no']}] {s['dokumen']} › {s['bagian']}: {s['kutipan'][:260]}…" for s in output["sumber"][:3]]
                parts.append("Rujukan knowledge base (sumber: cari_regulasi):\n" + "\n".join(lines))
            elif tool == "draf_memo_investigasi":
                parts.append("Draf memo investigasi sudah disusun dan MENUNGGU PERSETUJUAN investigator (lihat panel di bawah).")
            elif tool == "ringkasan_kpi":
                parts.append(f"Tahun {output['tahun']} ({output['bulan_data']} bulan): {output['transaksi_dinilai']:,} transaksi dinilai, "
                             f"{output['alert_tinggi']:,} alert Tinggi (±{output['alert_tinggi_per_bulan']:.0f}/bulan), nominal alert Tinggi "
                             f"Rp{output['nominal_alert_tinggi'] / 1e9:,.1f} M. Precision populasi diperkirakan "
                             f"{output['precision_populasi_batas_bawah']:.0%}–{output['precision_populasi_batas_atas']:.0%}. "
                             f"Sebagai pembanding, rules menandai {output['transaksi_rules_menandai']:,} transaksi (sumber: ringkasan_kpi).")
            elif tool == "risiko_unit":
                lines = [f"- {u['requester_unit']} ({u['region']}): {u['alert_tinggi']} alert Tinggi dari {u['transaksi']} transaksi "
                         f"({u['alert_rate']:.1%})" for u in output["unit_teratas"]]
                parts.append(f"Unit dengan tingkat alert tertinggi {output['tahun']} (sumber: risiko_unit):\n" + "\n".join(lines))
            elif tool == "simulasi_kapasitas":
                if "investigator" in output:
                    parts.append(f"Dengan {output['investigator']} investigator, kapasitas ±{output['kapasitas_alert_per_bulan']:.0f} alert/bulan. "
                                 f"Skor minimum yang disarankan {output['skor_minimum_disarankan']:.0f} → ±{output['alert_per_bulan']:.0f} alert/bulan, "
                                 f"recall {output['recall_test']:.0%}, precision {output['precision_test']:.0%} (sumber: simulasi_kapasitas).")
                else:
                    parts.append(f"Pada skor minimum {output['skor_minimum']:.0f}: ±{output['alert_per_bulan']:.0f} alert/bulan, recall "
                                 f"{output['recall_test']:.0%}, precision {output['precision_test']:.0%}, butuh ±{output['investigator_dibutuhkan']:.1f} "
                                 "investigator (sumber: simulasi_kapasitas).")
            elif tool == "performa_model":
                lines = [f"- {m['model']}: precision {m['precision']:.0%}, recall {m['recall']:.0%}, PR-AUC {m['PR-AUC']:.3f}"
                         for m in output["classifier_test_2026"]]
                low, high = output["fairness_FPR_model_rentang"]
                parts.append("Kinerja di test 2026 (sumber: performa_model):\n" + "\n".join(lines)
                             + f"\nAlarm palsu (FPR) model antarsegmen berkisar {low:.1%}–{high:.1%}.")
        return "\n\n".join(p for p in parts if p)

    def _finish(self, result: dict, role: str, username: str, started: float) -> dict:
        result["latency_ms"] = (time.time() - started) * 1000
        if self.store is not None:
            self.store.log_ai_event({
                "username": username, "role": role, "mode": f"agent:{result['mode']}", "question_masked": result["question_masked"],
                "guard_flags": json.dumps(result["checks"], ensure_ascii=False, default=str), "retrieved": "[]",
                "tool_calls": json.dumps([{k: s[k] for k in ("step", "tool", "args", "status")} for s in result["steps"]],
                                         ensure_ascii=False, default=str),
                "answer": result["final_answer"], "latency_ms": result["latency_ms"], "model": self.settings.generator_model,
                "fallback_used": result["mode"] != "llm", "blocked": result["mode"] == "blocked"})
        return result
