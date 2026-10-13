-- Kumpulan query untuk DBeaver (koneksi: PostgreSQL, host localhost, port 5432, database radar, user/password dari .env)

-- 1. Alert Tinggi 2026 yang belum diputus investigator, urut skor
SELECT t.request_id, t.event_time, t.requester_id, t.approver_id, t.requester_unit, t.amount, t.risk_score, t.top_reasons
FROM radar.transactions_scored t
LEFT JOIN (SELECT DISTINCT request_id FROM app.case_reviews) r ON r.request_id = t.request_id
WHERE t.risk_band = 'Tinggi' AND EXTRACT(YEAR FROM t.event_time) = 2026 AND r.request_id IS NULL
ORDER BY t.risk_score DESC, t.event_time DESC LIMIT 50;

-- 2. Alasan SHAP untuk satu transaksi
SELECT rank, feature, feature_group, feature_value, shap_value, reason
FROM radar.alert_explanations WHERE request_id = 'R0156061' ORDER BY rank;

-- 3. Komunitas risiko terkuat beserta anggotanya
SELECT c.community_id, c.skor_komunitas, c.prioritas, m.employee_name, m.role, m.position, m.unit, m.pagerank
FROM radar.communities c JOIN radar.community_members m USING (community_id)
ORDER BY c.skor_komunitas DESC, m.pagerank DESC;

-- 4. Approver dengan skor anomali ≥ 95 pada bulan terakhir
SELECT approver_id, employee_name, position, unit, month, if_score, if_reason
FROM radar.approver_anomaly
WHERE month = (SELECT MAX(month) FROM radar.approver_anomaly) AND if_score >= 95 ORDER BY if_score DESC;

-- 5. Tingkat alert per unit dan wilayah
SELECT region, requester_unit, transaksi, alert_tinggi, ROUND(alert_rate::numeric, 4) AS alert_rate
FROM radar.unit_risk WHERE year = 2026 ORDER BY alert_rate DESC;

-- 6. Jejak audit AI: interaksi yang diblokir guardrail
SELECT logged_at, username, role, mode, question_masked, answer FROM app.ai_audit_log WHERE blocked ORDER BY logged_at DESC;

-- 7. Umpan balik investigator (precision di lapangan)
SELECT decision, COUNT(*) AS jumlah FROM app.case_reviews GROUP BY decision ORDER BY jumlah DESC;
