"""Objek kecil yang ikut tersimpan di file model (joblib).

Kenapa di modul terpisah? File joblib menyimpan *referensi* ke kelas, bukan kodenya. Kalau kelas ini hanya didefinisikan
di notebook, aplikasi Streamlit tidak bisa membuka model. Dengan menaruhnya di sini, notebook (training) dan aplikasi
(scoring) memakai kelas yang sama persis.

Isi:
- SimpleSMOTE            : SMOTE sederhana (dipakai hanya kalau paket imbalanced-learn tidak terpasang)
- ResamplingClassifier   : preprocessing -> resampling (SMOTE) -> model, resampling HANYA saat fit (data latih)
- CalibratedRiskModel    : model + kalibrasi isotonic + threshold -> skor risiko 0-100 yang bisa dipertanggungjawabkan
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors


class SimpleSMOTE:
    """SMOTE (Chawla et al., 2002): buat titik sintetis di antara satu kasus minoritas dan tetangga terdekatnya."""

    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        minority = X[y == 1]
        n_needed = int((y == 0).sum() - (y == 1).sum())
        if n_needed <= 0 or len(minority) <= self.k_neighbors:
            return X, y
        neighbors = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(minority)
        _, neighbor_index = neighbors.kneighbors(minority)
        base = rng.integers(0, len(minority), n_needed)
        partner = neighbor_index[base, rng.integers(1, self.k_neighbors + 1, n_needed)]
        gap = rng.random((n_needed, 1))
        synthetic = minority[base] + gap * (minority[partner] - minority[base])
        return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_needed, dtype=int)])


class ResamplingClassifier(ClassifierMixin, BaseEstimator):
    """preprocessor -> sampler.fit_resample (hanya saat fit) -> estimator.

    Setara `imblearn.pipeline.Pipeline`, tapi tidak butuh paket tambahan. Saat predict, TIDAK ada resampling:
    data validasi/test/produksi dinilai apa adanya.
    """

    def __init__(self, preprocessor=None, sampler=None, estimator=None):
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.estimator = estimator

    def fit(self, X, y):
        self.preprocessor_ = clone(self.preprocessor).fit(X, y)
        X_transformed = self.preprocessor_.transform(X)
        X_resampled, y_resampled = self.sampler.fit_resample(X_transformed, np.asarray(y))
        self.estimator_ = clone(self.estimator).fit(X_resampled, y_resampled)
        self.classes_ = self.estimator_.classes_
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.preprocessor_.transform(X))

    def predict(self, X):
        return self.estimator_.predict(self.preprocessor_.transform(X))


def split_preprocessor_and_estimator(model):
    """Ambil (preprocessor terlatih, estimator terlatih) dari Pipeline sklearn/imblearn atau ResamplingClassifier."""
    if isinstance(model, ResamplingClassifier):
        return model.preprocessor_, model.estimator_
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        return (steps[0] if len(steps) > 1 else None), steps[-1]
    return None, model


class CalibratedRiskModel:
    """Skor akhir transaksi = probabilitas terkalibrasi (0-1), ditampilkan sebagai 0-100.

    - base_model   : pipeline terlatih (train 2021-2024)
    - calibrator   : IsotonicRegression yang dipetakan di data validasi 2025 (skor mentah -> peluang empiris)
    - threshold    : batas prioritas "Tinggi" (dipilih di validasi, pada skala terkalibrasi)
    Isotonic bersifat monoton, jadi URUTAN skor tidak berubah; hanya angkanya yang jadi bisa dibaca sebagai peluang.
    """

    def __init__(self, base_model, calibrator, threshold: float, features: list[str],
                 categorical_features: list[str], model_name: str, watch_ratio: float = 0.5):
        self.base_model = base_model
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.features = list(features)
        self.categorical_features = list(categorical_features)
        self.model_name = model_name
        self.watch_ratio = watch_ratio

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        matrix = df[self.features].copy()
        numeric = [f for f in self.features if f not in self.categorical_features]
        matrix[numeric] = matrix[numeric].astype(float)
        for column in self.categorical_features:
            matrix[column] = matrix[column].astype(object).where(matrix[column].notna(), np.nan)
        return matrix

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(self.prepare(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw_score(df)), 0, 1)

    def risk_band(self, probability) -> np.ndarray:
        probability = np.asarray(probability)
        return np.select([probability >= self.threshold, probability >= self.threshold * self.watch_ratio],
                         ["Tinggi", "Sedang"], "Rendah")

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        probability = self.predict_proba(df)
        return pd.DataFrame({"risk_probability": probability, "risk_score": np.round(probability * 100, 1),
                             "risk_band": self.risk_band(probability)}, index=df.index)
