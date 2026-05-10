"""Tests T06 — Anomalies (todos los detectores + consenso)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.anomalies.benford import detect_benford_anomalies
from src.anomalies.consensus import build_consensus_matrix
from src.anomalies.isolation_forest import detect_isolation_forest
from src.anomalies.statistical import detect_iqr, detect_zscore
from src.anomalies.temporal import detect_temporal_anomalies
from src.etl.cleaner import clean_dataset
from src.etl.features import add_features
from src.etl.generator import generate_dataset


def test_consensus_severity_thresholds() -> None:
    """Verifica que el score ponderado y la severidad se calculen correctamente.

    Pesos: IF=0.30, Z-score=0.15, IQR=0.15, Benford=0.15, Temporal=0.25
    Umbrales: >=0.70→crítico, >=0.45→alto, >=0.25→medio, >=0.10→bajo
    """
    df_test = pd.DataFrame({
        "fecha": pd.date_range("2024-01-01", periods=1, freq="D"),
    })
    true_ser: pd.Series = pd.Series([True], dtype=bool)
    false_ser: pd.Series = pd.Series([False], dtype=bool)

    # Caso 1: todos los 5 métodos → score=1.0 → 'crítico'
    consenso = build_consensus_matrix(
        df_test, true_ser, true_ser, true_ser, true_ser, true_ser
    )
    assert consenso["consenso_score"].iloc[0] == 1.0
    assert consenso["severidad"].iloc[0] == "crítico"

    # Caso 2: IF (0.30) + Temporal (0.25) = 0.55 → 'alto'
    consenso = build_consensus_matrix(
        df_test, true_ser, false_ser, false_ser, false_ser, true_ser
    )
    assert consenso["consenso_score"].iloc[0] == pytest.approx(0.55)
    assert consenso["severidad"].iloc[0] == "alto"

    # Caso 3: solo IF (0.30) → 'medio'
    consenso = build_consensus_matrix(
        df_test, true_ser, false_ser, false_ser, false_ser, false_ser
    )
    assert consenso["consenso_score"].iloc[0] == pytest.approx(0.30)
    assert consenso["severidad"].iloc[0] == "medio"

    # Caso 4: ninguno detecta → score=0.0 → severidad=None
    consenso = build_consensus_matrix(
        df_test, false_ser, false_ser, false_ser, false_ser, false_ser
    )
    assert consenso["consenso_score"].iloc[0] == 0.0
    assert consenso["severidad"].iloc[0] is None


def test_detection_rate_vs_known_anomalies() -> None:
    """Verifica que al menos 1 detector encuentre ≥60% de anomalías conocidas.

    NOTA: usa la columna 'tiene_anomalia' del dataset generado como ground truth.
    """
    df = generate_dataset(seed=42, periods=24)
    df_clean, _ = clean_dataset(df)
    df_feat = add_features(df_clean)

    # Ejecutar todos los detectores
    if_anom: pd.Series = detect_isolation_forest(df_feat)
    z_anom: pd.Series = detect_zscore(df_feat)
    iqr_anom: pd.Series = detect_iqr(df_feat)
    benf_anom: pd.Series = detect_benford_anomalies(df_feat)
    temp_df: pd.DataFrame = detect_temporal_anomalies(df_feat)
    temp_anom: pd.Series = temp_df["es_anomalia_temporal"]

    # Construir matriz de consenso
    _consenso = build_consensus_matrix(
        df_feat, if_anom, z_anom, iqr_anom, benf_anom, temp_anom
    )

    # Cargar anomaly_report.csv existente
    anomaly_report = pd.read_csv("reports/anomaly_report.csv")
    assert not anomaly_report.empty, "anomaly_report.csv está vacío"

    # Usar 'tiene_anomalia' como ground truth (columnas preservadas en df_feat)
    ground_truth: pd.Series = df_feat["tiene_anomalia"]
    known_mask = ground_truth == True
    n_known = int(known_mask.sum())
    assert n_known > 0, "No hay anomalías conocidas en el dataset generado"

    detectors = {
        "Isolation Forest": if_anom,
        "Z-score": z_anom,
        "IQR": iqr_anom,
        "Benford": benf_anom,
        "Temporal": temp_anom,
    }

    best_recall = 0.0
    best_name = ""
    for name, det in detectors.items():
        true_positives = int((det & ground_truth).sum())
        recall = true_positives / n_known
        if recall > best_recall:
            best_recall = recall
            best_name = name

    assert best_recall >= 0.60, (
        f"Ningún detector alcanzó ≥60% de recall. "
        f"Mejor: '{best_name}' con {best_recall:.1%}"
    )


def test_isolation_forest_returns_boolean_series() -> None:
    """Verifica que detect_isolation_forest retorne una Serie booleana válida."""
    df = generate_dataset(seed=42, periods=12)
    df_clean, _ = clean_dataset(df)
    df_feat = add_features(df_clean)

    result: pd.Series = detect_isolation_forest(df_feat)

    assert isinstance(result, pd.Series), "El resultado debe ser pd.Series"
    assert result.dtype == bool, f"El dtype debe ser bool, got {result.dtype}"
    assert len(result) == len(df_feat), (
        f"Longitud {len(result)} != {len(df_feat)}"
    )
    assert result.index.equals(df_feat.index), "Los índices no coinciden"
