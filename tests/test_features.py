"""Tests T03 — Features (src.etl.features)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.etl.cleaner import clean_dataset
from src.etl.features import add_features
from src.etl.generator import generate_dataset


def test_no_future_leakage() -> None:
    """Verifica que lag_1 y rolling_mean_7 no usen información del día actual."""
    df = generate_dataset(seed=42, periods=12)
    df_clean, _ = clean_dataset(df)
    df_feat = add_features(df_clean)

    # lag_1 debe ser caja_neta del día anterior, no del mismo día
    lag_matches = (df_feat["lag_1"] == df_feat["caja_neta"]).sum()
    assert lag_matches == 0, f"lag_1 igual a caja_neta en {lag_matches} filas (leakage)"

    # rolling_mean_7 no debe ser idéntico a caja_neta
    rm_matches = (df_feat["rolling_mean_7"] == df_feat["caja_neta"]).sum()
    assert rm_matches == 0, f"rolling_mean_7 igual a caja_neta en {rm_matches} filas"


def test_all_expected_columns_present() -> None:
    """Verifica que add_features genere todas las columnas esperadas."""
    df = generate_dataset(seed=42, periods=12)
    df_clean, _ = clean_dataset(df)
    df_feat = add_features(df_clean)

    expected = [
        "caja_neta",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_30",
        "dia_semana",
        "mes",
        "trimestre",
        "es_finde",
        "es_cierre_mes",
        "dia_del_mes",
        "dia_del_año",
    ]
    for col in expected:
        assert col in df_feat.columns, f"Falta la columna '{col}' en el output"


def test_rolling_windows_use_past_only() -> None:
    """Verifica que las rolling windows usen solo pasado y no tengan NaN posteriores."""
    df = generate_dataset(seed=42, periods=12)
    df_clean, _ = clean_dataset(df)
    df_feat = add_features(df_clean)

    # rolling_mean_7 no debe ser idéntico a caja_neta (derivado, no copia)
    assert not df_feat["rolling_mean_7"].equals(df_feat["caja_neta"]), (
        "rolling_mean_7 es idéntico a caja_neta"
    )

    # No debe haber NaN después de las primeras 30 filas
    assert df_feat.iloc[30:]["rolling_mean_7"].notna().all(), (
        "rolling_mean_7 tiene NaN después de la fila 30"
    )
    assert df_feat.iloc[30:]["rolling_std_30"].notna().all(), (
        "rolling_std_30 tiene NaN después de la fila 30"
    )
