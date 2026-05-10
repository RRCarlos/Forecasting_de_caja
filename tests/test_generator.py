"""Tests T01 — Generator (src.etl.generator)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.etl.generator import generate_dataset


def test_generates_correct_number_of_rows() -> None:
    """Verifica que el dataset tenga ~24*30 filas y tipos correctos."""
    df = generate_dataset(seed=42, periods=24)

    # ≈ 24 × 30.4 = 729 — entre 700 y 750
    assert 700 <= len(df) <= 750, f"Esperado ~729 filas, obtenido {len(df)}"

    # Columnas exactas
    expected_columns = [
        "fecha",
        "ingreso_efectivo",
        "ingreso_tarjeta",
        "ingreso_transferencia",
        "gasto_operativo",
        "gasto_extraordinario",
        "saldo_diario",
        "saldo_acumulado",
        "es_festivo",
        "es_finde",
        "tiene_anomalia",
        "tipo_anomalia",
    ]
    assert list(df.columns) == expected_columns, "Las columnas no coinciden"

    # Tipos de datos
    assert pd.api.types.is_datetime64_any_dtype(df["fecha"]), "fecha no es datetime64"
    assert pd.api.types.is_float_dtype(df["ingreso_efectivo"]), "ingreso_efectivo no es float"
    assert pd.api.types.is_bool_dtype(df["tiene_anomalia"]), "tiene_anomalia no es bool"

    # Tasa de anomalías ~5% (entre 3% y 7%)
    anomaly_rate = df["tiene_anomalia"].mean()
    assert 0.03 <= anomaly_rate <= 0.07, (
        f"Tasa de anomalías {anomaly_rate:.2%} fuera del rango [3%, 7%]"
    )


def test_reproducible_with_seed() -> None:
    """Verifica que la misma semilla produzca el mismo dataset."""
    df1 = generate_dataset(seed=42)
    df2 = generate_dataset(seed=42)
    assert_frame_equal(df1, df2)


def test_expected_anomaly_types() -> None:
    """Verifica que existan al menos 4 tipos de anomalía con ≥3 instancias c/u."""
    df = generate_dataset(seed=42, periods=24)
    anomalies = df[df["tiene_anomalia"]]
    tipos = anomalies["tipo_anomalia"].value_counts()

    assert len(tipos) >= 4, f"Se esperaban ≥4 tipos, se encontraron {len(tipos)}"
    assert (tipos >= 3).all(), (
        f"Hay tipos con menos de 3 instancias: {dict(tipos[tipos < 3])}"
    )


def test_date_range_continuous() -> None:
    """Verifica que las fechas sean contiguas sin huecos (>1 día)."""
    df = generate_dataset(seed=42, periods=24)
    diffs = df["fecha"].diff().dt.days.iloc[1:]  # primera diff es NaN
    assert (diffs == 1).all(), "Existen huecos en la serie de fechas"
