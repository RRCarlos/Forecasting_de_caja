"""Tests T02 — Cleaner (src.etl.cleaner)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.etl.cleaner import COLUMNAS_REQUERIDAS, clean_dataset
from src.etl.generator import generate_dataset


def test_removes_duplicates() -> None:
    """Verifica que se eliminen duplicados exactos y se reporten."""
    df = generate_dataset(seed=42, periods=6)

    # Insertar duplicados intencionales de las primeras 5 filas
    dupes = pd.concat([df, df.iloc[:5]], ignore_index=True)

    df_clean, report = clean_dataset(dupes)

    cols_sin_fecha = [c for c in COLUMNAS_REQUERIDAS if c != "fecha"]
    assert df_clean.duplicated(subset=cols_sin_fecha).sum() == 0, (
        "Quedaron duplicados después de clean_dataset"
    )
    assert report["filas_duplicadas_eliminadas"] > 0, (
        "No se reportaron duplicados eliminados"
    )


def test_returns_cleaning_report() -> None:
    """Verifica que el reporte de limpieza contenga todas las keys esperadas."""
    df = generate_dataset(seed=42, periods=6)
    df_clean, report = clean_dataset(df)

    required_keys = [
        "total_filas",
        "filas_duplicadas_eliminadas",
        "nulos_imputados",
        "esquema_validado",
        "huecos_fecha",
        "fecha_inicio",
        "fecha_fin",
        "nulos_por_columna",
        "valores_negativos",
        "dias_sin_operacion",
        "valores_extremos_marcados",
    ]
    for key in required_keys:
        assert key in report, f"Falta la clave '{key}' en el reporte"

    assert report["total_filas"] > 0, "total_filas debe ser > 0"
    assert report["esquema_validado"] is True, "El esquema debería validarse correctamente"


def test_schema_validation() -> None:
    """Verifica que clean_dataset rechace un DataFrame con columna faltante."""
    df = generate_dataset(seed=42, periods=6)
    df_bad = df.drop(columns=["saldo_acumulado"])

    with pytest.raises(ValueError, match="Esquema inválido"):
        clean_dataset(df_bad)
