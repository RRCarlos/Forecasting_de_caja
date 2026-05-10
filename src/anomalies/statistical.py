"""
Detección de anomalías mediante métodos estadísticos univariados.

Implementa dos métodos clásicos:
- Z-score: detecta valores cuya distancia a la media supera N desviaciones.
- IQR (Rango Intercuartílico): detecta valores fuera del rango
  [Q1 - k*IQR, Q3 + k*IQR].

Uso:
    from src.anomalies.statistical import detect_zscore, detect_iqr
    z_anom = detect_zscore(df)
    iqr_anom = detect_iqr(df)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_zscore(
    df: pd.DataFrame,
    column: str = "caja_neta",
    threshold: float = 3.0,
) -> pd.Series:
    """Detecta anomalías usando el método Z-score.

    Calcula el Z-score de cada valor: z = (x - μ) / σ.
    Si |z| > threshold, el valor se marca como anomalía.

    Args:
        df: DataFrame que contiene la columna a analizar.
        column: Nombre de la columna numérica a evaluar.
        threshold: Umbral de Z-score. Por defecto 3.0 (99.7% en distribución
            normal).

    Returns:
        Serie booleana con mismo índice que df: True = anomalía detectada.

    Raises:
        ValueError: Si el DataFrame está vacío o la columna no existe.
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se puede calcular Z-score")
        return pd.Series([], dtype=bool, name="zscore_anomalia")

    if column not in df.columns:
        raise ValueError(f"Columna '{column}' no encontrada en el DataFrame")

    valores = df[column].dropna()

    if len(valores) < 2:
        logger.warning(
            "Z-score: insuficientes valores no nulos (%d) para calcular std",
            len(valores),
        )
        return pd.Series(False, index=df.index, name="zscore_anomalia", dtype=bool)

    media = valores.mean()
    std = valores.std()

    if std < 1e-10:
        logger.warning(
            "Z-score: desviación estándar casi nula (std=%e) — "
            "no se pueden detectar anomalías",
            std,
        )
        return pd.Series(False, index=df.index, name="zscore_anomalia", dtype=bool)

    # Calcular z-score solo donde hay valores no nulos
    z_scores = (df[column] - media) / std

    resultado = z_scores.abs() > threshold

    # Donde hay NaN en la columna original, no marcar anomalía
    resultado = resultado.fillna(False).astype(bool)
    resultado.name = "zscore_anomalia"

    n_anomalias = resultado.sum()
    logger.info(
        "Z-score (|z| > %.1f): %d anomalías detectadas en '%s'",
        threshold,
        n_anomalias,
        column,
    )

    return resultado


def detect_iqr(
    df: pd.DataFrame,
    column: str = "caja_neta",
    multiplier: float = 1.5,
) -> pd.Series:
    """Detecta anomalías usando el método de Rango Intercuartílico (IQR).

    Calcula Q1 (percentil 25) y Q3 (percentil 75), luego define:
    - Límite inferior = Q1 - multiplier * IQR
    - Límite superior = Q3 + multiplier * IQR
    Valores fuera de estos límites se marcan como anomalías.

    Args:
        df: DataFrame que contiene la columna a analizar.
        column: Nombre de la columna numérica a evaluar.
        multiplier: Factor multiplicador del IQR. Por defecto 1.5
            (criterio estándar de Tukey). Usar 3.0 para outliers extremos.

    Returns:
        Serie booleana con mismo índice que df: True = anomalía detectada.

    Raises:
        ValueError: Si el DataFrame está vacío o la columna no existe.
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se puede calcular IQR")
        return pd.Series([], dtype=bool, name="iqr_anomalia")

    if column not in df.columns:
        raise ValueError(f"Columna '{column}' no encontrada en el DataFrame")

    valores = df[column].dropna()

    if len(valores) < 4:
        logger.warning(
            "IQR: insuficientes valores no nulos (%d) para cuartiles",
            len(valores),
        )
        return pd.Series(False, index=df.index, name="iqr_anomalia", dtype=bool)

    Q1 = valores.quantile(0.25)
    Q3 = valores.quantile(0.75)
    IQR = Q3 - Q1

    if IQR < 1e-10:
        logger.warning(
            "IQR casi nulo (IQR=%e) — posiblemente datos constantes",
            IQR,
        )
        return pd.Series(False, index=df.index, name="iqr_anomalia", dtype=bool)

    limite_inferior = Q1 - multiplier * IQR
    limite_superior = Q3 + multiplier * IQR

    resultado = (df[column] < limite_inferior) | (df[column] > limite_superior)
    resultado = resultado.fillna(False).astype(bool)
    resultado.name = "iqr_anomalia"

    n_anomalias = resultado.sum()
    logger.info(
        "IQR [%.2f, %.2f] (mult=%.1f): %d anomalías detectadas en '%s'",
        limite_inferior,
        limite_superior,
        multiplier,
        n_anomalias,
        column,
    )

    return resultado
