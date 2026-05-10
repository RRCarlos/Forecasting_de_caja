"""
Detección de anomalías mediante Isolation Forest (sklearn).

Aísla observaciones anómalas usando la estrategia de particionamiento
aleatorio del algoritmo Isolation Forest. Útil para detectar outliers
multivariados sin necesidad de etiquetas.

Uso:
    from src.anomalies.isolation_forest import detect_isolation_forest
    anomalias = detect_isolation_forest(df)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Features por defecto para la detección multivariada
_FEATURES_DEFAULT: List[str] = [
    "caja_neta",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
]


def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float | str = "auto",
    n_estimators: int = 100,
    random_state: int = 42,
    feature_columns: List[str] | None = None,
) -> pd.Series:
    """Detecta anomalías usando Isolation Forest.

    Entrena un modelo Isolation Forest sobre las features seleccionadas
    y retorna una máscara booleana donde True indica anomalía.

    Args:
        df: DataFrame con las features para la detección.
        contamination: Proporción esperada de anomalías en el dataset.
            Si es 'auto', se estima automáticamente.
        n_estimators: Número de árboles en el ensemble.
        random_state: Semilla para reproducibilidad.
        feature_columns: Lista de columnas a usar como features.
            Por defecto usa: caja_neta, lag_1, lag_7, rolling_mean_7,
            rolling_std_7, rolling_mean_30.

    Returns:
        Serie booleana con mismo índice que df: True = anomalía detectada.

    Raises:
        ValueError: Si el DataFrame está vacío o no contiene las features
            requeridas.
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se pueden detectar anomalías")
        return pd.Series([], dtype=bool, name="if_anomalia")

    cols = feature_columns or _FEATURES_DEFAULT
    _validar_columnas(df, cols)

    X = df[cols].copy()

    # Eliminar filas con NaN para el entrenamiento
    mask_valid = ~X.isna().any(axis=1)
    n_drop = (~mask_valid).sum()
    if n_drop > 0:
        logger.info(
            "Isolation Forest: %d filas con NaN excluidas del entrenamiento",
            n_drop,
        )

    if mask_valid.sum() == 0:
        logger.warning("Isolation Forest: no hay filas válidas (sin NaN)")
        return pd.Series([False] * len(df), index=df.index, name="if_anomalia")

    logger.info(
        "Entrenando Isolation Forest (%d filas, %d features, contamination=%s)",
        mask_valid.sum(),
        X.shape[1],
        contamination,
    )

    modelo = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    # Predict: -1 = anomalía, 1 = normal
    predicciones = modelo.fit_predict(X.loc[mask_valid])

    resultado = pd.Series(False, index=df.index, name="if_anomalia", dtype=bool)
    resultado.loc[mask_valid] = predicciones == -1

    n_anomalias = resultado.sum()
    logger.info(
        "Isolation Forest: %d anomalías detectadas (%.2f%% de filas válidas)",
        n_anomalias,
        100.0 * n_anomalias / mask_valid.sum() if mask_valid.sum() > 0 else 0.0,
    )

    return resultado


def get_if_score(
    df: pd.DataFrame,
    contamination: float | str = "auto",
    n_estimators: int = 100,
    random_state: int = 42,
    feature_columns: List[str] | None = None,
) -> pd.Series:
    """Calcula el anomaly score del Isolation Forest normalizado a [0, 1].

    sklearn devuelve scores en [-1, 1] donde:
    - Más cercano a -1 → más anómalo
    - Más cercano a +1 → más normal

    Esta función normaliza a [0, 1] donde:
    - 1 = máxima anomalía
    - 0 = mínima anomalía (totalmente normal)

    Args:
        df: DataFrame con las features para la detección.
        contamination: Proporción esperada de anomalías.
        n_estimators: Número de árboles en el ensemble.
        random_state: Semilla para reproducibilidad.
        feature_columns: Lista de columnas a usar como features.

    Returns:
        Serie con scores en [0, 1] (1 = más anómalo), mismo índice que df.
        Las filas con NaN en features reciben score = 0.0 (normales por omisión).

    Raises:
        ValueError: Si el DataFrame está vacío o faltan columnas.
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se pueden calcular scores")
        return pd.Series([], dtype=float, name="if_score")

    cols = feature_columns or _FEATURES_DEFAULT
    _validar_columnas(df, cols)

    X = df[cols].copy()
    mask_valid = ~X.isna().any(axis=1)

    if mask_valid.sum() == 0:
        return pd.Series(0.0, index=df.index, name="if_score")

    modelo = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    modelo.fit(X.loc[mask_valid])

    # score_samples: más bajo = más anómalo (en sklearn, negativo = anomalía)
    scores_raw = modelo.score_samples(X.loc[mask_valid])

    # Normalizar de [-1, 1] a [0, 1]
    # score_samples en IsolationForest: a mayor valor, más normal.
    # Queremos: 1 = muy anómalo, 0 = muy normal.
    # Típicamente range: ~[-0.5, 0] para anomalías, ~[0, 0.5] para normales
    # Normalizamos invirtiendo: score_norm = 1 - (score - min) / (max - min)
    s_min = scores_raw.min()
    s_max = scores_raw.max()

    if s_max - s_min < 1e-10:
        scores_norm = np.zeros_like(scores_raw)
    else:
        scores_norm = 1.0 - (scores_raw - s_min) / (s_max - s_min)
        scores_norm = np.clip(scores_norm, 0.0, 1.0)

    resultado = pd.Series(0.0, index=df.index, name="if_score", dtype=float)
    resultado.loc[mask_valid] = scores_norm

    return resultado


def _validar_columnas(df: pd.DataFrame, cols: List[str]) -> None:
    """Valida que las columnas requeridas existan en el DataFrame.

    Args:
        df: DataFrame a validar.
        cols: Lista de columnas requeridas.

    Raises:
        ValueError: Si alguna columna no está presente.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas requeridas faltantes en Isolation Forest: {missing}"
        )
