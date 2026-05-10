"""
Matriz de consenso para detección de anomalías.

Combina múltiples métodos de detección (Isolation Forest, Z-score, IQR,
Benford, Temporal) en un score ponderado único y clasifica la severidad.

Ponderación:
  - Isolation Forest: 0.30
  - Z-score:         0.15
  - IQR:             0.15
  - Benford:         0.15
  - Temporal:        0.25

Uso:
    from src.anomalies.consensus import build_consensus_matrix
    consenso = build_consensus_matrix(df, if_anom, z_anom, iqr_anom,
                                      benford_anom, temp_anom)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pesos de cada método en el consenso
_PESOS: Dict[str, float] = {
    "if": 0.30,
    "zscore": 0.15,
    "iqr": 0.15,
    "benford": 0.15,
    "temporal": 0.25,
}

# Umbrales de severidad
_SEVERIDAD: List[tuple] = [
    (0.70, "crítico"),
    (0.45, "alto"),
    (0.25, "medio"),
    (0.10, "bajo"),
]


def build_consensus_matrix(
    df: pd.DataFrame,
    if_anomalies: pd.Series,
    zscore_anomalies: pd.Series,
    iqr_anomalies: pd.Series,
    benford_anomalies: pd.Series,
    temporal_anomalies: pd.Series,
    if_scores: pd.Series | None = None,
) -> pd.DataFrame:
    """Construye la matriz de consenso a partir de múltiples detectores.

    Calcula un score ponderado para cada fila basado en cuántos métodos
    detectaron una anomalía y sus pesos respectivos. Clasifica la severidad
    según el score obtenido.

    Args:
        df: DataFrame original (se usa para extraer la columna 'fecha').
        if_anomalies: Serie booleana del detector Isolation Forest.
        zscore_anomalies: Serie booleana del detector Z-score.
        iqr_anomalies: Serie booleana del detector IQR.
        benford_anomalies: Serie booleana del detector Benford.
        temporal_anomalies: Serie booleana del detector Temporal (columna
            'es_anomalia_temporal' o booleana directa).
        if_scores: Serie opcional con scores numéricos del IF (para
            enriquecer el reporte).

    Returns:
        DataFrame con columnas:
            - fecha: fecha del registro (del df original)
            - consenso_score: score ponderado (0.0 a 1.0)
            - severidad: str (crítico, alto, medio, bajo) o None
            - metodos_detectores: lista de nombres de métodos que detectaron
            - if_detectado: bool
            - zscore_detectado: bool
            - iqr_detectado: bool
            - benford_detectado: bool
            - temporal_detectado: bool

    Raises:
        ValueError: Si las series tienen longitudes distintas.
    """
    n = len(df)
    _validar_series(
        n,
        if_anomalies=if_anomalies,
        zscore_anomalies=zscore_anomalies,
        iqr_anomalies=iqr_anomalies,
        benford_anomalies=benford_anomalies,
        temporal_anomalies=temporal_anomalies,
    )

    # Normalizar temporal_anomalies: si es DataFrame, extraer columna booleana
    if isinstance(temporal_anomalies, pd.DataFrame):
        if "es_anomalia_temporal" in temporal_anomalies.columns:
            temporal_bool = temporal_anomalies["es_anomalia_temporal"].astype(bool)
        else:
            temporal_bool = temporal_anomalies.iloc[:, 0].astype(bool)
    else:
        temporal_bool = temporal_anomalies.astype(bool)

    # Asegurar booleanos
    if_bool = if_anomalies.astype(bool)
    zscore_bool = zscore_anomalies.astype(bool)
    iqr_bool = iqr_anomalies.astype(bool)
    benford_bool = benford_anomalies.astype(bool)

    # Calcular score ponderado
    consenso_score = (
        if_bool.astype(float) * _PESOS["if"]
        + zscore_bool.astype(float) * _PESOS["zscore"]
        + iqr_bool.astype(float) * _PESOS["iqr"]
        + benford_bool.astype(float) * _PESOS["benford"]
        + temporal_bool.astype(float) * _PESOS["temporal"]
    )

    # Clasificar severidad
    severidad = consenso_score.apply(_clasificar_severidad)

    # Lista de métodos detectores por fila
    metodos_detectores = _listar_metodos(
        if_bool, zscore_bool, iqr_bool, benford_bool, temporal_bool
    )

    # Armar fecha
    if "fecha" in df.columns:
        fecha = pd.to_datetime(df["fecha"])
    else:
        fecha = pd.Series([None] * n, index=df.index)

    # Incluir caja_neta si existe en el df original (issue #2)
    caja_neta = df.get("caja_neta", pd.Series([float("nan")] * n, index=df.index))

    df_consenso = pd.DataFrame(
        {
            "fecha": fecha,
            "caja_neta": caja_neta,
            "consenso_score": consenso_score,
            "severidad": severidad,
            "metodos_detectores": metodos_detectores,
            "if_detectado": if_bool,
            "zscore_detectado": zscore_bool,
            "iqr_detectado": iqr_bool,
            "benford_detectado": benford_bool,
            "temporal_detectado": temporal_bool,
        },
        index=df.index,
    )

    # Resumen
    n_total = (severidad.notna()).sum()
    logger.info(
        "Consenso: %d anomalías detectadas (%.2f%% del total)",
        n_total,
        100.0 * n_total / n if n > 0 else 0.0,
    )
    for sev in ["bajo", "medio", "alto", "crítico"]:
        count = int((severidad == sev).sum())
        if count > 0:
            logger.info("  %s: %d", sev, count)

    return df_consenso


def _clasificar_severidad(score: float) -> str | None:
    """Clasifica la severidad según el score de consenso.

    Args:
        score: Score ponderado (0.0 a 1.0).

    Returns:
        'crítico', 'alto', 'medio', 'bajo' o None si no es anomalía.
    """
    for umbral, nivel in _SEVERIDAD:
        if score >= umbral:
            return nivel
    return None


def _listar_metodos(
    *detectores: pd.Series,
) -> pd.Series:
    """Crea lista de nombres de métodos que detectaron anomalía por fila.

    Args:
        detectores: Series booleanas de cada detector.

    Returns:
        Serie donde cada elemento es una lista de strings con los nombres
        de los métodos que detectaron la anomalía.
    """
    nombres = ["if", "zscore", "iqr", "benford", "temporal"]
    # Mapear a nombres más legibles
    nombres_legibles = {
        "if": "Isolation Forest",
        "zscore": "Z-score",
        "iqr": "IQR",
        "benford": "Benford",
        "temporal": "Temporal",
    }

    n = len(detectores[0])
    resultados: List[List[str]] = [[] for _ in range(n)]

    for detector, nombre in zip(detectores, nombres):
        for i in range(n):
            if detector.iloc[i]:
                resultados[i].append(nombres_legibles[nombre])

    return pd.Series(resultados, index=detectores[0].index)


def _validar_series(n: int, **series: pd.Series) -> None:
    """Valida que todas las series tengan la misma longitud.

    Args:
        n: Longitud esperada.
        series: Keyword args con nombre y serie a validar.

    Raises:
        ValueError: Si alguna serie no coincide en longitud.
    """
    for nombre, serie in series.items():
        if len(serie) != n:
            raise ValueError(
                f"Longitud de '{nombre}' ({len(serie)}) no coincide "
                f"con df ({n})"
            )
