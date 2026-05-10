"""
Detección de anomalías temporales en series de flujo de caja.

Detecta 4 tipos de irregularidades temporales:
1. Cambios abruptos día a día
2. Patrones atípicos en fin de semana
3. Outliers en ventana móvil (rolling)
4. Picos de gasto extraordinario

Uso:
    from src.anomalies.temporal import detect_temporal_anomalies
    df_temporal = detect_temporal_anomalies(df)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_temporal_anomalies(
    df: pd.DataFrame,
    date_column: str = "fecha",
    value_column: str = "caja_neta",
    suspicious_hours: Tuple[int, int] = (0, 6),
    weekend_anomaly_threshold: float = 2.0,
    change_anomaly_threshold: float = 3.0,
) -> pd.DataFrame:
    """Detecta anomalías temporales en la serie de flujo de caja.

    Aplica 4 métodos de detección sobre la serie temporal:

    1. **Cambios abruptos**: variación día a día en caja_neta que supera
       `change_anomaly_threshold` desviaciones estándar de los cambios.
    2. **Patrones de fin de semana**: si es fin de semana y el |caja_neta|
       supera `weekend_anomaly_threshold` veces el promedio de findes.
    3. **Outliers en ventana móvil**: |caja_neta - rolling_mean_30| >
       3 * rolling_std_30.
    4. **Picos de gasto**: gasto_extraordinario > 3 * su media histórica.

    Args:
        df: DataFrame con datos diarios. Debe contener las columnas
            especificadas y opcionalmente las rolling features calculadas
            en features.py.
        date_column: Nombre de la columna de fecha.
        value_column: Columna numérica principal para evaluar cambios.
        suspicious_hours: Tupla (hora_inicio, hora_fin) para el rango
            de horas sospechosas. Nota: el dataset es diario, este
            parámetro se incluye para compatibilidad pero no aplica
            directamente (se usa como metadata).
        weekend_anomaly_threshold: Multiplicador del promedio de findes
            para considerar un valor atípico.
        change_anomaly_threshold: Número de desviaciones estándar para
            considerar un cambio día a día como abrupto.

    Returns:
        DataFrame con las mismas filas que df y columnas:
            - fecha: la fecha del registro
            - es_anomalia_temporal: bool, True si alguna detección activó
            - tipo_anomalia_temporal: str con el/los tipo(s) detectados
            - score_temporal: float, proporción de métodos que detectaron
              (0.0 a 1.0)

    Raises:
        ValueError: Si faltan columnas requeridas en el DataFrame.
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se detectan anomalías temporales")
        return pd.DataFrame(
            columns=[
                "fecha",
                "es_anomalia_temporal",
                "tipo_anomalia_temporal",
                "score_temporal",
            ]
        )

    _validar_columnas_temporales(df, date_column, value_column)

    df_resultado = pd.DataFrame(index=df.index)
    df_resultado["fecha"] = pd.to_datetime(df[date_column])

    # Inicializar tipos detectados
    tipos: pd.DataFrame = pd.DataFrame(
        "", index=df.index, columns=["tipo"]
    )
    score: pd.DataFrame = pd.DataFrame(0.0, index=df.index, columns=["score"])
    n_metodos = 4  # Total de métodos aplicados

    # ── 1. Cambios abruptos día a día ───────────────────────────────────────
    cambios = df[value_column].diff().abs()
    std_cambios = cambios.std()

    if std_cambios > 1e-10 and len(cambios.dropna()) > 1:
        es_cambio_abrupto = cambios > change_anomaly_threshold * std_cambios
        es_cambio_abrupto = es_cambio_abrupto.fillna(False)

        n_cambios = es_cambio_abrupto.sum()
        if n_cambios > 0:
            score.loc[es_cambio_abrupto, "score"] += 1.0 / n_metodos
            _agregar_tipo(tipos, es_cambio_abrupto, "cambio_abrupto")
            logger.info(
                "Temporal - Cambios abruptos: %d detectados (umbral=%.2f)",
                n_cambios,
                change_anomaly_threshold * std_cambios,
            )
    else:
        logger.warning(
            "Temporal: std de cambios casi nula — saltando detección de cambios abruptos"
        )

    # ── 2. Patrones de fin de semana ────────────────────────────────────────
    col_finde = "es_finde"
    if col_finde in df.columns:
        es_finde = df[col_finde].astype(bool)
        if es_finde.any():
            valores_finde = df.loc[es_finde, value_column]
            media_finde = valores_finde.mean()
            std_finde = valores_finde.std()

            if std_finde > 1e-10:
                es_finde_atipico = es_finde & (
                    df[value_column].abs()
                    > abs(media_finde) * weekend_anomaly_threshold
                )
                es_finde_atipico = es_finde_atipico.fillna(False)

                n_finde = es_finde_atipico.sum()
                if n_finde > 0:
                    score.loc[es_finde_atipico, "score"] += 1.0 / n_metodos
                    _agregar_tipo(tipos, es_finde_atipico, "finde_atipico")
                    logger.info(
                        "Temporal - Findes atípicos: %d detectados (umbral=%.1fx media)",
                        n_finde,
                        weekend_anomaly_threshold,
                    )
    else:
        logger.info(
            "Temporal: columna '%s' no disponible — saltando detección de findes",
            col_finde,
        )

    # ── 3. Outliers en ventana móvil ────────────────────────────────────────
    col_mean = "rolling_mean_30"
    col_std = "rolling_std_30"

    if col_mean in df.columns and col_std in df.columns:
        rolling_mean = df[col_mean]
        rolling_std = df[col_std]

        # Evitar división por cero
        rolling_std_safe = rolling_std.replace(0, np.nan)
        z_rolling = (df[value_column] - rolling_mean).abs() / rolling_std_safe

        es_outlier_ventana = z_rolling > 3.0
        es_outlier_ventana = es_outlier_ventana.fillna(False)

        n_outlier = es_outlier_ventana.sum()
        if n_outlier > 0:
            score.loc[es_outlier_ventana, "score"] += 1.0 / n_metodos
            _agregar_tipo(tipos, es_outlier_ventana, "outlier_ventana")
            logger.info(
                "Temporal - Outliers ventana móvil: %d detectados", n_outlier
            )
    else:
        logger.warning(
            "Temporal: columnas '%s' o '%s' no disponibles — "
            "saltando detección de outliers ventana",
            col_mean,
            col_std,
        )

    # ── 4. Picos de gasto extraordinario ────────────────────────────────────
    col_gasto_extra = "gasto_extraordinario"

    if col_gasto_extra in df.columns:
        gasto_extra = df[col_gasto_extra]
        # Solo considerar días con gasto extraordinario > 0
        tiene_gasto = gasto_extra > 0
        if tiene_gasto.any():
            media_hist = gasto_extra[gasto_extra > 0].mean()

            if media_hist > 1e-10:
                es_pico_gasto = gasto_extra > 3.0 * media_hist
                es_pico_gasto = es_pico_gasto.fillna(False)

                n_pico = es_pico_gasto.sum()
                if n_pico > 0:
                    score.loc[es_pico_gasto, "score"] += 1.0 / n_metodos
                    _agregar_tipo(tipos, es_pico_gasto, "pico_gasto")
                    logger.info(
                        "Temporal - Picos de gasto: %d detectados (umbral=%.2f)",
                        n_pico,
                        3.0 * media_hist,
                    )
    else:
        logger.info(
            "Temporal: columna '%s' no disponible — saltando detección de picos",
            col_gasto_extra,
        )

    # ── Armar resultado ─────────────────────────────────────────────────────
    df_resultado["es_anomalia_temporal"] = score["score"] > 0.0
    df_resultado["tipo_anomalia_temporal"] = tipos["tipo"]
    df_resultado["score_temporal"] = score["score"]

    n_total = df_resultado["es_anomalia_temporal"].sum()
    logger.info(
        "Temporal: %d anomalías temporales detectadas (%.2f%% del total)",
        n_total,
        100.0 * n_total / len(df) if len(df) > 0 else 0.0,
    )

    return df_resultado


def _agregar_tipo(
    tipos: pd.DataFrame,
    mask: pd.Series,
    tipo: str,
) -> None:
    """Agrega un tipo de anomalía a las filas indicadas por la máscara.

    Si la fila ya tiene un tipo asignado, concatena con ' + '.

    Args:
        tipos: DataFrame con columna 'tipo' que se modifica in-place.
        mask: Máscara booleana de las filas a modificar.
        tipo: Nombre del tipo de anomalía a agregar.
    """
    tipos.loc[mask, "tipo"] = tipos.loc[mask, "tipo"].apply(
        lambda x: f"{x} + {tipo}" if x else tipo
    )


def _validar_columnas_temporales(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> None:
    """Valida que existan las columnas mínimas requeridas.

    Args:
        df: DataFrame a validar.
        date_column: Nombre de la columna de fecha.
        value_column: Nombre de la columna de valor.

    Raises:
        ValueError: Si faltan columnas esenciales.
    """
    required = [date_column, value_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas requeridas faltantes en detect_temporal_anomalies: {missing}"
        )
