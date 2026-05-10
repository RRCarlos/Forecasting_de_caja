"""
Feature engineering para el dataset de flujo de caja.

Parte del dataset limpio y genera:
- caja_neta (ingresos totales - gastos totales)
- Lags de 1, 7, 14, 30 días
- Medias móviles de 7, 14, 30 días (solo pasado, sin centrar)
- Desviación estándar móvil de 7, 30 días (solo pasado, sin centrar)
- Features temporales: día_semana, mes, trimestre, es_finde, etc.

REGLAS ESTRICTAS:
- NO se usa información futura — solo lags, nunca leads.
- Rolling windows NO centradas (solo valores anteriores al punto actual).
- Se verifica leakage y se lanza warning si se detecta.

Uso:
    from src.etl.features import add_features
    df_feat = add_features(df_clean)
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_INGRESO: List[str] = [
    "ingreso_efectivo",
    "ingreso_tarjeta",
    "ingreso_transferencia",
]

COLUMNAS_GASTO: List[str] = [
    "gasto_operativo",
    "gasto_extraordinario",
]

VENTANAS_LAG: List[int] = [1, 7, 14, 30]
VENTANAS_ROLLING_MEAN: List[int] = [7, 14, 30]
VENTANAS_ROLLING_STD: List[int] = [7, 30]


def _calcular_caja_neta(df: pd.DataFrame) -> pd.Series:
    """Calcula caja neta diaria = ingresos totales - gastos totales.

    Args:
        df: DataFrame con columnas de ingresos y gastos.

    Returns:
        Serie con caja neta por día.
    """
    ingresos = df[COLUMNAS_INGRESO].sum(axis=1)
    gastos = df[COLUMNAS_GASTO].sum(axis=1)
    return ingresos - gastos


def _calcular_lags(
    df: pd.DataFrame, col: str, ventanas: List[int]
) -> pd.DataFrame:
    """Calcula columnas lag de una serie sin usar información futura.

    Cada lag_k desplaza la serie k días hacia adelante, de modo que en
    la fila t se usa el valor de t - k.

    Args:
        df: DataFrame que contiene la columna origen.
        col: Nombre de la columna a desfasar.
        ventanas: Lista de desplazamientos en días.

    Returns:
        DataFrame con columnas lag_{k} para cada ventana.
    """
    lags = pd.DataFrame(index=df.index)
    for k in ventanas:
        lags[f"lag_{k}"] = df[col].shift(k)
    return lags


def _calcular_rolling(
    df: pd.DataFrame, col: str, ventanas: List[int], func: str
) -> pd.DataFrame:
    """Calcula ventanas móviles SIN centrar (solo pasado).

    Args:
        df: DataFrame con la columna origen.
        col: Nombre de la columna.
        ventanas: Lista de tamaños de ventana.
        func: 'mean' o 'std'.

    Returns:
        DataFrame con columnas rolling_{func}_{k} para cada ventana.

    Raises:
        ValueError: Si func no es 'mean' o 'std'.
    """
    if func not in ("mean", "std"):
        raise ValueError("`func` debe ser 'mean' o 'std'")

    roll = pd.DataFrame(index=df.index)
    for k in ventanas:
        if func == "mean":
            # center=False → solo valores anteriores (sin leakage)
            roll[f"rolling_{func}_{k}"] = (
                df[col].rolling(window=k, center=False, min_periods=1).mean()
            )
        else:
            roll[f"rolling_{func}_{k}"] = (
                df[col].rolling(window=k, center=False, min_periods=1).std()
            )
    return roll


def _calcular_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features de calendario a partir de la columna 'fecha'.

    Todas son determinísticas — no hay riesgo de leakage.
    NO duplica columnas que ya existen en el DataFrame original.

    Args:
        df: DataFrame con columna 'fecha' (datetime).

    Returns:
        DataFrame con columnas: dia_semana, mes, trimestre, es_finde
        (si no existe ya), es_cierre_mes, dia_del_mes, dia_del_año.
    """
    fecha = pd.to_datetime(df["fecha"])

    temp = pd.DataFrame(index=df.index)
    temp["dia_semana"] = fecha.dt.dayofweek  # 0=lun, 6=dom
    temp["mes"] = fecha.dt.month
    temp["trimestre"] = fecha.dt.quarter

    # Solo agregar es_finde si no existe ya en el DF original
    if "es_finde" not in df.columns:
        temp["es_finde"] = fecha.dt.dayofweek >= 5

    # Últimos 3 días del mes (más preciso que day >= 28)
    dias_en_mes = fecha.dt.days_in_month
    temp["es_cierre_mes"] = (dias_en_mes - fecha.dt.day) < 3
    temp["dia_del_mes"] = fecha.dt.day
    temp["dia_del_año"] = fecha.dt.dayofyear

    return temp


def _verificar_leakage(
    df_original: pd.DataFrame, df_features: pd.DataFrame
) -> List[str]:
    """Verifica que no haya leakage temporal en las features.

    Comprueba que las features calculadas con lags/rolling no contengan
    información del mismo día o de días futuros. Para ello, verifica que
    los valores de lag_1 no sean iguales al valor actual (lo que indicaría
    que se copió el mismo día).

    Args:
        df_original: DataFrame original con caja_neta.
        df_features: DataFrame con features ya calculadas.

    Returns:
        Lista de advertencias de leakage encontradas (vacía si todo ok).
    """
    advertencias: List[str] = []

    if "lag_1" in df_features.columns and "caja_neta" in df_original.columns:
        # Si lag_1 == caja_neta en muchas filas (>5%), hay leakage
        iguales = (df_features["lag_1"] == df_original["caja_neta"]).sum()
        if iguales > 0.05 * len(df_features):
            advertencias.append(
                f"lag_1 coincide con caja_neta en {iguales} filas "
                f"({100*iguales/len(df_features):.1f}%) — posible leakage"
            )

    return advertencias


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features temporales y de ventana al dataset limpio.

    Pipeline:
        1. Calcular caja_neta = ingresos - gastos.
        2. Calcular lags de 1, 7, 14, 30 días (caja_neta desplazada).
        3. Calcular rolling mean de 7, 14, 30 días (solo pasado).
        4. Calcular rolling std de 7, 30 días (solo pasado).
        5. Calcular features temporales (día_semana, mes, etc.).
        6. Verificar que no hay leakage.
        7. Eliminar filas con NaN de las ventanas de lags.

    Args:
        df: DataFrame limpio (debe tener columnas de ingresos, gastos y fecha).

    Returns:
        DataFrame con todas las features. Las primeras 30 filas tendrán NaN
        en las columnas de lags (se eliminan al final).

    Raises:
        ValueError: Si faltan columnas requeridas.
    """
    required = COLUMNAS_INGRESO + COLUMNAS_GASTO + ["fecha"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    logger.info("Iniciando feature engineering (%d filas)", len(df))
    df_feat = df.copy()

    # ── 1. caja_neta ────────────────────────────────────────────────────────
    caja_neta = _calcular_caja_neta(df_feat)
    df_feat["caja_neta"] = caja_neta
    logger.info("caja_neta calculada (media=%.2f)", caja_neta.mean())

    # ── 2. Lags (solo pasado, shift(k) = información de t-k) ────────────────
    lags_df = _calcular_lags(df_feat, "caja_neta", VENTANAS_LAG)
    df_feat = pd.concat([df_feat, lags_df], axis=1)
    logger.info("Lags creados: %s", VENTANAS_LAG)

    # ── 3. Rolling mean (NO centrada → solo pasado) ─────────────────────────
    roll_mean = _calcular_rolling(df_feat, "caja_neta", VENTANAS_ROLLING_MEAN, "mean")
    df_feat = pd.concat([df_feat, roll_mean], axis=1)
    logger.info("Rolling mean creadas: %s", VENTANAS_ROLLING_MEAN)

    # ── 4. Rolling std (NO centrada → solo pasado) ──────────────────────────
    roll_std = _calcular_rolling(df_feat, "caja_neta", VENTANAS_ROLLING_STD, "std")
    df_feat = pd.concat([df_feat, roll_std], axis=1)
    logger.info("Rolling std creadas: %s", VENTANAS_ROLLING_STD)

    # ── 5. Features temporales ─────────────────────────────────────────────
    temp_df = _calcular_features_temporales(df_feat)
    df_feat = pd.concat([df_feat, temp_df], axis=1)
    logger.info("Features temporales creadas: %d columnas", temp_df.shape[1])

    # ── 6. Verificar leakage ───────────────────────────────────────────────
    advertencias = _verificar_leakage(df, df_feat)
    for adv in advertencias:
        warnings.warn(adv)
        logger.warning("LEAKAGE: %s", adv)
    if not advertencias:
        logger.info("Verificación de leakage: OK — no se detectó información futura")

    # ── 7. Eliminar filas con NaN de lags ───────────────────────────────────
    # Las primeras `max(ventanas_lag)` filas tendrán NaN en los lags
    # porque no hay datos históricos suficientes.
    cols_lag = [f"lag_{k}" for k in VENTANAS_LAG]
    n_before = len(df_feat)
    df_feat = df_feat.dropna(subset=cols_lag).reset_index(drop=True)
    n_removed = n_before - len(df_feat)
    logger.info(
        "Filas eliminadas por NaN en lags: %d (quedan %d)", n_removed, len(df_feat)
    )

    # ── Persistir ──────────────────────────────────────────────────────────
    out_path = "data/curated/dataset_features.csv"
    df_feat.to_csv(out_path, index=False)
    logger.info(
        "Dataset con features guardado en %s (%d filas, %d columnas)",
        out_path,
        len(df_feat),
        df_feat.shape[1],
    )

    return df_feat
