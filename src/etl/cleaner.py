"""
Limpieza y validación del dataset de flujo de caja.

Valida esquema, tipos, contigüidad de fechas, imputa nulos, elimina
duplicados exactos, y genera un reporte detallado de calidad.

Uso:
    from src.etl.cleaner import clean_dataset
    df_clean, report = clean_dataset(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas requeridas y sus tipos esperados
COLUMNAS_REQUERIDAS: List[str] = [
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

COLUMNAS_MONTO: List[str] = [
    "ingreso_efectivo",
    "ingreso_tarjeta",
    "ingreso_transferencia",
    "gasto_operativo",
    "gasto_extraordinario",
    "saldo_diario",
    "saldo_acumulado",
]

COLUMNAS_MONTO_NO_NEGATIVAS: List[str] = [
    "ingreso_efectivo",
    "ingreso_tarjeta",
    "ingreso_transferencia",
    "gasto_operativo",
    "gasto_extraordinario",
    "saldo_acumulado",
]

COLUMNAS_BOOL: List[str] = [
    "es_festivo",
    "es_finde",
    "tiene_anomalia",
]

UMBRAL_DIA_SIN_OPERACION: float = 1_000.0
"""Ingreso total por debajo de este valor se considera 'día sin operación'."""


def _validar_esquema(df: pd.DataFrame) -> Dict[str, Any]:
    """Valida que el DataFrame tenga las columnas esperadas y tipos correctos.

    Args:
        df: DataFrame a validar.

    Returns:
        Dict con 'ok' (bool), 'faltantes' (list), 'tipos_incorrectos' (list).
    """
    resultado: Dict[str, Any] = {
        "ok": True,
        "faltantes": [],
        "tipos_incorrectos": [],
    }

    # Verificar columnas faltantes
    for col in COLUMNAS_REQUERIDAS:
        if col not in df.columns:
            resultado["faltantes"].append(col)
            resultado["ok"] = False

    if resultado["faltantes"]:
        logger.warning("Columnas faltantes: %s", resultado["faltantes"])

    # Verificar tipos de columnas monto
    for col in COLUMNAS_MONTO:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            resultado["tipos_incorrectos"].append(
                f"{col} (esperado numérico, got {df[col].dtype})"
            )
            resultado["ok"] = False

    # Verificar columna fecha
    if "fecha" in df.columns:
        try:
            pd.to_datetime(df["fecha"])
        except (ValueError, TypeError):
            resultado["tipos_incorrectos"].append(
                "fecha (no convertible a datetime)"
            )
            resultado["ok"] = False

    return resultado


def _reportar_nulos(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Cuenta valores nulos por columna y devuelve reporte.

    Args:
        df: DataFrame a inspeccionar.

    Returns:
        Dict anidado: {columna: {'count': int, 'porcentaje': float}}.
    """
    nulos: Dict[str, Dict[str, Any]] = {}
    total = len(df)
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            nulos[col] = {
                "count": n,
                "porcentaje": round(100.0 * n / total, 2),
            }
    return nulos


def _imputar_nulos(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Imputa valores nulos en el DataFrame.

    - Saldos: interpolación lineal.
    - Montos de ingresos/gastos: relleno con 0.
    - Columnas booleanas: relleno con False.
    - tipo_anomalia: relleno con cadena vacía (no aplica).

    Args:
        df: DataFrame con posibles nulos.

    Returns:
        (DataFrame sin nulos, dict con conteo de imputaciones por columna).
    """
    imputadas: Dict[str, int] = {}
    df_clean = df.copy()

    # Saldos — interpolación lineal
    for col in ["saldo_diario", "saldo_acumulado"]:
        n = int(df_clean[col].isna().sum())
        if n > 0:
            df_clean[col] = df_clean[col].interpolate(method="linear")
            # Si aún quedan nulos al inicio, rellenar con el primer valor válido
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].bfill()
            imputadas[col] = n
            logger.info("  %s: %d nulos imputados por interpolación", col, n)

    # Montos — relleno con 0
    for col in COLUMNAS_MONTO:
        if col in imputadas:
            continue  # ya procesado
        n = int(df_clean[col].isna().sum())
        if n > 0:
            df_clean[col] = df_clean[col].fillna(0.0)
            imputadas[col] = n
            logger.info("  %s: %d nulos imputados con 0", col, n)

    # Booleanas
    for col in COLUMNAS_BOOL:
        if col in df_clean.columns:
            n = int(df_clean[col].isna().sum())
            if n > 0:
                df_clean[col] = df_clean[col].fillna(False)
                imputadas[col] = n

    # tipo_anomalia (object/string)
    if "tipo_anomalia" in df_clean.columns:
        n = int(df_clean["tipo_anomalia"].isna().sum())
        if n > 0:
            df_clean["tipo_anomalia"] = df_clean["tipo_anomalia"].fillna("")
            imputadas["tipo_anomalia"] = n

    return df_clean, imputadas


def _detectar_dias_sin_operacion(df: pd.DataFrame) -> int:
    """Cuenta días donde el ingreso total es menor al umbral definido.

    Args:
        df: DataFrame con columnas de ingresos.

    Returns:
        Número de días sin operación detectados.
    """
    ingreso_total = (
        df["ingreso_efectivo"]
        + df["ingreso_tarjeta"]
        + df["ingreso_transferencia"]
    )
    return int((ingreso_total < UMBRAL_DIA_SIN_OPERACION).sum())


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Limpia y valida el dataset de flujo de caja.

    Pipeline:
        1. Validar esquema (columnas, tipos, fechas).
        2. Verificar fechas contiguas sin huecos > 1 día.
        3. Reportar e imputar valores nulos.
        4. Eliminar duplicados exactos.
        5. Verificar montos no negativos (salvo saldo_diario).
        6. Detectar días sin operación.
        7. Marcar valores extremos (> 5σ) en el reporte (sin eliminar).

    Args:
        df: DataFrame crudo con los datos generados.

    Returns:
        Tupla (DataFrame limpio, dict con reporte de limpieza).

    Raises:
        ValueError: Si faltan columnas requeridas o el esquema es inválido.
    """
    logger.info("Iniciando limpieza del dataset (%d filas)", len(df))
    df_clean = df.copy()
    report: Dict[str, Any] = {}

    # ── 1. Validar esquema ──────────────────────────────────────────────────
    esquema = _validar_esquema(df_clean)
    if not esquema["ok"]:
        raise ValueError(
            f"Esquema inválido. Faltantes: {esquema['faltantes']}. "
            f"Tipos incorrectos: {esquema['tipos_incorrectos']}"
        )
    report["esquema_validado"] = True
    logger.info("Esquema validado correctamente.")

    # Asegurar tipo datetime en fecha
    df_clean["fecha"] = pd.to_datetime(df_clean["fecha"])
    df_clean = df_clean.sort_values("fecha").reset_index(drop=True)

    # ── 2. Contigüidad de fechas ────────────────────────────────────────────
    date_diff = df_clean["fecha"].diff().dt.days
    gaps = int((date_diff > 1).sum())
    if gaps > 0:
        logger.warning("Se detectaron %d huecos en la serie temporal", gaps)
    report["huecos_fecha"] = gaps

    # ── 3. Fecha inicio / fin ───────────────────────────────────────────────
    report["fecha_inicio"] = str(df_clean["fecha"].min().date())
    report["fecha_fin"] = str(df_clean["fecha"].max().date())
    report["total_filas"] = len(df_clean)

    # ── 4. Reporte de nulos por columna ─────────────────────────────────────
    nulos_por_columna = _reportar_nulos(df_clean)
    report["nulos_por_columna"] = nulos_por_columna

    nulos_imputados: Dict[str, int] = {}
    if nulos_por_columna:
        logger.info("Nulos detectados: %s", nulos_por_columna)
        df_clean, nulos_imputados = _imputar_nulos(df_clean)
    report["nulos_imputados"] = nulos_imputados

    # ── 5. Duplicados exactos ───────────────────────────────────────────────
    cols_sin_fecha = [c for c in COLUMNAS_REQUERIDAS if c != "fecha"]
    duplicados_mask = df_clean.duplicated(subset=cols_sin_fecha, keep="first")
    n_duplicados = int(duplicados_mask.sum())
    if n_duplicados > 0:
        df_clean = df_clean[~duplicados_mask].reset_index(drop=True)
        logger.info("Eliminados %d duplicados exactos", n_duplicados)
    report["filas_duplicadas_eliminadas"] = n_duplicados

    # ── 6. Montos no negativos ──────────────────────────────────────────────
    negativos_report: Dict[str, int] = {}
    for col in COLUMNAS_MONTO_NO_NEGATIVAS:
        if col in df_clean.columns:
            n_neg = int((df_clean[col] < 0).sum())
            if n_neg > 0:
                negativos_report[col] = n_neg
                logger.warning("%s tiene %d valores negativos", col, n_neg)
    report["valores_negativos"] = negativos_report

    # ── 7. Días sin operación ───────────────────────────────────────────────
    n_dias_sin_operacion = _detectar_dias_sin_operacion(df_clean)
    report["dias_sin_operacion"] = n_dias_sin_operacion
    if n_dias_sin_operacion > 0:
        logger.info("Detectados %d días sin operación", n_dias_sin_operacion)

    # ── 8. Valores extremos (solo reporte, no se eliminan) ──────────────────
    extremos_report: Dict[str, int] = {}
    for col in COLUMNAS_MONTO:
        if col in df_clean.columns:
            mean_v = df_clean[col].mean()
            std_v = df_clean[col].std() or 1.0
            n_ext = int((np.abs(df_clean[col] - mean_v) > 5 * std_v).sum())
            if n_ext > 0:
                extremos_report[col] = n_ext
    report["valores_extremos_marcados"] = extremos_report

    # Volver a calcular total_filas tras limpieza
    report["total_filas"] = len(df_clean)

    # ── Persistir ───────────────────────────────────────────────────────────
    out_path = "data/curated/dataset_clean.csv"
    df_clean.to_csv(out_path, index=False)
    logger.info(
        "Dataset limpio guardado en %s (%d filas, %d nulos imputados, "
        "%d duplicados eliminados)",
        out_path,
        len(df_clean),
        sum(nulos_imputados.values()),
        n_duplicados,
    )

    return df_clean, report
