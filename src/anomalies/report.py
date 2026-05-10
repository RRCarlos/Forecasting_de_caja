"""
Generación de reportes de anomalías en formato CSV y resumen textual.

Toma la matriz de consenso y el DataFrame original para producir:
- anomaly_report.csv — resumen por fila con fecha, monto, score y severidad
- consensus_matrix.csv — matriz completa con scores por método
- anomaly_summary.txt — resumen textual con estadísticas de detección

Uso:
    from src.anomalies.report import generate_anomaly_report
    stats = generate_anomaly_report(consensus_df, df_original)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columnas para el reporte resumido
_COLUMNAS_REPORTE = [
    "fecha",
    "caja_neta",
    "consenso_score",
    "severidad",
    "metodos_detectores",
]


def generate_anomaly_report(
    consensus_df: pd.DataFrame,
    df_original: pd.DataFrame,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Genera reportes de anomalías en archivos CSV y resumen textual.

    Produce 3 archivos:
    1. {output_dir}/anomaly_report.csv — resumen por registro
    2. {output_dir}/consensus_matrix.csv — matriz completa de consenso
    3. {output_dir}/anomaly_summary.txt — resumen textual con estadísticas

    Args:
        consensus_df: DataFrame generado por build_consensus_matrix.
        df_original: DataFrame original con los datos (debe contener
            'fecha', 'caja_neta' y opcionalmente 'tiene_anomalia').
        output_dir: Directorio donde guardar los reportes.

    Returns:
        Diccionario con estadísticas:
            - total_anomalies: int, total de anomalías detectadas
            - by_severity: dict, {severidad: count}
            - by_method: dict, {método: count}
            - detection_rate_vs_known: float o None, proporción de
              anomalías conocidas detectadas (si existe tiene_anomalia)
            - total_known: int o None, total de anomalías conocidas
            - output_files: lista de rutas de archivos generados
    """
    if consensus_df.empty or df_original.empty:
        logger.warning("DataFrames vacíos — no se generan reportes")
        return {
            "total_anomalies": 0,
            "by_severity": {},
            "by_method": {},
            "detection_rate_vs_known": None,
            "total_known": None,
            "output_files": [],
        }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. anomaly_report.csv ────────────────────────────────────────────────
    reporte = _armar_reporte_resumido(consensus_df, df_original)
    path_report = os.path.join(output_dir, "anomaly_report.csv")
    reporte.to_csv(path_report, index=False)
    logger.info("Reporte de anomalías guardado: %s", path_report)

    # ── 2. consensus_matrix.csv ──────────────────────────────────────────────
    path_matrix = os.path.join(output_dir, "consensus_matrix.csv")
    consensus_df.to_csv(path_matrix, index=False)
    logger.info("Matriz de consenso guardada: %s", path_matrix)

    # ── 3. anomaly_summary.txt ───────────────────────────────────────────────
    stats = _calcular_estadisticas(consensus_df, df_original)
    path_summary = os.path.join(output_dir, "anomaly_summary.txt")
    _escribir_resumen_textual(path_summary, stats, consensus_df, df_original)

    stats["output_files"] = [path_report, path_matrix, path_summary]

    logger.info(
        "Reportes generados: %d anomalías totales, %d archivos",
        stats["total_anomalies"],
        len(stats["output_files"]),
    )

    return stats


def _armar_reporte_resumido(
    consensus_df: pd.DataFrame,
    df_original: pd.DataFrame,
) -> pd.DataFrame:
    """Arma el reporte resumido combinando datos de consenso y originales.

    Args:
        consensus_df: DataFrame de consenso.
        df_original: DataFrame original.

    Returns:
        DataFrame con columnas: fecha, caja_neta, consenso_score,
        severidad, metodos_detectores.
    """
    reporte = pd.DataFrame(index=consensus_df.index)

    # fecha
    if "fecha" in consensus_df.columns:
        reporte["fecha"] = consensus_df["fecha"]
    elif "fecha" in df_original.columns:
        reporte["fecha"] = pd.to_datetime(df_original["fecha"])

    # caja_neta
    if "caja_neta" in df_original.columns:
        reporte["caja_neta"] = df_original["caja_neta"]
    else:
        reporte["caja_neta"] = float("nan")

    # consenso_score
    reporte["consenso_score"] = consensus_df.get(
        "consenso_score", float("nan")
    )

    # severidad
    reporte["severidad"] = consensus_df.get("severidad", None)

    # metodos_detectores — convertir lista a string para CSV
    metodos = consensus_df.get("metodos_detectores", None)
    if metodos is not None:
        reporte["metodos_detectores"] = metodos.apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    else:
        reporte["metodos_detectores"] = ""

    return reporte[_COLUMNAS_REPORTE]


def _calcular_estadisticas(
    consensus_df: pd.DataFrame,
    df_original: pd.DataFrame,
) -> Dict[str, Any]:
    """Calcula estadísticas de detección de anomalías.

    Args:
        consensus_df: DataFrame de consenso.
        df_original: DataFrame original.

    Returns:
        Dict con estadísticas calculadas.
    """
    severidad = consensus_df["severidad"]
    tiene_severidad = severidad.notna()

    total_anomalies = int(tiene_severidad.sum())

    # Por severidad
    by_severity: Dict[str, int] = {}
    for sev in ["bajo", "medio", "alto", "crítico"]:
        count = int((severidad == sev).sum())
        if count > 0:
            by_severity[sev] = count

    # Por método
    cols_metodos = [
        "if_detectado",
        "zscore_detectado",
        "iqr_detectado",
        "benford_detectado",
        "temporal_detectado",
    ]
    nombres_metodos = {
        "if_detectado": "Isolation Forest",
        "zscore_detectado": "Z-score",
        "iqr_detectado": "IQR",
        "benford_detectado": "Benford",
        "temporal_detectado": "Temporal",
    }
    by_method: Dict[str, int] = {}
    for col in cols_metodos:
        if col in consensus_df.columns:
            count = int(consensus_df[col].sum())
            if count > 0:
                by_method[nombres_metodos.get(col, col)] = count

    # Tasa de detección vs anomalías conocidas
    detection_rate_vs_known: float | None = None
    total_known: int | None = None

    if "tiene_anomalia" in df_original.columns:
        known_anomalies = df_original["tiene_anomalia"].astype(bool)
        total_known = int(known_anomalies.sum())

        if total_known > 0:
            # Coincidencia de índices
            if len(consensus_df) == len(df_original):
                detected_known = (
                    (severidad.notna()) & known_anomalies
                ).sum()
                detection_rate_vs_known = round(
                    detected_known / total_known, 4
                )
                logger.info(
                    "Tasa de detección vs anomalías conocidas: "
                    "%d/%d (%.2f%%)",
                    detected_known,
                    total_known,
                    detection_rate_vs_known * 100,
                )

    return {
        "total_anomalies": total_anomalies,
        "by_severity": by_severity,
        "by_method": by_method,
        "detection_rate_vs_known": detection_rate_vs_known,
        "total_known": total_known,
    }


def _escribir_resumen_textual(
    path: str,
    stats: Dict[str, Any],
    consensus_df: pd.DataFrame,
    df_original: pd.DataFrame,
) -> None:
    """Escribe el resumen textual de anomalías.

    Args:
        path: Ruta del archivo a generar.
        stats: Estadísticas calculadas.
        consensus_df: DataFrame de consenso.
        df_original: DataFrame original.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("RESUMEN DE DETECCIÓN DE ANOMALÍAS")
    lines.append("=" * 60)
    lines.append("")

    # Total de anomalías
    lines.append(f"Total de registros analizados: {len(consensus_df)}")
    lines.append(
        f"Total de anomalías detectadas: {stats['total_anomalies']}"
        f" ({100.0 * stats['total_anomalies'] / len(consensus_df):.2f}%)"
    )
    lines.append("")

    # Por severidad
    lines.append("--- Distribución por severidad ---")
    for sev in ["crítico", "alto", "medio", "bajo"]:
        count = stats["by_severity"].get(sev, 0)
        pct = 100.0 * count / stats["total_anomalies"] if stats["total_anomalies"] > 0 else 0.0
        lines.append(f"  {sev:12s}: {count:4d} ({pct:5.2f}%)")
    lines.append("")

    # Por método
    lines.append("--- Detección por método ---")
    for metodo, count in sorted(
        stats["by_method"].items(), key=lambda x: x[1], reverse=True
    ):
        lines.append(f"  {metodo:20s}: {count:4d}")
    lines.append("")

    # Top 10 anomalías más severas
    lines.append("--- Top 10 anomalías más severas ---")
    severidad_orden = {"crítico": 0, "alto": 1, "medio": 2, "bajo": 3}
    temp_df = consensus_df.copy()
    temp_df["_orden_sev"] = temp_df["severidad"].map(
        lambda s: severidad_orden.get(s, 99) if pd.notna(s) else 99
    )
    temp_df = temp_df.sort_values(
        ["_orden_sev", "consenso_score"],
        ascending=[True, False],
    )

    top10 = temp_df.head(10)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        fecha_str = (
            str(row["fecha"])
            if pd.notna(row.get("fecha"))
            else "N/A"
        )
        score = row.get("consenso_score", float("nan"))
        sev = row.get("severidad", "N/A")
        metodos = row.get("metodos_detectores", [])
        metodos_str = (
            ", ".join(metodos)
            if isinstance(metodos, list)
            else str(metodos)
        )
        lines.append(
            f"  {i:2d}. [{sev:8s}] score={score:.2f} | "
            f"{fecha_str} | métodos: {metodos_str}"
        )
    lines.append("")

    # Tasa de detección vs anomalías conocidas
    if stats["detection_rate_vs_known"] is not None:
        lines.append("--- Precisión vs anomalías conocidas ---")
        rate = stats["detection_rate_vs_known"]
        pct = rate * 100
        lines.append(
            f"  Tasa de detección: {pct:.2f}%"
        )
        lines.append(
            f"  Anomalías conocidas totales: {stats['total_known']}"
        )
    else:
        lines.append(
            "--- Precisión vs anomalías conocidas ---"
        )
        lines.append("  No disponible (dataset no tiene columna 'tiene_anomalia')")

    lines.append("")
    lines.append("=" * 60)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Resumen textual guardado: %s", path)
