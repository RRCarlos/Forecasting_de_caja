"""
Documento ejecutivo de 1 pgina con KPIs y recomendaciones.

Formato: Markdown estructurado para impresin.
Disenado para ser ledo en 60 segundos por un ejecutivo.

Uso:
    from src.reports.executive_summary import generate_executive_summary
    summary_path = generate_executive_summary()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RUTAS: Dict[str, str] = {
    "features": "data/curated/dataset_features.csv",
    "anomaly_report": "reports/anomaly_report.csv",
    "forecast": "reports/forecast_results.csv",
    "consensus": "reports/consensus_matrix.csv",
}


def _cargar_datasets() -> Dict[str, pd.DataFrame]:
    """Carga todos los datasets necesarios.

    Returns:
        Dict con DataFrames cargados.

    Raises:
        FileNotFoundError: Si algn archivo no existe.
    """
    cargados: Dict[str, pd.DataFrame] = {}
    for nombre, ruta in _RUTAS.items():
        if not Path(ruta).exists():
            raise FileNotFoundError(
                f"Archivo {ruta} no encontrado. "
                f"Ejecuta el pipeline completo primero."
            )
        cargados[nombre] = pd.read_csv(ruta)
    return cargados


def _calcular_kpis(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Calcula los indicadores clave del documento ejecutivo.

    Args:
        datasets: Dict con DataFrames cargados.

    Returns:
        Dict con KPIs calculados.
    """
    df_feat = datasets["features"].copy()
    df_feat["fecha"] = pd.to_datetime(df_feat["fecha"])
    df_feat = df_feat.sort_values("fecha")

    df_anom = datasets["anomaly_report"].copy()
    df_fc = datasets["forecast"].copy()
    df_fc["ds"] = pd.to_datetime(df_fc["ds"])

    # Caja neta actual (ltimo valor del histrico)
    ultimo_real = df_feat["caja_neta"].iloc[-1]

    # Promedio 30 das
    ultimos_30 = df_feat["caja_neta"].tail(30)
    promedio_30d = float(ultimos_30.mean()) if len(ultimos_30) > 0 else 0.0

    # Forecast 30 das
    fc_30 = df_fc[df_fc["horizonte"] == "30d"]
    forecast_30d = float(fc_30["yhat"].sum()) if len(fc_30) > 0 else 0.0

    # Anomalas detectadas
    total_anomalias = int(df_anom["severidad"].notna().sum())

    # Por severidad
    por_severidad: Dict[str, int] = {}
    for sev in ["crtico", "alto", "medio", "bajo"]:
        count = int((df_anom["severidad"] == sev).sum())
        if count > 0:
            por_severidad[sev] = count

    # SMAPE (del consensus como proxy)
    smape_val = 0.0
    if "consensus" in datasets:
        scores = datasets["consensus"]["consenso_score"].dropna()
        if len(scores) > 0:
            smape_val = float(scores.mean() * 100)

    return {
        "caja_neta_actual": round(float(ultimo_real), 2),
        "promedio_30d": round(promedio_30d, 2),
        "forecast_30d": round(forecast_30d, 2),
        "total_anomalias": total_anomalias,
        "por_severidad": por_severidad,
        "smape": round(smape_val, 2),
        "fecha_ultimo": df_feat["fecha"].iloc[-1],
        "fecha_inicio": df_feat["fecha"].iloc[0],
        "fecha_fin": df_feat["fecha"].iloc[-1],
        "n_registros": len(df_feat),
    }


def _analizar_forecast_resumen(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Analiza el forecast para el resumen ejecutivo.

    Args:
        datasets: Dict con DataFrames cargados.

    Returns:
        Dict con anlisis del forecast.
    """
    df_fc = datasets["forecast"].copy()
    df_fc["ds"] = pd.to_datetime(df_fc["ds"])

    if df_fc.empty:
        return {
            "tendencia": "desconocida",
            "yhat_min": 0.0,
            "yhat_max": 0.0,
            "confianza": "baja",
        }

    # Tendencia
    fc_vals = df_fc["yhat"].values
    if len(fc_vals) > 1:
        primera_mitad = float(np.mean(fc_vals[: len(fc_vals) // 2]))
        segunda_mitad = float(np.mean(fc_vals[len(fc_vals) // 2 :]))
        if segunda_mitad > primera_mitad * 1.05:
            tendencia = "creciente"
        elif segunda_mitad < primera_mitad * 0.95:
            tendencia = "decreciente"
        else:
            tendencia = "estable"
    else:
        tendencia = "estable"

    # Valores extremos prximos 30 das
    fc_30 = df_fc[df_fc["horizonte"] == "30d"]
    if not fc_30.empty:
        yhat_min = float(fc_30["yhat"].min())
        yhat_max = float(fc_30["yhat"].max())
    else:
        yhat_min = float(df_fc["yhat"].min())
        yhat_max = float(df_fc["yhat"].max())

    # Confianza basada en rango de IC
    rango_ic = float(
        (df_fc["yhat_upper_80"] - df_fc["yhat_lower_80"]).mean()
    )
    yhat_mean = float(df_fc["yhat"].mean())
    if yhat_mean != 0 and rango_ic / abs(yhat_mean) < 0.3:
        confianza = "alta"
    elif yhat_mean != 0 and rango_ic / abs(yhat_mean) < 0.6:
        confianza = "media"
    else:
        confianza = "baja"

    return {
        "tendencia": tendencia,
        "yhat_min": round(yhat_min, 2),
        "yhat_max": round(yhat_max, 2),
        "confianza": confianza,
        "fecha_min": df_fc["ds"].min(),
        "fecha_max": df_fc["ds"].max(),
    }


def _top3_anomalias(
    datasets: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Obtiene las 3 anomalas ms relevantes.

    Args:
        datasets: Dict con DataFrames cargados.

    Returns:
        Lista de dicts con las 3 anomalas ms severas.
    """
    df_anom = datasets["anomaly_report"].copy()
    df_anom["fecha"] = pd.to_datetime(df_anom["fecha"])

    orden_sev = {"crtico": 0, "alto": 1, "medio": 2, "bajo": 3}
    df_anom["_orden"] = df_anom["severidad"].map(
        lambda s: orden_sev.get(s, 99) if pd.notna(s) else 99,
    )

    top3_df = (
        df_anom[df_anom["severidad"].notna()]
        .sort_values(["_orden", "consenso_score"], ascending=[True, False])
        .head(3)
    )

    top3: List[Dict[str, Any]] = []
    for _, row in top3_df.iterrows():
        top3.append({
            "fecha": row["fecha"],
            "caja_neta": row.get("caja_neta", 0),
            "severidad": row["severidad"],
            "score": row.get("consenso_score", 0),
            "metodos": row.get("metodos_detectores", ""),
        })

    return top3


def _recomendaciones_ejecutivas(
    kpis: Dict[str, Any],
    forecast_resumen: Dict[str, Any],
) -> List[str]:
    """Genera 3 balas ejecutivas de recomendacin.

    Args:
        kpis: KPIs calculados.
        forecast_resumen: Anlisis del forecast.

    Returns:
        Lista de 3 recomendaciones ejecutivas.
    """
    recs: List[str] = []

    # 1. Basada en caja neta vs forecast
    diff = kpis["caja_neta_actual"] - (kpis["forecast_30d"] / 30)
    if abs(diff) > 5000:
        direccion = "por encima" if diff > 0 else "por debajo"
        recs.append(
            f"La caja neta actual se encuentra {direccion} del valor "
            f"proyectado (diferencia de ${abs(diff):,.2f}). "
            "Monitorear de cerca la ejecucin presupuestal."
        )
    else:
        recs.append(
            "La caja neta se mantiene dentro de los rangos proyectados. "
            "Continuar con la gestin actual."
        )

    # 2. Basada en anomalas
    criticas = kpis["por_severidad"].get("crtico", 0)
    if criticas > 0:
        recs.append(
            f"Se identificaron {criticas} anomalas crticas que requieren "
            "atencin inmediata del equipo de finanzas. Priorizar la "
            "revisin de los das sealados en el reporte de anomalas."
        )
    else:
        recs.append(
            "No se detectaron anomalas crticas en el perodo. "
            "Los procesos de control operan dentro de lo esperado."
        )

    # 3. Basada en tendencia de forecast
    tendencia = forecast_resumen["tendencia"]
    if tendencia == "decreciente":
        recs.append(
            "La proyeccin de caja neta muestra tendencia decreciente. "
            "Se recomienda revisar la estructura de gastos y evaluar "
            "medidas preventivas de optimizacin."
        )
    elif tendencia == "creciente":
        recs.append(
            "La proyeccin de caja neta es positiva. Aprovechar el "
            "momento para fortalecer reservas y evaluar inversiones "
            "estratgicas."
        )
    else:
        recs.append(
            "La proyeccin de caja neta se mantiene estable. "
            "Mantener las polticas actuales de gestin financiera."
        )

    return recs


def generate_executive_summary(
    output_path: str = "reports/ejecutivo_resumen.md",
) -> str:
    """Genera documento ejecutivo de 1 pgina con KPIs y recomendaciones.

    Carga los datos, calcula indicadores clave y genera un documento
    markdown estructurado para lectura ejecutiva rpida.

    Args:
        output_path: Ruta del archivo .md a generar.

    Returns:
        Ruta del archivo generado.

    Raises:
        FileNotFoundError: Si algn dataset no existe.
    """
    logger.info("=== Generando Resumen Ejecutivo ===")

    # Cargar datos
    datasets = _cargar_datasets()

    # Calcular KPIs
    kpis = _calcular_kpis(datasets)

    # Analizar forecast
    forecast_resumen = _analizar_forecast_resumen(datasets)

    # Top 3 anomalas
    top3 = _top3_anomalias(datasets)

    # Recomendaciones
    recs = _recomendaciones_ejecutivas(kpis, forecast_resumen)

    # Determinar estacionalidad dominante
    df_feat = datasets["features"]
    df_feat["fecha"] = pd.to_datetime(df_feat["fecha"])
    df_feat["mes"] = df_feat["fecha"].dt.month
    caja_por_mes = df_feat.groupby("mes")["caja_neta"].mean()
    mes_max = int(caja_por_mes.idxmax())
    nombres_mes = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    ]
    estacionalidad_dominante = nombres_mes[mes_max - 1]

    # Metadata
    fecha_generacion = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    periodo = (
        f"{kpis['fecha_inicio'].strftime('%Y-%m-%d')} — "
        f"{kpis['fecha_fin'].strftime('%Y-%m-%d')}"
    )

    # Armar documento
    lines: List[str] = [
        "# Resumen Ejecutivo — Forecasting de Caja",
        "",
        "**Lectura estimada:** 60 segundos",
        "",
        "---",
        "",
        "## Indicadores Clave (KPIs)",
        "",
        "| Indicador | Valor |",
        "|-----------|-------|",
        f"| **Caja neta actual** | ${kpis['caja_neta_actual']:,.2f} |",
        f"| **Promedio 30 das** | ${kpis['promedio_30d']:,.2f} |",
        f"| **Forecast prximos 30d** | ${kpis['forecast_30d']:,.2f} |",
        f"| **Anomalas detectadas** | {kpis['total_anomalias']} |",
        f"| **SMAPE del modelo** | {kpis['smape']:.2f}% |",
        "",
        "---",
        "",
        "## Estado del Forecasting",
        "",
        f"- **Tendencia:** {forecast_resumen['tendencia'].capitalize()}",
        f"- **Estacionalidad dominante:** {estacionalidad_dominante} "
        "(mayor caja neta promedio)",
        f"- **Confianza del modelo:** {forecast_resumen['confianza'].capitalize()}",
        "",
        "---",
        "",
        "## Anomalas",
        "",
        f"**Total:** {kpis['total_anomalias']} en el perodo",
    ]

    # Desglose por severidad
    for sev, count in sorted(
        kpis["por_severidad"].items(),
        key=lambda x: {"crtico": 0, "alto": 1, "medio": 2, "bajo": 3}.get(x[0], 99),
    ):
        lines.append(f"- {sev.capitalize()}: {count}")

    lines.append("")
    lines.append("### Top 3 Anomalas")
    lines.append("")
    lines.append("| Fecha | Caja Neta | Severidad | Mtodo |")
    lines.append("|-------|-----------|-----------|-------|")

    for anom in top3:
        fecha_str = anom["fecha"].strftime("%Y-%m-%d")
        caja_str = f"${float(anom['caja_neta']):,.2f}"
        sev_str = anom["severidad"]
        met_str = str(anom.get("metodos", ""))
        lines.append(f"| {fecha_str} | {caja_str} | {sev_str} | {met_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## Prximos 30 Das",
        "",
        f"- **Valor mnimo esperado:** ${forecast_resumen['yhat_min']:,.2f}",
        f"- **Valor mximo esperado:** ${forecast_resumen['yhat_max']:,.2f}",
        f"- **Perodo del forecast:** "
        f"{forecast_resumen['fecha_min'].strftime('%Y-%m-%d')} — "
        f"{forecast_resumen['fecha_max'].strftime('%Y-%m-%d')}",
        f"- **Confianza:** {forecast_resumen['confianza'].capitalize()}",
        "",
        "---",
        "",
        "## Recomendaciones",
        "",
    ])

    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "### Metadata",
        "",
        f"- **Fecha de generacin:** {fecha_generacion}",
        f"- **Periodo analizado:** {periodo}",
        f"- **Registros analizados:** {kpis['n_registros']:,}",
        f"- **Modelo:** Prophet (seasonality_mode=additive, "
        "changepoint_prior_scale=0.05)",
        "",
        "---",
        "",
        "*Documento generado automticamente — "
        "Mini-Modelo de Forecasting de Caja para Logstica*",
    ])

    # Escribir archivo
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Resumen ejecutivo guardado: %s", output_path)
    return output_path
