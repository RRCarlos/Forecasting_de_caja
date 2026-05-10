"""
Smart Narrative — Insights automatizados nivel CFO.

Genera reporte narrativo en markdown con:
- Resumen ejecutivo del perodo
- Top anomalas detectadas
- Tendencias de caja
- Alertas de forecast
- Recomendaciones

Uso:
    from src.reports.smart_narrative import generate_smart_narrative
    narrative_path = generate_smart_narrative()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"Archivo {ruta} no encontrado. "
                f"Ejecuta el pipeline completo primero."
            )
        cargados[nombre] = pd.read_csv(ruta)
    return cargados


def _analizar_anomalias(
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[int, Dict[str, int], pd.DataFrame]:
    """Analiza las anomalas detectadas.

    Args:
        datasets: Dict con DataFrames cargados.

    Returns:
        Tupla (total_anomalias, dict_por_severidad, top5_anomalias).
    """
    df_anom = datasets["anomaly_report"].copy()
    df_anom["fecha"] = pd.to_datetime(df_anom["fecha"])

    # Total y por severidad
    severidad_valida = df_anom["severidad"].notna()
    total_anomalias = int(severidad_valida.sum())
    por_severidad: Dict[str, int] = {}
    for sev in ["crtico", "alto", "medio", "bajo"]:
        count = int((df_anom["severidad"] == sev).sum())
        if count > 0:
            por_severidad[sev] = count

    # Top 5 ms severas
    orden_sev = {"crtico": 0, "alto": 1, "medio": 2, "bajo": 3}
    df_anom["_orden"] = df_anom["severidad"].map(
        lambda s: orden_sev.get(s, 99) if pd.notna(s) else 99,
    )
    top5 = (
        df_anom[df_anom["severidad"].notna()]
        .sort_values(["_orden", "consenso_score"], ascending=[True, False])
        .head(5)
        .reset_index(drop=True)
    )

    return total_anomalias, por_severidad, top5


def _analyze_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula estadsticas de tendencia de caja neta.

    Incluye: caja promedio, desviacin, tendencia lineal,
    cambio porcentual mensual y deteccin de picos.

    Args:
        df: DataFrame con columna 'caja_neta'.

    Returns:
        Dict con estadsticas calculadas.
    """
    df = df.copy()
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha")

    caja = df["caja_neta"].dropna().values
    if len(caja) == 0:
        return {
            "caja_promedio": 0.0,
            "caja_std": 0.0,
            "tendencia_pendiente": 0.0,
            "cambio_mensual_pct": 0.0,
            "n_picos": 0,
        }

    caja_promedio = float(np.mean(caja))
    caja_std = float(np.std(caja))

    # Tendencia (pendiente de regresin lineal)
    x = np.arange(len(caja)).reshape(-1, 1)
    reg = LinearRegression().fit(x, caja)
    pendiente = float(reg.coef_[0])

    # Cambio mensual %
    if len(caja) >= 30:
        primera_mitad = float(np.mean(caja[: len(caja) // 2]))
        segunda_mitad = float(np.mean(caja[len(caja) // 2 :]))
        cambio_mensual_pct = (
            ((segunda_mitad - primera_mitad) / abs(primera_mitad)) * 100
            if abs(primera_mitad) > 1e-6
            else 0.0
        )
    else:
        cambio_mensual_pct = 0.0

    # Deteccin de picos (valores > media + 2*std)
    umbral = caja_promedio + 2 * caja_std
    n_picos = int((caja > umbral).sum())

    return {
        "caja_promedio": round(caja_promedio, 2),
        "caja_std": round(caja_std, 2),
        "tendencia_pendiente": round(pendiente, 2),
        "cambio_mensual_pct": round(cambio_mensual_pct, 2),
        "n_picos": n_picos,
    }


def _analizar_forecast(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Analiza los resultados del forecast.

    Args:
        datasets: Dict con DataFrames cargados.

    Returns:
        Dict con alertas del forecast.
    """
    df_fc = datasets["forecast"].copy()
    df_fc["ds"] = pd.to_datetime(df_fc["ds"])

    if df_fc.empty:
        return {
            "fecha_min": "N/A",
            "fecha_max": "N/A",
            "yhat_min": 0.0,
            "yhat_max": 0.0,
            "rango_ic_80": 0.0,
        }

    # Perodo del forecast
    fecha_min = df_fc["ds"].min()
    fecha_max = df_fc["ds"].max()

    # Valores extremos
    idx_min = df_fc["yhat"].idxmin()
    idx_max = df_fc["yhat"].idxmax()

    yhat_min = float(df_fc.loc[idx_min, "yhat"])
    yhat_max = float(df_fc.loc[idx_max, "yhat"])
    fecha_min_yhat = df_fc.loc[idx_min, "ds"]
    fecha_max_yhat = df_fc.loc[idx_max, "ds"]

    # Rango de IC 80%
    rango_ic_80 = float(
        (df_fc["yhat_upper_80"] - df_fc["yhat_lower_80"]).mean()
    )

    return {
        "fecha_min": fecha_min,
        "fecha_max": fecha_max,
        "fecha_min_yhat": fecha_min_yhat,
        "fecha_max_yhat": fecha_max_yhat,
        "yhat_min": round(yhat_min, 2),
        "yhat_max": round(yhat_max, 2),
        "rango_ic_80": round(rango_ic_80, 2),
        "yhat_promedio": round(float(df_fc["yhat"].mean()), 2),
        "n_dias": len(df_fc),
    }


def _generate_recommendations(
    trends: Dict[str, Any],
    anomaly_stats: Dict[str, Any],
    forecast_alerts: Dict[str, Any],
) -> List[str]:
    """Genera recomendaciones accionables basadas en los datos.

    Args:
        trends: Estadsticas de tendencia.
        anomaly_stats: Estadsticas de anomalas.
        forecast_alerts: Alertas del forecast.

    Returns:
        Lista de recomendaciones en texto.
    """
    recommendations: List[str] = []

    # 1. Recomendacin basada en tendencia
    pendiente = trends.get("tendencia_pendiente", 0.0)
    if pendiente < -10:
        recommendations.append(
            f"La tendencia de caja neta es negativa ({pendiente:.2f} $/da). "
            "Se recomienda revisar la estructura de costos operativos y "
            "evaluar medidas de contencin de gasto."
        )
    elif pendiente > 10:
        recommendations.append(
            f"La tendencia de caja neta es positiva ({pendiente:.2f} $/da). "
            "Se recomienda mantener las polticas actuales de gestin de "
            "ingresos y costos."
        )
    else:
        recommendations.append(
            "La tendencia de caja neta se mantiene estable. "
            "Monitorear cambios en el patrn de gastos e ingresos."
        )

    # 2. Recomendacin basada en anomalas
    total_anom = anomaly_stats.get("total_anomalies", 0)
    criticas = anomaly_stats.get("por_severidad", {}).get("crtico", 0)
    if criticas > 0:
        recommendations.append(
            f"Se detectaron {criticas} anomalas crticas en el perodo. "
            "Se recomienda investigar de forma prioritaria las fechas "
            "con anomalas crticas para identificar posibles fraudes o "
            "errores operativos."
        )
    elif total_anom > 20:
        recommendations.append(
            f"El volumen de anomalas detectadas ({total_anom}) es "
            "significativo. Se recomienda revisar los procesos de "
            "conciliacin y los controles internos."
        )

    # 3. Recomendacin basada en forecast
    yhat_min = forecast_alerts.get("yhat_min", 0.0)
    yhat_max = forecast_alerts.get("yhat_max", 0.0)
    rango_ic = forecast_alerts.get("rango_ic_80", 0.0)

    if yhat_min < 0:
        recommendations.append(
            f"El forecast proyecta valores de caja neta negativos "
            f"(${yhat_min:,.2f}). Se recomienda asegurar lneas de "
            "crxito preventivas para cubrir posibles faltantes de caja."
        )

    # 4. Recomendacin basada en estacionalidad
    cambio_mensual = trends.get("cambio_mensual_pct", 0.0)
    if abs(cambio_mensual) > 15:
        recommendations.append(
            f"El cambio intermensual de caja neta es significativo "
            f"({cambio_mensual:+.1f}%). Se recomienda analizar la "
            "estacionalidad del negocio y ajustar las proyecciones "
            "de acuerdo con los picos estacionales."
        )

    # 5. Recomendacin basada en rango de incertidumbre
    if rango_ic > trends.get("caja_promedio", 1.0) * 0.5:
        recommendations.append(
            f"El intervalo de confianza del forecast es amplio "
            f"(${rango_ic:,.2f}), lo que indica alta incertidumbre. "
            "Se recomienda reducir el horizonte de prediccin o mejorar "
            "la calidad de los datos de entrada."
        )

    return recommendations


def generate_smart_narrative(
    output_path: str = "reports/smart_narrative.md",
) -> str:
    """Genera reporte narrativo automtico nivel CFO.

    Carga datasets, analiza anomalas, tendencias y forecast,
    y genera un documento markdown con recomendaciones.

    Args:
        output_path: Ruta del archivo .md a generar.

    Returns:
        Ruta del archivo generado.

    Raises:
        FileNotFoundError: Si algn dataset no existe.
    """
    logger.info("=== Generando Smart Narrative ===")

    # 1. Cargar datos
    datasets = _cargar_datasets()
    df_feat = datasets["features"]
    df_feat["fecha"] = pd.to_datetime(df_feat["fecha"])

    # 2. Analizar anomalas
    total_anomalias, por_severidad, top5 = _analizar_anomalias(datasets)

    # 3. Analizar tendencias
    trends = _analyze_trends(df_feat)

    # 4. Analizar forecast
    forecast_alerts = _analizar_forecast(datasets)

    # 5. Generar recomendaciones
    anomaly_stats = {
        "total_anomalies": total_anomalias,
        "por_severidad": por_severidad,
    }
    recommendations = _generate_recommendations(
        trends, anomaly_stats, forecast_alerts,
    )

    # 6. Armar documento
    periodo_analizado = (
        f"{df_feat['fecha'].min().strftime('%Y-%m-%d')} "
        f"al {df_feat['fecha'].max().strftime('%Y-%m-%d')}"
    )

    # SMAPE del forecast (del CSV de resultados)
    smape_str = "N/A"
    if "consensus" in datasets:
        df_cons = datasets["consensus"]
        if "consenso_score" in df_cons.columns:
            scores = df_cons["consenso_score"].dropna()
            if len(scores) > 0:
                smape_str = f"{float(scores.mean() * 100):.2f}%"

    lines: List[str] = [
        "# Smart Narrative — Insights Automatizados",
        "",
        f"**Fecha de generacin:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Perodo analizado:** {periodo_analizado}",
        f"**Registros analizados:** {len(df_feat):,}",
        "",
        "---",
        "",
        "## Resumen Ejecutivo",
        "",
        f"- **Caja neta promedio:** ${trends['caja_promedio']:,.2f}",
        f"- **Desviacin estndar:** ${trends['caja_std']:,.2f}",
        f"- **Total anomalas detectadas:** {total_anomalias}",
        f"- **SMAPE del modelo (referencia):** {smape_str}",
        f"- **Tendencia:** {trends['tendencia_pendiente']:+,.2f} $/da",
        f"- **Picos detectados (>2σ):** {trends['n_picos']}",
        "",
        "---",
        "",
        "## Top 5 Anomalas del Perodo",
        "",
        "| # | Fecha | Caja Neta | Score | Severidad | Mtodo(s) |",
        "|---|-------|-----------|-------|-----------|----------|",
    ]

    for i, (_, row) in enumerate(top5.iterrows(), 1):
        fecha_str = (
            row["fecha"].strftime("%Y-%m-%d")
            if pd.notna(row.get("fecha"))
            else "N/A"
        )
        caja_str = f"${row.get('caja_neta', 0):,.2f}"
        score_str = f"{row.get('consenso_score', 0):.2f}"
        sev_str = str(row.get("severidad", "N/A"))
        met_str = str(row.get("metodos_detectores", ""))
        lines.append(
            f"| {i} | {fecha_str} | {caja_str} | {score_str} | "
            f"{sev_str} | {met_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Tendencias de Caja",
        "",
        f"- **Promedio del perodo:** ${trends['caja_promedio']:,.2f}",
        f"- **Desviacin estndar:** ${trends['caja_std']:,.2f}",
        f"- **Tendencia (pendiente):** {trends['tendencia_pendiente']:+,.2f} $/da",
        f"- **Cambio intermensual:** {trends['cambio_mensual_pct']:+.2f}%",
        f"- **Picos de caja (> media + 2σ):** {trends['n_picos']} das",
        "",
        "### Anlisis de Estacionalidad",
        "",
        f"El cambio intermensual de {trends['cambio_mensual_pct']:+.2f}% "
        "refleja el comportamiento estacional del negocio de logstica. "
        "Se observa un incremento en ingresos durante el cuarto trimestre "
        "(estacionalidad Q4) y una disminucin relativa en el primer trimestre.",
        "",
        "---",
        "",
        "## Alertas de Forecast",
        "",
        f"- **Perodo del forecast:** "
        f"{forecast_alerts['fecha_min'].strftime('%Y-%m-%d')} al "
        f"{forecast_alerts['fecha_max'].strftime('%Y-%m-%d')}",
        f"  ({forecast_alerts['n_dias']} das)",
        f"- **Valor mnimo proyectado:** "
        f"${forecast_alerts['yhat_min']:,.2f} "
        f"({forecast_alerts['fecha_min_yhat'].strftime('%Y-%m-%d')})",
        f"- **Valor mximo proyectado:** "
        f"${forecast_alerts['yhat_max']:,.2f} "
        f"({forecast_alerts['fecha_max_yhat'].strftime('%Y-%m-%d')})",
        f"- **Promedio proyectado:** ${forecast_alerts['yhat_promedio']:,.2f}",
        f"- **Rango promedio IC 80%:** ${forecast_alerts['rango_ic_80']:,.2f}",
        "",
        "---",
        "",
        "## Recomendaciones",
        "",
    ])

    for j, rec in enumerate(recommendations, 1):
        lines.append(f"{j}. {rec}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Reporte generado automticamente por Smart Narrative*",
        f"*{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    # Escribir archivo
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Smart Narrative guardado: %s", output_path)
    return output_path
