"""
Dashboard Profesional de Forecasting de Caja para Logística.

Dashboard web interactivo construido con Dash + Plotly.
Estilo: corporativo, minimalista, con enfoque en KPIs y evolución temporal.

Uso:
    python dashboards/app.py
    # Abrir http://127.0.0.1:8050
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "curated"
REPORTS_DIR = BASE_DIR / "reports"
DASHBOARDS_DIR = BASE_DIR / "dashboards"

COLORS = {
    "primary": "#1a237e",
    "primary_light": "#0d47a1",
    "primary_lighter": "#e8eaf6",
    "green": "#00c853",
    "red": "#ff1744",
    "amber": "#ffc107",
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "purple": "#7b1fa2",
    "teal": "#00bcd4",
    "grey": "#90a4ae",
    "text": "#1a1a2e",
    "text_secondary": "#6c757d",
    "bg": "#f4f5fa",
    "card_bg": "#ffffff",
    "border": "#e9ecef",
}

SEVERIDAD_COLORS = {
    "crítico": COLORS["red"],
    "alto": COLORS["amber"],
    "medio": COLORS["blue"],
    "bajo": COLORS["grey"],
}

# ──────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────


def load_data() -> Dict[str, Any]:
    """Carga todos los datos necesarios para el dashboard.

    Returns:
        Dict con DataFrames: powerbi, anomalies, features, backtest.
        Si falla la carga, retorna DataFrames vacíos.
    """
    data: Dict[str, Any] = {}
    data["loaded"] = False

    # ── PowerBI CSV (merge completo) ──────────────────────────────────
    pbi_path = REPORTS_DIR / "forecast_powerbi.csv"
    if pbi_path.exists():
        df_pbi = pd.read_csv(pbi_path, parse_dates=["fecha"])
        data["powerbi"] = df_pbi
        logger.info("PowerBI CSV cargado: %d filas", len(df_pbi))
    else:
        data["powerbi"] = pd.DataFrame()
        logger.warning("No encontrado: %s", pbi_path)

    # ── Anomaly report ────────────────────────────────────────────────
    anom_path = REPORTS_DIR / "anomaly_report.csv"
    if anom_path.exists():
        df_anom = pd.read_csv(anom_path, parse_dates=["fecha"])
        data["anomalies"] = df_anom
    else:
        data["anomalies"] = pd.DataFrame()

    # ── Forecast results ──────────────────────────────────────────────
    fcst_path = REPORTS_DIR / "forecast_results.csv"
    if fcst_path.exists():
        df_fcst = pd.read_csv(fcst_path, parse_dates=["ds"])
        data["forecast"] = df_fcst
    else:
        data["forecast"] = pd.DataFrame()

    # ── Features (full dataset) ───────────────────────────────────────
    feat_path = DATA_DIR / "dataset_features.csv"
    if feat_path.exists():
        df_feat = pd.read_csv(feat_path, parse_dates=["fecha"])
        data["features"] = df_feat
    else:
        data["features"] = pd.DataFrame()

    # ── Consensus matrix ──────────────────────────────────────────────
    cons_path = REPORTS_DIR / "consensus_matrix.csv"
    if cons_path.exists():
        df_cons = pd.read_csv(cons_path, parse_dates=["fecha"])
        data["consensus"] = df_cons
    else:
        data["consensus"] = pd.DataFrame()

    data["loaded"] = all(
        not v.empty for k, v in data.items() if k != "loaded"
    )

    # Calcular KPIs base
    data["kpis"] = _calculate_kpis(data)
    return data


def _calculate_kpis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todos los KPIS a partir de los datos cargados."""
    kpis: Dict[str, Any] = {
        "caja_neta_actual": None,
        "caja_neta_prom_30d": None,
        "forecast_30d": None,
        "tendencia_mensual": None,
        "total_anomalies": 0,
        "anomalies_this_month": 0,
        "smape": None,
        "caja_neta_min": None,
        "caja_neta_max": None,
        "total_ingresos": None,
        "total_gastos": None,
        "dias_datos": 0,
    }

    df = data.get("powerbi")
    if df is None or df.empty:
        return kpis

    # Separar histórico y forecast
    historical = df[df["caja_neta"].notna()].copy()
    forecast = df[df["horizonte"].notna()].copy()

    if historical.empty:
        return kpis

    # Último valor real
    last_real = historical.sort_values("fecha").iloc[-1]
    kpis["caja_neta_actual"] = last_real["caja_neta"]
    kpis["dias_datos"] = len(historical)

    # Promedio 30d
    last_30 = historical.sort_values("fecha").tail(30)
    kpis["caja_neta_prom_30d"] = last_30["caja_neta"].mean()

    # Forecast 30d (suma)
    fcst_30 = forecast[forecast["horizonte"] == "30d"]["yhat"].sum()
    kpis["forecast_30d"] = fcst_30 if not pd.isna(fcst_30) else None

    # Tendencia mensual
    if len(historical) >= 60:
        last_month = historical.sort_values("fecha").tail(30)
        prev_month = historical.sort_values("fecha").tail(60).head(30)
        avg_last = last_month["caja_neta"].mean()
        avg_prev = prev_month["caja_neta"].mean()
        if avg_prev != 0:
            kpis["tendencia_mensual"] = ((avg_last - avg_prev) / abs(avg_prev)) * 100
        else:
            kpis["tendencia_mensual"] = 0.0
    else:
        kpis["tendencia_mensual"] = 0.0

    # Anomalías
    anom = data.get("anomalies")
    if anom is not None and not anom.empty:
        kpis["total_anomalies"] = int(anom["severidad"].notna().sum())
        # Este mes
        today = pd.Timestamp.now()
        month_start = today.replace(day=1)
        this_month = anom[
            (anom["fecha"] >= month_start) & (anom["fecha"] <= today)
        ]
        kpis["anomalies_this_month"] = int(this_month["severidad"].notna().sum())

    # SMAPE (estimar desde forecast histórico)
    if not historical.empty and "yhat" in historical.columns:
        hist_fcst = historical[historical["yhat"].notna()]
        if len(hist_fcst) > 10:
            y_t = hist_fcst["caja_neta"].values
            y_p = hist_fcst["yhat"].values
            denom = np.abs(y_t) + np.abs(y_p)
            mask = denom > 1e-12
            if mask.any():
                smape = float(
                    np.mean(2.0 * np.abs(y_t[mask] - y_p[mask]) / denom[mask]) * 100.0
                )
                kpis["smape"] = smape

    # Min/Max de caja neta
    kpis["caja_neta_min"] = historical["caja_neta"].min()
    kpis["caja_neta_max"] = historical["caja_neta"].max()

    # Totales
    if "ingreso_efectivo" in historical.columns:
        kpis["total_ingresos"] = (
            historical["ingreso_efectivo"].sum()
            + historical["ingreso_tarjeta"].sum()
            + historical["ingreso_transferencia"].sum()
        )
    if "gasto_operativo" in historical.columns:
        kpis["total_gastos"] = (
            historical["gasto_operativo"].sum()
            + historical.get("gasto_extraordinario", pd.Series(0)).sum()
        )

    return kpis


# ──────────────────────────────────────────────────────────────────────
# GRÁFICOS
# ──────────────────────────────────────────────────────────────────────


def create_forecast_chart(data: Dict[str, Any]) -> go.Figure:
    """Gráfico principal: serie temporal con forecast y bandas de confianza."""
    df = data.get("powerbi", pd.DataFrame())
    if df.empty:
        return _empty_figure("No hay datos disponibles")

    fig = go.Figure()

    # Separar histórico y forecast
    historical = df[df["caja_neta"].notna()].sort_values("fecha")
    forecast = df[df["horizonte"].notna()].sort_values("fecha")

    # IC 95%
    if not forecast.empty and "yhat_lower" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["fecha"], forecast["fecha"][::-1]]),
            y=pd.concat([
                forecast["yhat_upper"],
                forecast["yhat_lower"][::-1],
            ]),
            fill="toself",
            fillcolor="rgba(26, 35, 126, 0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 95%",
            hoverinfo="skip",
            legendrank=3,
        ))

    # IC 80%
    if not forecast.empty and "yhat_upper_80" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["fecha"], forecast["fecha"][::-1]]),
            y=pd.concat([
                forecast["yhat_upper_80"],
                forecast["yhat_lower_80"][::-1],
            ]),
            fill="toself",
            fillcolor="rgba(26, 35, 126, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 80%",
            hoverinfo="skip",
            legendrank=2,
        ))

    # Línea de forecast
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["fecha"],
            y=forecast["yhat"],
            mode="lines",
            line=dict(color=COLORS["red"], width=2, dash="dash"),
            name="Forecast",
            hovertemplate="%{x|%d %b %Y}<br><b>Forecast:</b> $%{y:,.0f}<extra></extra>",
            legendrank=1,
        ))

    # Línea histórica real
    fig.add_trace(go.Scatter(
        x=historical["fecha"],
        y=historical["caja_neta"],
        mode="lines",
        line=dict(color=COLORS["primary_light"], width=2),
        name="Caja Neta Real",
        hovertemplate="%{x|%d %b %Y}<br><b>Caja Neta:</b> $%{y:,.0f}<extra></extra>",
        legendrank=0,
    ))

    # Anomalías como marcadores
    anom = data.get("consensus", pd.DataFrame())
    if not anom.empty and "fecha" in anom.columns and "severidad" in anom.columns:
        anom_plot = anom[anom["severidad"].notna()].copy()
        if not anom_plot.empty:
            # Tomar caja_neta desde consensus_matrix si existe
            if "caja_neta" not in anom_plot.columns:
                anom_plot["caja_neta"] = float("nan")
            # Rellenar con historical donde sea posible
            hist_map = historical.set_index("fecha")["caja_neta"].to_dict()
            anom_plot["caja_neta"] = anom_plot["fecha"].map(hist_map).fillna(anom_plot["caja_neta"])
            # Si sigue sin datos, usar 0 (no ideal, pero evita crash)
            anom_plot["caja_neta"] = anom_plot["caja_neta"].fillna(0)
            for sev, color in SEVERIDAD_COLORS.items():
                subset = anom_plot[anom_plot["severidad"] == sev]
                if not subset.empty:
                    size = 10 if sev in ("crítico", "alto") else 6
                    symbol = "x" if sev == "crítico" else "circle"
                    fig.add_trace(go.Scatter(
                        x=subset["fecha"],
                        y=subset["caja_neta"],
                        mode="markers",
                        marker=dict(
                            color=color,
                            size=size,
                            symbol=symbol,
                            line=dict(color="white", width=1),
                        ),
                        name=f"Anomalía {sev}",
                        hovertemplate=(
                            "%{x|%d %b %Y}<br>"
                            f"<b>{sev.title()}</b><br>"
                            "Caja: $%{y:,.0f}<extra></extra>"
                        ),
                        legendrank=10,
                    ))

    # Línea divisoria presente
    if not forecast.empty:
        last_hist = historical["fecha"].max()
        fig.add_vline(
            x=pd.Timestamp(last_hist).timestamp() * 1000,
            line_dash="dash",
            line_color="rgba(144, 164, 174, 0.5)",
            annotation_text="  HOY",
            annotation_position="top right",
            annotation_font=dict(size=11, color=COLORS["grey"]),
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=12, r=12, t=8, b=8),
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="#f0f0f0",
            rangeslider=dict(visible=True, thickness=0.05),
            type="date",
        ),
        yaxis=dict(
            title="Caja Neta ($)",
            showgrid=True,
            gridcolor="#f0f0f0",
            tickformat="$,.0f",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_income_expense_chart(data: Dict[str, Any]) -> go.Figure:
    """Gráfico de ingresos vs gastos como áreas apiladas."""
    df = data.get("powerbi", pd.DataFrame())
    if df.empty:
        return _empty_figure("No hay datos")

    historical = df[df["caja_neta"].notna()].sort_values("fecha").copy()
    if historical.empty:
        return _empty_figure("No hay datos históricos")

    # Agrupar por mes
    historical["mes"] = historical["fecha"].dt.to_period("M").astype(str)
    monthly = historical.groupby("mes").agg({
        "ingreso_efectivo": "sum",
        "ingreso_tarjeta": "sum",
        "ingreso_transferencia": "sum",
        "gasto_operativo": "sum",
        "gasto_extraordinario": "sum",
    }).reset_index()

    monthly["total_ingresos"] = (
        monthly["ingreso_efectivo"]
        + monthly["ingreso_tarjeta"]
        + monthly["ingreso_transferencia"]
    )
    monthly["total_gastos"] = (
        monthly["gasto_operativo"] + monthly["gasto_extraordinario"]
    )
    monthly["resultado"] = monthly["total_ingresos"] - monthly["total_gastos"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly["mes"],
        y=monthly["total_ingresos"],
        name="Ingresos",
        marker_color=COLORS["green"],
        opacity=0.85,
        hovertemplate="%{x}<br>Ingresos: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=monthly["mes"],
        y=monthly["total_gastos"],
        name="Gastos",
        marker_color=COLORS["red"],
        opacity=0.85,
        hovertemplate="%{x}<br>Gastos: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=monthly["mes"],
        y=monthly["resultado"],
        mode="lines+markers",
        name="Resultado Neto",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6, color=COLORS["primary"]),
        hovertemplate="%{x}<br>Resultado: $%{y:,.0f}<extra></extra>",
        yaxis="y2",
    ))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        ),
        margin=dict(l=12, r=12, t=4, b=8),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(
            title="Monto ($)",
            showgrid=True,
            gridcolor="#f0f0f0",
            tickformat="$,.0f",
        ),
        yaxis2=dict(
            title="Resultado ($)",
            overlaying="y",
            side="right",
            showgrid=False,
            tickformat="$,.0f",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_anomaly_severity_chart(data: Dict[str, Any]) -> go.Figure:
    """Gráfico de distribución de anomalías por severidad."""
    df = data.get("anomalies", pd.DataFrame())
    if df.empty or "severidad" not in df.columns:
        return _empty_figure("Sin datos de anomalías")

    sev_counts = df["severidad"].value_counts()
    sev_order = ["crítico", "alto", "medio", "bajo"]
    sev_counts = sev_counts.reindex(
        [s for s in sev_order if s in sev_counts.index]
    )

    colors = [SEVERIDAD_COLORS.get(s, COLORS["grey"]) for s in sev_counts.index]

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=sev_counts.index,
        values=sev_counts.values,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=11),
        hole=0.5,
        hovertemplate="<b>%{label}</b><br>%{value} anomalías (%{percent})<extra></extra>",
        sort=False,
    ))

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=8, r=8, t=4, b=8),
        annotations=[
            dict(
                text=f"{sev_counts.sum():,}",
                x=0.5, y=0.5,
                font=dict(size=22, color=COLORS["text"]),
                showarrow=False,
            )
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def create_anomaly_timeline(data: Dict[str, Any]) -> go.Figure:
    """Heatmap de anomalías por mes y severidad."""
    df = data.get("anomalies", pd.DataFrame())
    if df.empty or "severidad" not in df.columns or "fecha" not in df.columns:
        return _empty_figure("Sin datos de anomalías")

    df = df[df["severidad"].notna()].copy()
    if df.empty:
        return _empty_figure("Sin anomalías detectadas")

    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["mes_num"] = df["fecha"].dt.month
    df["año"] = df["fecha"].dt.year

    # Tabla pivote: mes × severidad
    pivot = df.pivot_table(
        index="mes", columns="severidad", aggfunc="size", fill_value=0
    )
    sev_order = ["crítico", "alto", "medio", "bajo"]
    pivot = pivot[[s for s in sev_order if s in pivot.columns]]

    if pivot.empty:
        return _empty_figure("Sin datos para heatmap")

    z = pivot.values
    y_labels = pivot.index.tolist()
    x_labels = pivot.columns.tolist()

    colorscale = [
        [0.0, "#f5f5f5"],
        [0.25, "#e8eaf6"],
        [0.50, "#7986cb"],
        [0.75, "#3949ab"],
        [1.0, COLORS["primary"]],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        y=y_labels,
        x=x_labels,
        colorscale=colorscale,
        text=z,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} anomalías<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=8, r=8, t=4, b=8),
        xaxis=dict(title=None, side="top"),
        yaxis=dict(title=None, autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=200,
    )

    return fig


def create_top_alerts_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Tabla de top 10 anomalías más severas."""
    df = data.get("anomalies", pd.DataFrame())
    if df.empty or "severidad" not in df.columns:
        return pd.DataFrame()

    alerts = df[df["severidad"].notna()].copy()
    if alerts.empty:
        return pd.DataFrame()

    sev_order = {"crítico": 0, "alto": 1, "medio": 2, "bajo": 3}
    alerts["_sev_order"] = alerts["severidad"].map(sev_order)

    if "consenso_score" in alerts.columns:
        alerts = alerts.sort_values(["_sev_order", "consenso_score"])
    else:
        alerts = alerts.sort_values("_sev_order")

    top = alerts.head(10).copy()

    result = top[["fecha", "severidad", "metodos_detectores"]].copy()
    if "caja_neta" in top.columns:
        result["caja_neta"] = top["caja_neta"]
    elif "caja_neta" in data.get("powerbi", pd.DataFrame()).columns:
        # Merge con powerbi
        pbi = data["powerbi"][["fecha", "caja_neta"]]
        result = result.merge(pbi, on="fecha", how="left")

    if "consenso_score" in top.columns:
        result["score"] = top["consenso_score"]

    result["fecha"] = result["fecha"].dt.strftime("%d %b %Y")
    result["severidad"] = result["severidad"].str.title()
    if "metodos_detectores" in result.columns:
        result["métodos"] = result["metodos_detectores"]
        result = result.drop(columns=["metodos_detectores"])

    # Renombrar columnas
    cols_map = {
        "fecha": "Fecha",
        "severidad": "Severidad",
        "caja_neta": "Caja Neta",
        "score": "Score",
        "métodos": "Detectores",
    }
    result = result.rename(columns=cols_map)
    return result.reset_index(drop=True)


def create_forecast_components_chart(data: Dict[str, Any]) -> go.Figure:
    """Gráfico de descomposición del forecast (tendencia, estacionalidad)."""
    df = data.get("powerbi", pd.DataFrame())
    if df.empty:
        return _empty_figure("No hay datos")

    historical = df[df["caja_neta"].notna()].sort_values("fecha").copy()
    if historical.empty:
        return _empty_figure("No hay datos históricos")

    # Calcular tendencia simple (media móvil 30d) y estacionalidad semanal
    historical["trend"] = historical["caja_neta"].rolling(30, min_periods=1).mean()

    # Estacionalidad semanal: promedio por día de semana
    historical["weekday"] = historical["fecha"].dt.dayofweek
    weekday_avg = historical.groupby("weekday")["caja_neta"].mean()
    historical["weekly_seas"] = historical["weekday"].map(weekday_avg)
    historical["weekly_seas"] = historical["weekly_seas"] - weekday_avg.mean()

    # Residual
    historical["residual"] = (
        historical["caja_neta"] - historical["trend"] - historical["weekly_seas"]
    )

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Tendencia (Media Móvil 30d)", "Estacionalidad Semanal", "Residual"),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=historical["fecha"], y=historical["caja_neta"],
            mode="lines",
            line=dict(color="rgba(144, 164, 174, 0.4)", width=1),
            name="Caja Neta",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=historical["fecha"], y=historical["trend"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=2.5),
            name="Tendencia",
            hovertemplate="%{x|%d %b %Y}<br>Tendencia: $%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[ "Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
            y=weekday_avg.values,
            marker_color=COLORS["primary_light"],
            opacity=0.8,
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=historical["fecha"], y=historical["residual"],
            mode="markers",
            marker=dict(color=COLORS["teal"], size=3, opacity=0.6),
            name="Residual",
            hovertemplate="%{x|%d %b %Y}<br>Residual: $%{y:,.0f}<extra></extra>",
        ),
        row=3, col=1,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(144, 164, 174, 0.4)", row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=12, r=12, t=20, b=8),
        height=280,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(title=None, showgrid=True, gridcolor="#f0f0f0", row=3, col=1)
    fig.update_yaxes(title=None, showgrid=True, gridcolor="#f0f0f0")

    return fig


def create_forecast_accuracy_chart(data: Dict[str, Any]) -> go.Figure:
    """Dispersión de precisión: real vs predicho."""
    df = data.get("powerbi", pd.DataFrame())
    if df.empty:
        return _empty_figure("No hay datos")

    historical = df[df["caja_neta"].notna() & df["yhat"].notna()].copy()
    if historical.empty or len(historical) < 10:
        return _empty_figure("Datos insuficientes")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical["caja_neta"],
        y=historical["yhat"],
        mode="markers",
        marker=dict(
            color=COLORS["primary_light"],
            size=5,
            opacity=0.5,
            line=dict(color="white", width=0.5),
        ),
        name="Real vs Predicho",
        hovertemplate="Real: $%{x:,.0f}<br>Predicho: $%{y:,.0f}<extra></extra>",
    ))

    # Línea de referencia y=x
    min_val = min(historical["caja_neta"].min(), historical["yhat"].min())
    max_val = max(historical["caja_neta"].max(), historical["yhat"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["red"], dash="dash", width=1.5),
        name="Predicción Perfecta",
        hovertemplate="<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10),
        ),
        margin=dict(l=12, r=12, t=4, b=8),
        xaxis=dict(
            title="Real ($)", showgrid=True, gridcolor="#f0f0f0",
            tickformat="$,.0f",
        ),
        yaxis=dict(
            title="Predicho ($)", showgrid=True, gridcolor="#f0f0f0",
            tickformat="$,.0f",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def _empty_figure(msg: str = "Sin datos") -> go.Figure:
    """Figura vacía con mensaje."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["text_secondary"]),
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# HELPER — KPI Card
# ──────────────────────────────────────────────────────────────────────


def _kpi_card(
    label: str,
    value: str,
    change: str | None = None,
    change_type: str = "neutral",
    accent_color: str = "blue",
    value_size: str = "normal",
) -> html.Div:
    """Genera una tarjeta KPI con formato corporativo."""
    change_el = html.Div()
    if change is not None:
        arrow = "▲" if change_type == "positive" else "▼"
        cls = f"kpi-change {change_type}"
        change_el = html.Div(arrow + " " + change, className=cls)

    cls_value = "kpi-value" if value_size == "normal" else "kpi-value kpi-value-small"

    return html.Div([
        html.Div(className=f"kpi-accent {accent_color}"),
        html.Div(label, className="kpi-label"),
        html.Div(value, className=cls_value),
        change_el,
    ], className="kpi-card")


def format_currency(val: float | None) -> str:
    """Formatea un valor numérico como moneda."""
    if val is None or np.isnan(val):
        return "—"
    if abs(val) >= 1_000_000:
        return f"${val:,.0f}"
    elif abs(val) >= 1_000:
        return f"${val:,.0f}"
    else:
        return f"${val:,.2f}"


def format_pct(val: float | None) -> str:
    """Formatea un porcentaje."""
    if val is None or np.isnan(val):
        return "—"
    return f"{val:+.1f}%"


# ──────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ──────────────────────────────────────────────────────────────────────

# Cargar datos una vez al inicio
DATA = load_data()
KPIS = DATA["kpis"]
FECHA_GEN = datetime.now().strftime("%d %b %Y %H:%M")

# Inicializar app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Forecasting de Caja — Dashboard",
    suppress_callback_exceptions=True,
)

server = app.server

app.index_string = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forecasting de Caja — Dashboard</title>
    <link rel="stylesheet" href="/assets/style.css">
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>"""

# ── Construir KPIs ───────────────────────────────────────────────────
def _build_kpi_row(kpis: Dict) -> html.Div:
    """Construye la fila de 6 tarjetas KPI."""
    pct_kpi = kpis.get("tendencia_mensual")
    if pct_kpi is not None:
        change = format_pct(pct_kpi)
        change_type = "positive" if pct_kpi > 1 else "negative" if pct_kpi < -1 else "neutral"
    else:
        change = None
        change_type = "neutral"

    kpi_cards = html.Div([
        html.Div(
            _kpi_card(
                "Caja Neta Actual",
                format_currency(kpis.get("caja_neta_actual")),
                change,
                change_type,
                accent_color="blue",
            ),
            className="col-md-2 col-sm-4 col-6 mb-3",
        ),
        html.Div(
            _kpi_card(
                "Promedio 30 Días",
                format_currency(kpis.get("caja_neta_prom_30d")),
                f"Min: {format_currency(kpis.get('caja_neta_min'))}",
                "neutral",
                accent_color="blue",
            ),
            className="col-md-2 col-sm-4 col-6 mb-3",
        ),
        html.Div(
            _kpi_card(
                "Forecast 30 Días",
                format_currency(kpis.get("forecast_30d")),
                "Proyección acumulada",
                "neutral",
                accent_color="green",
            ),
            className="col-md-2 col-sm-4 col-6 mb-3",
        ),
        html.Div(
            _kpi_card(
                "Confianza (SMAPE)",
                f"{kpis.get('smape', 0):.1f}%" if kpis.get("smape") else "—",
                "Menor = mejor",
                "negative" if (kpis.get("smape") or 200) > 100 else "positive",
                accent_color="amber",
                value_size="small",
            ),
            className="col-md-2 col-sm-4 col-6 mb-3",
        ),
        html.Div(
            _kpi_card(
                "Anomalías (Total)",
                f"{kpis.get('total_anomalies', 0):,}",
                f"{kpis.get('anomalies_this_month', 0)} este mes",
                "neutral",
                accent_color="red",
            ),
            className="col-md-2 col-sm-4 col-6 mb-3",
        ),
        html.Div(
            _kpi_card(
                "Días de Datos",
                f"{kpis.get('dias_datos', 0)}",
                f"{'Con forecast' if kpis.get('forecast_30d') else 'Sin forecast'}",
                "neutral",
                accent_color="blue",
                value_size="small",
            ),
            className="col-md-2 col-sm-4 col-6 mb-3",
        ),
    ], className="row")

    return kpi_cards


# ── App Layout ───────────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.Div([
            html.Div("🏦 Dashboard de Caja", className="header-title"),
            html.Div("Forecasting Logístico · Mini-Modelo Prophet", className="header-subtitle"),
        ]),
        html.Div([
            html.Span(f"Actualizado: {FECHA_GEN}", className="header-badge"),
        ]),
    ], className="header"),

    # Contenido principal
    html.Div([

        # Fila de KPIs
        _build_kpi_row(KPIS),

        # Fila principal: gráfico grande
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Evolución de Caja Neta", className="chart-title"),
                    html.Div(id="forecast-chart", children=[
                        dcc.Graph(
                            figure=create_forecast_chart(DATA),
                            config={"responsive": True, "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                            style={"height": "420px"},
                        )
                    ]),
                ], className="chart-card"),
            ], className="col-12 mb-3"),
        ], className="row"),

        # Fila media: 2 columnas
        html.Div([
            # Columna izquierda: Ingresos vs Gastos
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Ingresos vs Gastos (Mensual)", className="chart-title"),
                        html.Span("Agrupado por mes", className="chart-title-badge"),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    dcc.Graph(
                        figure=create_income_expense_chart(DATA),
                        config={"responsive": True, "displayModeBar": False},
                        style={"height": "280px"},
                    ),
                ], className="chart-card"),
            ], className="col-md-7 col-12 mb-3"),

            # Columna derecha: Anomalías donut + distribución
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Distribución de Anomalías", className="chart-title"),
                        html.Span("Por severidad", className="chart-title-badge"),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    dcc.Graph(
                        figure=create_anomaly_severity_chart(DATA),
                        config={"responsive": True, "displayModeBar": False},
                        style={"height": "180px"},
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=create_anomaly_timeline(DATA),
                            config={"responsive": True, "displayModeBar": False},
                            style={"height": "100px"},
                        ),
                    ),
                ], className="chart-card"),
            ], className="col-md-5 col-12 mb-3"),
        ], className="row"),

        # Fila inferior: 2 columnas
        html.Div([
            # Columna izquierda: descomposición + precisión
            html.Div([
                html.Div([
                    html.Div("Descomposición del Forecast", className="chart-title"),
                    dcc.Graph(
                        figure=create_forecast_components_chart(DATA),
                        config={"responsive": True, "displayModeBar": False},
                        style={"height": "280px"},
                    ),
                ], className="chart-card mb-3"),
                html.Div([
                    html.Div([
                        html.Div("Precisión: Real vs Predicho", className="chart-title"),
                        html.Span("Cada punto es un día", className="chart-title-badge"),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    dcc.Graph(
                        figure=create_forecast_accuracy_chart(DATA),
                        config={"responsive": True, "displayModeBar": False},
                        style={"height": "240px"},
                    ),
                ], className="chart-card"),
            ], className="col-md-6 col-12 mb-3"),

            # Columna derecha: Top alertas
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Top 10 Alertas", className="chart-title"),
                        html.Span("Anomalías más severas", className="chart-title-badge"),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px"}),
                    html.Div(id="alerts-table"),
                ], className="chart-card"),
            ], className="col-md-6 col-12 mb-3"),
        ], className="row"),

    ], className="container-fluid", style={"padding": "20px 24px", "maxWidth": "1440px", "margin": "0 auto"}),

    # Footer
    html.Div([
        html.Span(
            "Mini-Modelo de Forecasting de Caja para Logística · "
            "Generado con Python + Prophet + Plotly · "
            f"SMAPE: {KPIS.get('smape', 0):.1f}%" if KPIS.get("smape") else "",
            style={"opacity": 0.6},
        ),
    ], className="footer"),

], style={"minHeight": "100vh", "background": COLORS["bg"]})


# ──────────────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────────────


@callback(
    Output("alerts-table", "children"),
    Input("forecast-chart", "id"),
)
def update_alerts_table(_) -> html.Div:
    """Genera la tabla de top 10 alertas."""
    alerts_df = create_top_alerts_table(DATA)

    if alerts_df.empty:
        return html.Div(
            "No hay anomalías detectadas",
            style={"color": COLORS["text_secondary"], "padding": "20px", "textAlign": "center"},
        )

    # Construir tabla HTML manual para control total de estilo
    rows = []
    for _, row in alerts_df.iterrows():
        sev = str(row.get("Severidad", "")).lower()
        color = SEVERIDAD_COLORS.get(sev, COLORS["grey"])
        caja = row.get("Caja Neta", "")
        if isinstance(caja, (int, float)):
            caja_fmt = f"${caja:,.0f}"
        else:
            caja_fmt = str(caja)

        score = row.get("Score", "")
        score_fmt = f"{score:.2f}" if isinstance(score, (int, float)) and not pd.isna(score) else ""

        rows.append(html.Tr([
            html.Td(str(row.get("Fecha", "")),
                    style={"fontSize": "12px", "color": COLORS["text"]}),
            html.Td(
                html.Span(str(row.get("Severidad", "")).title(),
                          style={
                              "background": color + "20",
                              "color": color,
                              "padding": "2px 8px",
                              "borderRadius": "10px",
                              "fontSize": "11px",
                              "fontWeight": 600,
                          }),
            ),
            html.Td(caja_fmt,
                    style={"fontSize": "12px", "fontWeight": 600, "color": COLORS["text"]}),
            html.Td(score_fmt,
                    style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
            html.Td(str(row.get("Detectores", "")),
                    style={"fontSize": "11px", "color": COLORS["text_secondary"], "maxWidth": "150px",
                           "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
        ], style={
            "borderBottom": f"1px solid {COLORS['border']}",
            "transition": "background 0.15s",
        }))

    table = html.Table(
        # Header
        html.Thead(html.Tr([
            html.Th("Fecha", style={"fontSize": "10px", "fontWeight": 600, "textTransform": "uppercase",
                                     "color": COLORS["text_secondary"], "padding": "8px 6px",
                                     "borderBottom": f"2px solid {COLORS['border']}"}),
            html.Th("Severidad", style={"fontSize": "10px", "fontWeight": 600, "textTransform": "uppercase",
                                        "color": COLORS["text_secondary"], "padding": "8px 6px",
                                        "borderBottom": f"2px solid {COLORS['border']}"}),
            html.Th("Caja Neta", style={"fontSize": "10px", "fontWeight": 600, "textTransform": "uppercase",
                                        "color": COLORS["text_secondary"], "padding": "8px 6px",
                                        "borderBottom": f"2px solid {COLORS['border']}"}),
            html.Th("Score", style={"fontSize": "10px", "fontWeight": 600, "textTransform": "uppercase",
                                    "color": COLORS["text_secondary"], "padding": "8px 6px",
                                    "borderBottom": f"2px solid {COLORS['border']}"}),
            html.Th("Detectores", style={"fontSize": "10px", "fontWeight": 600, "textTransform": "uppercase",
                                         "color": COLORS["text_secondary"], "padding": "8px 6px",
                                         "borderBottom": f"2px solid {COLORS['border']}"}),
        ])),
        # Body
        html.Tbody(rows),
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    return html.Div([
        table,
        html.Div(
            f"Mostrando {len(alerts_df)} de {KPIS.get('total_anomalies', 0)} anomalías totales",
            style={"fontSize": "10px", "color": COLORS["text_secondary"],
                   "marginTop": "8px", "textAlign": "right"},
        ),
    ], style={"overflowX": "auto"})


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Dashboard de Forecasting de Caja")
    logger.info("=" * 60)
    logger.info("Datos cargados: %s", "OK" if DATA["loaded"] else "PARCIAL")
    logger.info("KPIs calculados: %d métricas", len(KPIS))
    logger.info("")
    logger.info("Servidor iniciado en: http://127.0.0.1:8050")
    logger.info("Presiona CTRL+C para detener")
    logger.info("=" * 60)

    app.run(debug=False, host="127.0.0.1", port=8050)
