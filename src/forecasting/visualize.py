"""
Visualización de resultados de forecasting con Plotly.

Genera 4+ gráficos HTML interactivos:
1. Forecast con intervalos de confianza
2. Componentes (tendencia, semanal, anual)
3. Análisis de residuos
4. Anomalías sobre el forecast

Uso:
    from src.forecasting.visualize import generate_all_plots
    generate_all_plots(forecast, model, df_anomalies)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from scipy import stats

from src.forecasting.predict import _interpolate_confidence

# Intentar importar plotly; si no está, se muestra advertencia
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# Paleta de colores
_COLOR_PRIMARY = "#1f77b4"
_COLOR_SECONDARY = "#ff7f0e"
_COLOR_HISTORICAL = "#1f77b4"
_COLOR_FORECAST = "#d62728"
_COLOR_CI_95 = "rgba(31, 119, 180, 0.15)"
_COLOR_CI_80 = "rgba(31, 119, 180, 0.30)"
_COLOR_ANOMALY_HIGH = "#d62728"
_COLOR_ANOMALY_MED = "#ff7f0e"
_COLOR_RESIDUAL = "#2ca02c"


def plot_forecast(
    forecast_df: pd.DataFrame,
    title: str = "Forecast de Caja Neta",
    output_path: str = "reports/forecast_plot.html",
) -> str:
    """Genera gráfico interactivo del forecast con intervalos de confianza.

    Args:
        forecast_df: DataFrame con columnas ds, yhat, yhat_lower, yhat_upper,
            yhat_lower_80, yhat_upper_80. Si faltan los IC 80%, se calculan.
        title: Título del gráfico.
        output_path: Ruta del archivo HTML de salida.

    Returns:
        Ruta del archivo HTML generado.
    """
    if not _HAS_PLOTLY:
        logger.warning("Plotly no está instalado — no se puede generar el gráfico")
        return ""

    df = forecast_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Si no tenemos IC 80%, calcularlos
    if "yhat_lower_80" not in df.columns or "yhat_upper_80" not in df.columns:
        lower_80, upper_80 = _interpolate_confidence(
            df["yhat"], df["yhat_lower"], df["yhat_upper"], target_alpha=0.20,
        )
        df["yhat_lower_80"] = lower_80
        df["yhat_upper_80"] = upper_80

    # Separar histórico (donde tenemos y real) y futuro
    has_y = "y" in df.columns and df["y"].notna().any()
    if has_y:
        historical = df[df["y"].notna()].copy()
    else:
        historical = pd.DataFrame(columns=df.columns)

    future = df[df["yhat_lower_80"].notna()].copy()

    fig = go.Figure()

    # IC 95% (más claro, más ancho)
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"][::-1]]),
        y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor=_COLOR_CI_95,
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="IC 95%",
    ))

    # IC 80% (más oscuro, más angosto)
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"][::-1]]),
        y=pd.concat([future["yhat_upper_80"], future["yhat_lower_80"][::-1]]),
        fill="toself",
        fillcolor=_COLOR_CI_80,
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="IC 80%",
    ))

    # Historical data
    if has_y and not historical.empty:
        fig.add_trace(go.Scatter(
            x=historical["ds"],
            y=historical["y"],
            mode="markers+lines",
            marker=dict(size=4, color=_COLOR_HISTORICAL),
            line=dict(color=_COLOR_HISTORICAL, width=1.5),
            name="Histórico",
            hovertemplate="%{x|%Y-%m-%d}<br>y: %{y:,.2f}<extra></extra>",
        ))

    # Predicción (línea central)
    fig.add_trace(go.Scatter(
        x=future["ds"],
        y=future["yhat"],
        mode="lines",
        line=dict(color=_COLOR_FORECAST, width=2.5),
        name="Predicción",
        hovertemplate="%{x|%Y-%m-%d}<br>yhat: %{y:,.2f}<extra></extra>",
    ))

    # Línea vertical divisoria entre histórico y futuro
    if has_y and not historical.empty:
        last_hist = historical["ds"].max()
        fig.add_vline(
            x=pd.Timestamp(last_hist).timestamp() * 1000,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="Presente",
            annotation_position="top right",
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Fecha", showgrid=True),
        yaxis=dict(title="Caja Neta ($)", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    _save_html(fig, output_path)
    return output_path


def plot_components(
    model: Prophet,
    forecast: pd.DataFrame,
    output_path: str = "reports/forecast_components.html",
) -> str:
    """Grafica los componentes del forecast (tendencia, weekly, yearly) con Plotly.

    Extrae las columnas de componentes del forecast DataFrame de Prophet
    y las grafica como subplots interactivos.

    Args:
        model: Modelo Prophet entrenado.
        forecast: DataFrame con predicciones (debe incluir componentes).
        output_path: Ruta del archivo HTML de salida.

    Returns:
        Ruta del archivo HTML generado.
    """
    if not _HAS_PLOTLY:
        logger.warning("Plotly no está instalado — no se puede generar el gráfico")
        return ""

    df = forecast.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Determinar qué componentes están disponibles
    component_cols = {
        "Tendencia": "trend" if "trend" in df.columns else None,
        "Semanal": "weekly" if "weekly" in df.columns else None,
        "Anual": "yearly" if "yearly" in df.columns else None,
    }

    available = [(name, col) for name, col in component_cols.items() if col is not None]

    if not available:
        logger.warning("No hay columnas de componentes en el forecast DataFrame")
        return ""

    n_plots = len(available)
    fig = make_subplots(
        rows=n_plots, cols=1,
        subplot_titles=[name for name, _ in available],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    colors = [_COLOR_PRIMARY, _COLOR_SECONDARY, _COLOR_HISTORICAL]

    for i, (name, col) in enumerate(available, start=1):
        color = colors[(i - 1) % len(colors)]

        # Para weekly, mostrar patrón semanal (no serie temporal completa)
        if col == "weekly" and "ds_weekday" not in df.columns:
            # Prophet devuelve 'weekly' como el efecto del día de semana
            # Graficamos contra ds (serie temporal)
            fig.add_trace(
                go.Scatter(
                    x=df["ds"],
                    y=df[col],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=name,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
                ),
                row=i, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["ds"],
                    y=df[col],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=name,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
                ),
                row=i, col=1,
            )

        fig.update_yaxes(title_text="Efecto ($)", row=i, col=1)

    fig.update_xaxes(title_text="Fecha", row=n_plots, col=1)

    fig.update_layout(
        title=dict(text="Componentes del Forecast", x=0.5),
        template="plotly_white",
        showlegend=False,
        height=250 * n_plots,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    _save_html(fig, output_path)
    return output_path


def plot_residuals(
    forecast_df: pd.DataFrame,
    output_path: str = "reports/forecast_residuals.html",
) -> str:
    """Genera gráfico interactivo del análisis de residuos.

    Incluye 3 subplots:
    a. Residuos vs tiempo (scatter + línea horizontal en 0)
    b. Histograma de residuos (distribución)
    c. Q-Q plot de residuos

    Args:
        forecast_df: DataFrame con columnas ds, y, yhat.
        output_path: Ruta del archivo HTML de salida.

    Returns:
        Ruta del archivo HTML generado.
    """
    if not _HAS_PLOTLY:
        logger.warning("Plotly no está instalado — no se puede generar el gráfico")
        return ""

    df = forecast_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Verificar que tengamos valores reales
    if "y" not in df.columns or df["y"].isna().all():
        logger.warning("No hay valores reales (columna 'y') para calcular residuos")
        return ""

    # Filtrar filas con valor real
    hist = df[df["y"].notna()].copy()
    if hist.empty:
        logger.warning("No hay filas históricas para residuos")
        return ""

    hist["residuo"] = hist["y"] - hist["yhat"]

    residuos = hist["residuo"]
    mean_r = float(residuos.mean())
    std_r = float(residuos.std())

    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuos vs Tiempo",
            "Histograma de Residuos",
            "Q-Q Plot de Residuos",
        ),
        column_widths=[0.6, 0.4],
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # (a) Residuos vs tiempo
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=residuos,
            mode="markers",
            marker=dict(color=_COLOR_RESIDUAL, size=4, opacity=0.7),
            name="Residuo",
            hovertemplate="%{x|%Y-%m-%d}<br>Residuo: %{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Línea horizontal en 0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        row=1, col=1,
    )

    # (b) Histograma
    fig.add_trace(
        go.Histogram(
            x=residuos,
            nbinsx=30,
            marker=dict(color=_COLOR_RESIDUAL, line=dict(color="white", width=0.5)),
            name="Distribución",
            hovertemplate="Rango: %{x}<br>Frecuencia: %{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    # (c) Q-Q plot
    # Calcular quantiles teóricos vs observados
    n = len(residuos)
    theoretical_quantiles = stats.norm.ppf(
        np.linspace(0.01, 0.99, min(n, 100))
    )
    sorted_resid = np.sort(residuos)
    sample_quantiles = np.quantile(
        sorted_resid,
        np.linspace(0.01, 0.99, min(n, 100)),
    )

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode="markers",
            marker=dict(color=_COLOR_RESIDUAL, size=4, opacity=0.7),
            name="Q-Q",
            hovertemplate="Teórico: %{x:.2f}<br>Observado: %{y:.2f}<extra></extra>",
        ),
        row=2, col=2,
    )

    # Línea diagonal de referencia (y = x)
    qq_min = min(theoretical_quantiles.min(), sample_quantiles.min())
    qq_max = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min, qq_max],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Referencia",
            hoverinfo="skip",
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"Análisis de Residuos (μ={mean_r:,.2f}, σ={std_r:,.2f})",
            x=0.5,
        ),
        template="plotly_white",
        showlegend=False,
        height=600,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    fig.update_xaxes(title_text="Fecha", row=1, col=1)
    fig.update_yaxes(title_text="Residuo ($)", row=1, col=1)
    fig.update_xaxes(title_text="Residuo ($)", row=2, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
    fig.update_xaxes(title_text="Quantil teórico", row=2, col=2)
    fig.update_yaxes(title_text="Quantil observado", row=2, col=2)

    _save_html(fig, output_path)
    return output_path


def plot_anomalies_forecast(
    forecast_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    output_path: str = "reports/forecast_anomalies.html",
) -> str:
    """Superpone las anomalías detectadas sobre el forecast histórico.

    Args:
        forecast_df: DataFrame con columnas ds, y, yhat.
        anomalies_df: DataFrame con columna 'fecha' y 'severidad' (o 'tipo_anomalia').
        output_path: Ruta del archivo HTML de salida.

    Returns:
        Ruta del archivo HTML generado.
    """
    if not _HAS_PLOTLY:
        logger.warning("Plotly no está instalado — no se puede generar el gráfico")
        return ""

    df = forecast_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    if anomalies_df is None or anomalies_df.empty:
        logger.warning("No hay anomalías para graficar")
        return ""

    # Buscar columna de severidad/tipo
    sev_col = None
    for col in ["severidad", "tipo_anomalia", "tipo"]:
        if col in anomalies_df.columns:
            sev_col = col
            break

    # Buscar columna de fecha
    date_col = None
    for col in ["fecha", "ds"]:
        if col in anomalies_df.columns:
            date_col = col
            break

    if date_col is None:
        logger.warning("No se encontró columna de fecha en anomalies_df")
        return ""

    anom = anomalies_df.copy()
    anom[date_col] = pd.to_datetime(anom[date_col])

    # Clasificar por severidad
    if sev_col:
        high_sev = anom[anom[sev_col].str.lower().isin(["crítico", "critico", "alto", "high"])]
        med_sev = anom[anom[sev_col].str.lower().isin(["medio", "medium", "bajo", "low"])]
    else:
        high_sev = anom
        med_sev = pd.DataFrame(columns=anom.columns)

    fig = go.Figure()

    # Datos históricos
    has_y = "y" in df.columns
    if has_y:
        hist = df[df["y"].notna()]
        fig.add_trace(go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            mode="lines",
            line=dict(color=_COLOR_HISTORICAL, width=1.5),
            name="Caja Neta (histórico)",
            hovertemplate="%{x|%Y-%m-%d}<br>y: %{y:,.2f}<extra></extra>",
        ))

    # Forecast (yhat)
    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=df["yhat"],
        mode="lines",
        line=dict(color=_COLOR_FORECAST, width=1.5, dash="dot"),
        name="Predicción",
        hovertemplate="%{x|%Y-%m-%d}<br>yhat: %{y:,.2f}<extra></extra>",
    ))

    # Anomalías de severidad alta/crítica
    if not high_sev.empty:
        fig.add_trace(go.Scatter(
            x=high_sev[date_col],
            y=high_sev.get("caja_neta", high_sev.get("saldo_diario", [0] * len(high_sev))),
            mode="markers",
            marker=dict(
                color=_COLOR_ANOMALY_HIGH,
                size=10,
                symbol="x",
                line=dict(color="darkred", width=1),
            ),
            name="Anomalía crítica/alta",
            hovertemplate="%{x|%Y-%m-%d}<br>Severidad: %{customdata}<extra></extra>",
            customdata=high_sev.get(sev_col, [""] * len(high_sev)) if sev_col else None,
        ))

    # Anomalías de severidad media/baja
    if not med_sev.empty:
        fig.add_trace(go.Scatter(
            x=med_sev[date_col],
            y=med_sev.get("caja_neta", med_sev.get("saldo_diario", [0] * len(med_sev))),
            mode="markers",
            marker=dict(
                color=_COLOR_ANOMALY_MED,
                size=8,
                symbol="circle",
                line=dict(color="darkorange", width=1),
            ),
            name="Anomalía media/baja",
            hovertemplate="%{x|%Y-%m-%d}<br>Severidad: %{customdata}<extra></extra>",
            customdata=med_sev.get(sev_col, [""] * len(med_sev)) if sev_col else None,
        ))

    fig.update_layout(
        title=dict(text="Anomalías sobre Forecast de Caja Neta", x=0.5),
        xaxis=dict(title="Fecha", showgrid=True),
        yaxis=dict(title="Caja Neta ($)", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
    )

    _save_html(fig, output_path)
    return output_path


def generate_all_plots(
    forecast_df: pd.DataFrame,
    model: Prophet,
    anomalies_df: pd.DataFrame | None = None,
    output_dir: str = "reports",
) -> List[str]:
    """Ejecuta todos los plots de forecasting.

    Args:
        forecast_df: DataFrame con predicciones.
        model: Modelo Prophet entrenado.
        anomalies_df: DataFrame con anomalías detectadas (opcional).
        output_dir: Directorio para guardar los archivos HTML.

    Returns:
        Lista de rutas de archivos generados.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths: List[str] = []

    # 1. Forecast
    p1 = plot_forecast(
        forecast_df,
        output_path=os.path.join(output_dir, "forecast_plot.html"),
    )
    if p1:
        paths.append(p1)

    # 2. Componentes
    p2 = plot_components(
        model,
        forecast_df,
        output_path=os.path.join(output_dir, "forecast_components.html"),
    )
    if p2:
        paths.append(p2)

    # 3. Residuos
    p3 = plot_residuals(
        forecast_df,
        output_path=os.path.join(output_dir, "forecast_residuals.html"),
    )
    if p3:
        paths.append(p3)

    # 4. Anomalías (si hay datos)
    if anomalies_df is not None and not anomalies_df.empty:
        p4 = plot_anomalies_forecast(
            forecast_df,
            anomalies_df,
            output_path=os.path.join(output_dir, "forecast_anomalies.html"),
        )
        if p4:
            paths.append(p4)

    logger.info("Gráficos generados: %s", paths)
    return paths


def _save_html(fig: "go.Figure", path: str) -> None:
    """Guarda una figura Plotly como HTML interactivo.

    Args:
        fig: Figura de Plotly.
        path: Ruta del archivo de salida.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
    )
    logger.info("Gráfico guardado: %s", path)
