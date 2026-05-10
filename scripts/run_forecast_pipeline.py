"""Pipeline completo de forecasting — Fase 3.

Entrena modelo, genera predicciones, ejecuta backtesting y produce plots.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from src.forecasting.backtest import walk_forward_backtest
from src.forecasting.predict import generate_forecast
from src.forecasting.train import train_prophet_model
from src.forecasting.visualize import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    """Ejecuta el pipeline completo de forecasting."""
    # ── Cargar datos ─────────────────────────────────────────────────────
    print("=" * 60)
    print("CARGANDO DATOS REALES")
    print("=" * 60)
    df = pd.read_csv("data/curated/dataset_features.csv")
    print(f"Shape: {df.shape}")
    print(f"Rango fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    print(
        f"caja_neta — min: {df['caja_neta'].min():.2f}, "
        f"max: {df['caja_neta'].max():.2f}, "
        f"mean: {df['caja_neta'].mean():.2f}"
    )

    # ── 1. Entrenar modelo ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("ENTRENANDO MODELO PROPHET")
    print("=" * 60)
    model, forecast = train_prophet_model(df)

    # ── 2. Generar forecast a 90 días ────────────────────────────────────
    print()
    print("=" * 60)
    print("GENERANDO FORECAST")
    print("=" * 60)
    predictions = generate_forecast(model, periods=90)
    print(f"Predicciones generadas: {len(predictions)} filas")
    if not predictions.empty:
        print(f"Rango: {predictions['ds'].min()} a {predictions['ds'].max()}")
        print(
            f"Horizontes: {predictions['horizonte'].value_counts().to_dict()}"
        )
        print(f"Última predicción (90d): yhat={predictions['yhat'].iloc[-1]:.2f}")
        print(
            f"IC 80%: [{predictions['yhat_lower_80'].iloc[-1]:.2f}, "
            f"{predictions['yhat_upper_80'].iloc[-1]:.2f}]"
        )
        print(
            f"IC 95%: [{predictions['yhat_lower'].iloc[-1]:.2f}, "
            f"{predictions['yhat_upper'].iloc[-1]:.2f}]"
        )

    # ── 3. Walk-forward backtesting ──────────────────────────────────────
    print()
    print("=" * 60)
    print("WALK-FORWARD BACKTESTING")
    print("=" * 60)
    results = walk_forward_backtest(
        df, train_window=90, test_window=30, min_iterations=6
    )
    print(f"Iteraciones: {results['n_iterations']}")
    print(f"MAPE global: {results['global_mape']:.2f}%")
    print(f"MAE global: {results['global_mae']:.2f}")
    print(f"RMSE global: {results['global_rmse']:.2f}")
    print(f"Éxito (MAPE < 15%): {results['success']}")

    # ── 4. Generar plots ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GENERANDO PLOTS")
    print("=" * 60)
    try:
        anom_df = pd.read_csv("reports/anomaly_report.csv")
        print(f"Anomalías cargadas: {len(anom_df)} filas")
    except FileNotFoundError:
        anom_df = None
        print("No se encontró anomaly_report.csv — se omite plot de anomalías")

    # Unir forecast con valores reales
    full_forecast = forecast.copy()
    full_forecast["ds"] = pd.to_datetime(full_forecast["ds"])
    hist = df[["fecha", "caja_neta"]].copy()
    hist.columns = ["ds", "y"]
    hist["ds"] = pd.to_datetime(hist["ds"])
    full_forecast = full_forecast.merge(hist, on="ds", how="left")

    plots = generate_all_plots(full_forecast, model, anomalies_df=anom_df)
    print(f"Plots generados: {len(plots)}")
    for p in plots:
        print(f"  — {p}")

    print()
    print("=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
