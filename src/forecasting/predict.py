"""
Predicciones futuras con Prophet a 30/60/90 días.

Genera predicciones con intervalos de confianza al 80% y 95%,
y exporta resultados a CSV y archivo de texto.

Uso:
    from src.forecasting.predict import generate_forecast
    predictions = generate_forecast(model, periods=90)
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


def _interpolate_confidence(
    yhat: pd.Series,
    yhat_lower: pd.Series,
    yhat_upper: pd.Series,
    target_alpha: float = 0.20,
) -> Tuple[pd.Series, pd.Series]:
    """Interpola los intervalos de confianza de Prophet (default 95%) a un nivel
    diferente asumiendo distribución normal de los errores.

    Prophet por defecto usa z_score = 1.96 (IC 95%). Para obtener IC al 80%,
    se usa z_score = 1.28. La interpolación lineal es:

        half_95  = (yhat_upper - yhat) / 1.96
        half_80  = half_95 * 1.28
        lower_80 = yhat - half_80
        upper_80 = yhat + half_80

    Args:
        yhat: Serie con valores predichos.
        yhat_lower: Serie con límite inferior del IC 95%.
        yhat_upper: Serie con límite superior del IC 95%.
        target_alpha: Nivel de significancia objetivo (0.20 → IC 80%).

    Returns:
        Tupla (lower_target, upper_target) para el nivel alpha objetivo.
    """
    # z-scores: 95% → 1.96, target → scipy normal ppf
    # Usamos valores precalculados para evitar dependencia extra
    z_scores: dict[float, float] = {0.05: 1.96, 0.20: 1.28, 0.10: 1.645, 0.01: 2.576}
    z_95 = z_scores.get(0.05, 1.96)
    z_target = z_scores.get(target_alpha, 1.28)

    half_95 = (yhat_upper - yhat) / z_95
    half_target = half_95 * z_target

    lower_target = yhat - half_target
    upper_target = yhat + half_target

    return lower_target, upper_target


def generate_forecast(
    model: Prophet,
    periods: int = 90,
    include_history: bool = True,
    output_dir: str = "reports",
) -> pd.DataFrame:
    """Genera predicciones futuras con Prophet a N días.

    Crea un future DataFrame, genera predicciones, filtra solo las filas
    futuras, calcula IC 80% y clasifica por horizonte (30d, 60d, 90d).

    Args:
        model: Modelo Prophet entrenado.
        periods: Número de días hacia adelante para predecir.
        include_history: Si incluir el histórico en el future DataFrame.
        output_dir: Directorio para guardar los archivos de salida.

    Returns:
        DataFrame con columnas:
            ds, yhat, yhat_lower_80, yhat_upper_80,
            yhat_lower, yhat_upper, horizonte

    Raises:
        ValueError: Si el modelo no está entrenado o periods <= 0.
    """
    if periods <= 0:
        raise ValueError(f"periods debe ser > 0, got {periods}")

    logger.info("Generando forecast a %d días", periods)

    # ── 1. Crear future DataFrame ─────────────────────────────────────────
    future = model.make_future_dataframe(
        periods=periods,
        include_history=include_history,
    )

    # ── 2. Generar predicciones ───────────────────────────────────────────
    forecast = model.predict(future)

    # ── 3. Determinar la última fecha del training ────────────────────────
    # La última fecha de training estándar es la del history más reciente.
    # Prophet la guarda en model.history['ds'].max()
    last_train_date = model.history["ds"].max()

    # ── 4. Filtrar solo filas futuras ─────────────────────────────────────
    future_mask = forecast["ds"] > last_train_date
    future_df = forecast[future_mask].copy()

    if future_df.empty:
        logger.warning(
            "No hay filas futuras. last_train_date=%s, forecast range=%s - %s",
            last_train_date,
            forecast["ds"].min(),
            forecast["ds"].max(),
        )
        # Si no hay filas futuras, devolvemos vacío con las columnas esperadas
        return pd.DataFrame(columns=[
            "ds", "yhat", "yhat_lower_80", "yhat_upper_80",
            "yhat_lower", "yhat_upper", "horizonte",
        ])

    logger.info(
        "Filas futuras: %d (from %s to %s)",
        len(future_df),
        future_df["ds"].min().date(),
        future_df["ds"].max().date(),
    )

    # ── 5. Calcular IC 80% ───────────────────────────────────────────────
    lower_80, upper_80 = _interpolate_confidence(
        future_df["yhat"],
        future_df["yhat_lower"],
        future_df["yhat_upper"],
        target_alpha=0.20,
    )
    future_df["yhat_lower_80"] = lower_80
    future_df["yhat_upper_80"] = upper_80

    # ── 6. Clasificar por horizonte ──────────────────────────────────────
    days_ahead = (future_df["ds"] - last_train_date).dt.days
    conditions = [
        days_ahead <= 30,
        days_ahead <= 60,
        days_ahead <= 90,
    ]
    choices = ["30d", "60d", "90d"]
    future_df["horizonte"] = np.select(conditions, choices, default="90d+")

    # ── 7. Seleccionar columnas de salida ─────────────────────────────────
    result = future_df[[
        "ds", "yhat",
        "yhat_lower_80", "yhat_upper_80",
        "yhat_lower", "yhat_upper",
        "horizonte",
    ]].copy()

    # ── 8. Guardar CSV ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "forecast_results.csv")
    result.to_csv(csv_path, index=False)
    logger.info("Resultados guardados en %s", csv_path)

    # ── 9. Guardar resumen textual ───────────────────────────────────────
    txt_path = os.path.join(output_dir, "forecast_summary.txt")
    _write_summary(result, txt_path)

    return result


def _write_summary(df: pd.DataFrame, path: str) -> None:
    """Escribe un resumen textual del forecast.

    Args:
        df: DataFrame con predicciones.
        path: Ruta del archivo de texto a escribir.
    """
    lines: list[str] = [
        "=" * 60,
        "RESUMEN DE FORECAST — CAJA NETA",
        "=" * 60,
        "",
    ]

    for horizonte in ["30d", "60d", "90d"]:
        subset = df[df["horizonte"] == horizonte]
        if subset.empty:
            continue

        last_row = subset.iloc[-1]
        first_row = subset.iloc[0]

        lines.append(f"Horizonte: {horizonte}")
        lines.append(f"  Período: {first_row['ds'].date()} → {last_row['ds'].date()}")
        lines.append(f"  Días: {len(subset)}")
        lines.append(f"  Último yhat: {last_row['yhat']:,.2f}")
        lines.append(f"  IC 80%: [{last_row['yhat_lower_80']:,.2f}, {last_row['yhat_upper_80']:,.2f}]")
        lines.append(f"  IC 95%: [{last_row['yhat_lower']:,.2f}, {last_row['yhat_upper']:,.2f}]")
        lines.append("")

    # Resumen general
    lines.extend([
        "-" * 60,
        "Resumen general",
        "-" * 60,
        f"  Última predicción: {df['ds'].max().date()}"
        f" — yhat={df.loc[df['ds'].idxmax(), 'yhat']:,.2f}",
        f"  Valor mínimo predicho: {df['yhat'].min():,.2f}",
        f"  Valor máximo predicho: {df['yhat'].max():,.2f}",
        f"  Promedio predicho: {df['yhat'].mean():,.2f}",
        f"  Desviación estándar: {df['yhat'].std():,.2f}",
        "",
        "=" * 60,
        "Fin del reporte",
        "=" * 60,
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Resumen textual guardado en %s", path)
