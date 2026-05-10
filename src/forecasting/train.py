"""
Entrenamiento del modelo Prophet para forecasting de caja neta.

Parámetros:
- seasonality_mode='additive'
- changepoint_prior_scale=0.05
- weekly_seasonality=True (modo 'additive')
- yearly_seasonality=True (modo 'additive')
- Festivos mexicanos como regresores adicionales

Uso:
    from src.forecasting.train import train_prophet_model, get_mexican_holidays
    model, forecast = train_prophet_model(df)
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import date, timedelta
from typing import Any, Tuple

import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Festivos mexicanos fijos
# ---------------------------------------------------------------------------
_FESTIVOS_FIJOS: dict[str, tuple[int, int]] = {
    "Año Nuevo": (1, 1),
    "Día de la Constitución": (2, 5),  # primer lunes de febrero — usamos 5-feb como approx
    "Natalicio de Benito Juárez": (3, 21),  # tercer lunes de marzo — usamos 21-mar como approx
    "Día del Trabajo": (5, 1),
    "Día de la Independencia": (9, 16),
    "Día de la Revolución": (11, 20),  # tercer lunes de noviembre — usamos 20-nov como approx
    "Día de la Virgen de Guadalupe": (12, 12),
    "Navidad": (12, 25),
    "Día de Reyes": (1, 6),
    "Día de Muertos": (11, 2),
}
"""Festivos mexicanos con fecha fija (mes, día).

NOTA: Día de la Constitución, Natalicio de Benito Juárez y Día de la Revolución
son oficialmente el primer/tercer lunes del mes. Usamos la fecha fija como
aproximación para el regresor festivo.
"""


def _calcular_semana_santa(anio: int) -> list[dict[str, Any]]:
    """Calcula las fechas de Jueves Santo y Viernes Santo para un año dado.

    Usa el algoritmo de Computus para calcular el Domingo de Pascua y
    retrocede 3 y 2 días respectivamente.

    Args:
        anio: Año para el cual calcular Semana Santa.

    Returns:
        Lista de dicts con {'ds': datetime.date, 'holiday': str}.
    """
    # Algoritmo de Computus (Meeus/Gauss)
    a = anio % 19
    b = anio // 100
    c = anio % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1

    pascua = date(anio, mes, dia)

    # Jueves Santo = Pascua - 3 días
    # Viernes Santo = Pascua - 2 días
    jueves_santo = pascua - timedelta(days=3)
    viernes_santo = pascua - timedelta(days=2)

    return [
        {"ds": jueves_santo, "holiday": "Jueves Santo"},
        {"ds": viernes_santo, "holiday": "Viernes Santo"},
    ]


def get_mexican_holidays() -> pd.DataFrame:
    """Genera DataFrame con los festivos mexicanos para 2024 y 2025.

    Incluye 12 festivos por año:
    - Fijos: Año Nuevo, Día de la Constitución, Natalicio de Benito Juárez,
      Día del Trabajo, Día de la Independencia, Día de la Revolución,
      Día de la Virgen de Guadalupe, Navidad, Día de Reyes, Día de Muertos
    - Variables: Jueves Santo, Viernes Santo (calculados con algoritmo de Computus)

    Returns:
        DataFrame con columnas 'ds' (datetime) y 'holiday' (str).
    """
    registros: list[dict[str, Any]] = []

    for anio in (2024, 2025):
        # Festivos fijos
        for nombre, (mes, dia) in _FESTIVOS_FIJOS.items():
            registros.append({
                "ds": date(anio, mes, dia),
                "holiday": nombre,
            })

        # Semana Santa variable
        registros.extend(_calcular_semana_santa(anio))

    df = pd.DataFrame(registros)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    logger.info("Festivos mexicanos generados: %d registros", len(df))
    return df


def add_holiday_regressors(
    model: Prophet, holidays_df: pd.DataFrame,
) -> Prophet:
    """Añade los festivos como regresores al modelo Prophet.

    Args:
        model: Modelo Prophet sin regresores de festivos.
        holidays_df: DataFrame con columnas 'ds' y 'holiday'.

    Returns:
        Modelo Prophet con los festivos configurados.
    """
    model.holidays = holidays_df
    logger.info("Regresores de festivos añadidos: %d festivos", len(holidays_df))
    return model


def train_prophet_model(
    df: pd.DataFrame,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    save_path: str | None = "models/prophet_model.pkl",
) -> Tuple[Prophet, pd.DataFrame]:
    """Entrena un modelo Prophet para forecasting de caja neta.

    Pipeline:
        1. Renombra columnas 'fecha' → 'ds' y 'caja_neta' → 'y'.
        2. Obtiene festivos mexicanos y los añade como regresores.
        3. Configura y entrena el modelo Prophet.
        4. Genera predicciones sobre los mismos datos (para ver residuos).
        5. Guarda el modelo con pickle.

    Args:
        df: DataFrame con columnas 'fecha' y 'caja_neta'.
        seasonality_mode: Modo de estacionalidad ('additive' o 'multiplicative').
        changepoint_prior_scale: Escala de sensibilidad a puntos de cambio.
        weekly_seasonality: Si incluir estacionalidad semanal.
        yearly_seasonality: Si incluir estacionalidad anual.
        save_path: Ruta para guardar el modelo pickle. None para no guardar.

    Returns:
        Tupla (modelo Prophet entrenado, DataFrame con predicciones).

    Raises:
        ValueError: Si el DataFrame está vacío o faltan columnas requeridas.
    """
    required_cols = {"fecha", "caja_neta"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas requeridas faltantes: {missing}"
        )

    if df.empty:
        raise ValueError("DataFrame vacío — no se puede entrenar el modelo")

    logger.info("Iniciando entrenamiento de Prophet con %d filas", len(df))

    # ── 1. Preparar datos ─────────────────────────────────────────────────
    train_df = df[["fecha", "caja_neta"]].copy()
    train_df = train_df.rename(columns={"fecha": "ds", "caja_neta": "y"})
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    train_df = train_df.sort_values("ds").reset_index(drop=True)

    logger.info(
        "Datos preparados: rango %s – %s, %d filas",
        train_df["ds"].min().date(),
        train_df["ds"].max().date(),
        len(train_df),
    )

    # ── 2. Festivos mexicanos ─────────────────────────────────────────────
    holidays_df = get_mexican_holidays()

    # ── 3. Configurar modelo ──────────────────────────────────────────────
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        mcmc_samples=0,  # seed determinista
    )

    model = add_holiday_regressors(model, holidays_df)

    logger.info(
        "Modelo configurado: seasonality_mode=%s, "
        "changepoint_prior_scale=%s, weekly=%s, yearly=%s",
        seasonality_mode,
        changepoint_prior_scale,
        weekly_seasonality,
        yearly_seasonality,
    )

    # ── 4. Entrenar ──────────────────────────────────────────────────────
    model.fit(train_df)

    # ── 5. Predecir sobre los mismos datos ───────────────────────────────
    future = model.make_future_dataframe(periods=0, include_history=True)
    forecast = model.predict(future)

    # Calcular métricas de entrenamiento
    merged = train_df.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")
    mae = float((merged["y"] - merged["yhat"]).abs().mean())
    rmse = float(((merged["y"] - merged["yhat"]) ** 2).mean() ** 0.5)

    # MAPE (solo referencia — no fiable con valores que cruzan cero)
    mape_mask = merged["y"].abs() > 1e-6
    if mape_mask.any():
        mape = float(
            ((merged.loc[mape_mask, "y"] - merged.loc[mape_mask, "yhat"]).abs()
             / merged.loc[mape_mask, "y"].abs()).mean() * 100
        )
    else:
        mape = 0.0

    # SMAPE (métrica principal — funciona con valores que cruzan cero)
    denom = merged["y"].abs() + merged["yhat"].abs()
    smape_mask = denom > 1e-12
    if smape_mask.any():
        smape = float(
            (2.0 * (merged.loc[smape_mask, "y"] - merged.loc[smape_mask, "yhat"]).abs()
             / denom[smape_mask]).mean() * 100
        )
    else:
        smape = 0.0

    logger.info(
        "Entrenamiento completado — MAE: %.2f, RMSE: %.2f, MAPE: %.2f%%, SMAPE: %.2f%%",
        mae, rmse, mape, smape,
    )

    # ── 6. Guardar modelo ─────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Modelo guardado en %s", save_path)

    return model, forecast
