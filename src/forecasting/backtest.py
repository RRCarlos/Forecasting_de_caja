"""
Walk-forward backtesting para validación temporal.

Divide los datos en ventanas de entrenamiento (90 días) y prueba (30 días)
con deslizamiento. Calcula MAE, RMSE, MAPE para cada ventana.

Criterio de éxito: MAPE global < 15%

Uso:
    from src.forecasting.backtest import walk_forward_backtest
    results = walk_forward_backtest(df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from prophet import Prophet

from src.forecasting.train import get_mexican_holidays, add_holiday_regressors

logger = logging.getLogger(__name__)


def _calculate_metrics(
    y_true: pd.Series, y_pred: pd.Series,
) -> Dict[str, float]:
    """Calcula métricas de error: MAE, RMSE, MAPE, SMAPE.

    SMAPE (Symmetric Mean Absolute Percentage Error) es la métrica
    recomendada para datos que cruzan por cero como flujo de caja.
    A diferencia de MAPE, SMAPE usa el promedio de |real| + |predicho|
    como denominador, lo que la hace simétrica y acotada entre 0% y 200%.

    MAPE se incluye como referencia pero NO debe usarse como criterio
    de éxito cuando los valores pueden ser negativos o cercanos a cero.

    Args:
        y_true: Serie con valores reales.
        y_pred: Serie con valores predichos.

    Returns:
        Dict con 'mae', 'rmse', 'mape', 'smape'.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # MAE
    mae = float(np.mean(np.abs(y_t - y_p)))

    # RMSE
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # MAPE: excluir puntos donde y_true == 0 o muy cercano a 0
    # ADVERTENCIA: MAPE no es fiable cuando y_true cruza por cero
    mask_mape = np.abs(y_t) > 1e-6
    if mask_mape.any():
        mape = float(
            np.mean(np.abs((y_t[mask_mape] - y_p[mask_mape]) / y_t[mask_mape])) * 100.0
        )
    else:
        mape = 0.0

    # SMAPE: simétrico, funciona con valores positivos, negativos y cero
    # SMAPE = 200 * mean(|y_t - y_p| / (|y_t| + |y_p|))
    # Rango: [0, 200]. Excluye puntos donde ambos son cero (predicción perfecta).
    denom = np.abs(y_t) + np.abs(y_p)
    mask_smape = denom > 1e-12  # evitar division por cero solo cuando ambos son ~0
    if mask_smape.any():
        smape = float(
            np.mean(2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]) * 100.0
        )
    else:
        smape = 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


def walk_forward_backtest(
    df: pd.DataFrame,
    train_window: int = 90,
    test_window: int = 30,
    min_iterations: int = 6,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
) -> Dict[str, Any]:
    """Ejecuta walk-forward backtesting para validar el modelo Prophet.

    Divide los datos cronológicamente en ventanas deslizantes de
    entrenamiento (train_window días) y prueba (test_window días).
    En cada iteración entrena Prophet y evalúa contra los valores reales.

    La métrica principal de éxito es SMAPE (Symmetric MAPE), que funciona
    correctamente con datos que cruzan por cero como flujo de caja.
    MAPE se incluye solo como referencia diagnóstica.

    Args:
        df: DataFrame con columnas 'fecha' y 'caja_neta'.
        train_window: Días de entrenamiento por ventana.
        test_window: Días de prueba por ventana.
        min_iterations: Número mínimo de iteraciones requeridas.
        seasonality_mode: Modo de estacionalidad para Prophet.
        changepoint_prior_scale: Sensibilidad a puntos de cambio.

    Returns:
        Dict con:
            - iterations: list[dict] — métricas por iteración
            - global_mae: float
            - global_rmse: float
            - global_mape: float (solo referencia — no fiable con valores que cruzan cero)
            - global_smape: float (métrica principal — funciona con ceros y negativos)
            - mape_per_iteration: list[float]
            - smape_per_iteration: list[float]
            - n_iterations: int
            - train_window: int
            - test_window: int
            - success_by_smape: bool (True si SMAPE global < 15%)
            - success_by_mae: bool (True si MAE global < 20% del valor absoluto medio)

    Raises:
        ValueError: Si el DataFrame es muy pequeño para el backtesting.
    """
    required_cols = {"fecha", "caja_neta"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas requeridas faltantes: {missing}")

    if df.empty:
        raise ValueError("DataFrame vacío — no se puede ejecutar backtesting")

    # Preparar datos
    data = df[["fecha", "caja_neta"]].copy()
    data["fecha"] = pd.to_datetime(data["fecha"])
    data = data.sort_values("fecha").reset_index(drop=True)

    total_days = len(data)
    min_days = train_window + test_window
    if total_days < min_days:
        raise ValueError(
            f"DataFrame demasiado pequeño: {total_days} días, "
            f"se necesitan al menos {min_days} "
            f"(train_window={train_window} + test_window={test_window})"
        )

    logger.info(
        "Iniciando walk-forward backtesting: %d total días, "
        "train=%d, test=%d, min_iterations=%d",
        total_days, train_window, test_window, min_iterations,
    )

    iterations: List[Dict[str, Any]] = []
    start = 0
    holidays_df = get_mexican_holidays()

    while start + train_window + test_window <= total_days:
        iter_num = len(iterations) + 1

        # Dividir en train/test
        train_end = start + train_window
        test_end = train_end + test_window

        train_df = data.iloc[start:train_end].copy()
        test_df = data.iloc[train_end:test_end].copy()

        # Renombrar para Prophet
        train_prophet = train_df.rename(columns={"fecha": "ds", "caja_neta": "y"})
        test_prophet = test_df.rename(columns={"fecha": "ds", "caja_neta": "y"})

        # Entrenar modelo
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            mcmc_samples=0,
        )
        model = add_holiday_regressors(model, holidays_df)

        try:
            model.fit(train_prophet)

            # Predecir ventana de prueba
            future = model.make_future_dataframe(
                periods=test_window,
                include_history=False,
            )
            forecast = model.predict(future)

            # Alinear predicciones con test_df
            merged = test_prophet.merge(
                forecast[["ds", "yhat"]], on="ds", how="left"
            )

            # Calcular métricas
            metrics = _calculate_metrics(
                merged["y"], merged["yhat"],
            )

            iter_info: Dict[str, Any] = {
                "iteracion": iter_num,
                "train_start": train_df["fecha"].min().date(),
                "train_end": train_df["fecha"].max().date(),
                "test_start": test_df["fecha"].min().date(),
                "test_end": test_df["fecha"].max().date(),
                "train_size": len(train_df),
                "test_size": len(test_df),
                **metrics,
            }
            iterations.append(iter_info)

            logger.info(
                "Iteracion %d: train=%s->%s (%d), test=%s->%s (%d) | "
                "MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
                iter_num,
                iter_info["train_start"], iter_info["train_end"],
                iter_info["train_size"],
                iter_info["test_start"], iter_info["test_end"],
                iter_info["test_size"],
                metrics["mae"], metrics["rmse"], metrics["mape"],
            )

        except Exception as e:
            logger.warning(
                "Iteracion %d fallo (train=%d->%d): %s",
                iter_num, start, train_end, e,
            )

        # Avanzar ventana
        start += test_window

    # Verificar mínimo de iteraciones
    n_iterations = len(iterations)
    if n_iterations < min_iterations:
        logger.warning(
            "Solo se completaron %d iteraciones (mínimo requerido: %d)",
            n_iterations, min_iterations,
        )

    # Calcular métricas globales
    if n_iterations == 0:
        logger.error("No se completó ninguna iteración de backtesting")
        return {
            "iterations": [],
            "global_mae": 0.0,
            "global_rmse": 0.0,
            "global_mape": 0.0,
            "global_smape": 0.0,
            "mape_per_iteration": [],
            "smape_per_iteration": [],
            "n_iterations": 0,
            "train_window": train_window,
            "test_window": test_window,
            "success_by_smape": False,
            "success_by_mae": False,
        }

    # Métricas globales
    mape_per_iteration = [it["mape"] for it in iterations]
    smape_per_iteration = [it["smape"] for it in iterations]
    global_mae = float(np.mean([it["mae"] for it in iterations]))
    global_rmse = float(np.mean([it["rmse"] for it in iterations]))
    global_mape = float(np.mean(mape_per_iteration))
    global_smape = float(np.mean(smape_per_iteration))

    # Criterios de éxito:
    # - SMAPE < 15% es el criterio principal (funciona con datos que cruzan por cero)
    # - MAPE se mantiene como referencia pero NO como criterio de éxito
    success_by_smape = global_smape < 15.0
    # MAE debe ser menor que el 20% del valor absoluto medio de caja_neta
    mean_abs_y = float(np.abs(data["caja_neta"]).mean()) if "caja_neta" in data.columns else 1.0
    success_by_mae = global_mae < 0.20 * mean_abs_y if mean_abs_y > 0 else False

    logger.info(
        "Backtesting completado: %d iteraciones, "
        "MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%, SMAPE=%.2f%% — %s",
        n_iterations,
        global_mae, global_rmse, global_mape, global_smape,
        "EXITOSO (SMAPE)" if success_by_smape else "NO ALCANZA UMBRAL SMAPE",
    )

    return {
        "iterations": iterations,
        "global_mae": global_mae,
        "global_rmse": global_rmse,
        "global_mape": global_mape,
        "global_smape": global_smape,
        "mape_per_iteration": mape_per_iteration,
        "smape_per_iteration": smape_per_iteration,
        "n_iterations": n_iterations,
        "train_window": train_window,
        "test_window": test_window,
        "success_by_smape": success_by_smape,
        "success_by_mae": success_by_mae,
    }
