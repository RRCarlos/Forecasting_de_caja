"""Tests T04 — Forecasting (train + predict)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from src.forecasting.predict import _interpolate_confidence, generate_forecast
from src.forecasting.train import get_mexican_holidays, train_prophet_model


@pytest.fixture
def synthetic_df_6m() -> pd.DataFrame:
    """Genera 6 meses de datos sintéticos para testing."""
    rng = np.random.default_rng(42)
    n = 180
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Tendencia lineal + ruido
    trend = np.linspace(10000, 15000, n)
    weekly = 2000 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 2000, n)
    caja_neta = trend + weekly + noise
    return pd.DataFrame({"fecha": dates, "caja_neta": caja_neta})


@pytest.fixture
def synthetic_df_18m() -> pd.DataFrame:
    """Genera 18 meses de datos sintéticos para backtesting."""
    rng = np.random.default_rng(42)
    n = 540
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = np.linspace(10000, 18000, n)
    weekly = 2000 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 3000 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = rng.normal(0, 2500, n)
    caja_neta = trend + weekly + yearly + noise
    return pd.DataFrame({"fecha": dates, "caja_neta": caja_neta})


class TestMexicanHolidays:
    """Tests para get_mexican_holidays."""

    def test_mexican_holidays(self) -> None:
        """Verifica que get_mexican_holidays retorna un DataFrame válido."""
        holidays = get_mexican_holidays()

        # Verificar columnas
        assert isinstance(holidays, pd.DataFrame)
        assert "ds" in holidays.columns
        assert "holiday" in holidays.columns

        # Verificar cantidad (12 festivos x 2 años = 24)
        assert len(holidays) >= 10, (
            f"Debe tener al menos 10 festivos, tiene {len(holidays)}"
        )
        assert len(holidays) == 24, (
            f"Debe tener 24 festivos (12/año × 2 años), tiene {len(holidays)}"
        )

        # Verificar que ds sea datetime
        assert pd.api.types.is_datetime64_any_dtype(holidays["ds"])

        # Verificar festivos clave presentes
        holiday_names = holidays["holiday"].unique()
        for expected in ["Navidad", "Año Nuevo", "Jueves Santo", "Viernes Santo"]:
            assert expected in holiday_names, f"Falta el festivo: {expected}"

    def test_holidays_2024_2025(self) -> None:
        """Verifica que hay festivos para ambos años."""
        holidays = get_mexican_holidays()
        years = holidays["ds"].dt.year.unique()
        assert 2024 in years
        assert 2025 in years

    def test_holidays_no_duplicates(self) -> None:
        """Verifica que no haya fechas duplicadas."""
        holidays = get_mexican_holidays()
        assert holidays["ds"].is_unique, "Hay fechas de festivos duplicadas"


class TestProphetTrain:
    """Tests para train_prophet_model."""

    def test_prophet_train_and_predict(self, synthetic_df_6m: pd.DataFrame) -> None:
        """Entrena Prophet con datos sintéticos y verifica predicciones."""
        # Entrenar modelo
        model, forecast = train_prophet_model(
            synthetic_df_6m,
            save_path=None,  # No guardar en tests
        )

        # Verificar tipos de retorno
        assert isinstance(model, Prophet)
        assert isinstance(forecast, pd.DataFrame)

        # Verificar columnas esperadas en forecast
        expected_cols = {"ds", "yhat", "yhat_lower", "yhat_upper"}
        assert expected_cols.issubset(forecast.columns), (
            f"Columnas faltantes: {expected_cols - set(forecast.columns)}"
        )

        # Verificar que no hay NaN en predicciones
        assert not forecast["yhat"].isna().any(), "yhat contiene NaN"
        assert not forecast["yhat_lower"].isna().any(), "yhat_lower contiene NaN"
        assert not forecast["yhat_upper"].isna().any(), "yhat_upper contiene NaN"

        # Verificar consistencia de IC: yhat_lower <= yhat <= yhat_upper
        assert (forecast["yhat_lower"] <= forecast["yhat"]).all(), (
            "yhat_lower > yhat en algunas filas"
        )
        assert (forecast["yhat"] <= forecast["yhat_upper"]).all(), (
            "yhat > yhat_upper en algunas filas"
        )

    def test_train_empty_df(self) -> None:
        """Verifica que train_prophet_model rechace DataFrame vacío."""
        df_empty = pd.DataFrame(columns=["fecha", "caja_neta"])
        with pytest.raises(ValueError, match="vacío"):
            train_prophet_model(df_empty, save_path=None)

    def test_train_missing_columns(self, synthetic_df_6m: pd.DataFrame) -> None:
        """Verifica que train_prophet_model rechace columnas faltantes."""
        df_bad = synthetic_df_6m.drop(columns=["caja_neta"])
        with pytest.raises(ValueError, match="Columnas requeridas"):
            train_prophet_model(df_bad, save_path=None)


class TestForecastPredict:
    """Tests para generate_forecast."""

    def test_forecast_confidence_intervals(self, synthetic_df_6m: pd.DataFrame) -> None:
        """Verifica que los IC 80% están contenidos dentro de los IC 95%."""
        model, full_forecast = train_prophet_model(
            synthetic_df_6m, save_path=None,
        )
        predictions = generate_forecast(model, periods=30)

        if predictions.empty:
            pytest.skip("No se generaron predicciones futuras")

        # Verificar columnas esperadas
        expected_cols = {
            "ds", "yhat", "yhat_lower_80", "yhat_upper_80",
            "yhat_lower", "yhat_upper", "horizonte",
        }
        assert expected_cols.issubset(predictions.columns), (
            f"Columnas faltantes: {expected_cols - set(predictions.columns)}"
        )

        # Verificar consistencia: IC 80% dentro de IC 95%
        assert (predictions["yhat_lower"] <= predictions["yhat_lower_80"]).all(), (
            "yhat_lower_95 > yhat_lower_80 en algunas filas"
        )
        assert (predictions["yhat_lower_80"] <= predictions["yhat_upper_80"]).all(), (
            "yhat_lower_80 > yhat_upper_80 en algunas filas"
        )
        assert (predictions["yhat_upper_80"] <= predictions["yhat_upper"]).all(), (
            "yhat_upper_80 > yhat_upper_95 en algunas filas"
        )

        # Verificar horizonte
        assert "horizonte" in predictions.columns
        unique_horizontes = predictions["horizonte"].unique()
        assert all(h in ["30d", "60d", "90d", "90d+"] for h in unique_horizontes)

    def test_interpolate_confidence(self) -> None:
        """Verifica que _interpolate_confidence funciona correctamente."""
        rng = np.random.default_rng(42)
        n = 100
        yhat = pd.Series(rng.normal(15000, 3000, n))
        # IC 95%: yhat ± 1.96 * sigma
        sigma = 2000
        yhat_lower = yhat - 1.96 * sigma
        yhat_upper = yhat + 1.96 * sigma

        lower_80, upper_80 = _interpolate_confidence(
            yhat, yhat_lower, yhat_upper, target_alpha=0.20,
        )

        # Para IC 80%, el rango debe ser más angosto que IC 95%
        range_95 = (yhat_upper - yhat_lower).mean()
        range_80 = (upper_80 - lower_80).mean()
        assert range_80 < range_95, "IC 80% debe ser más angosto que IC 95%"

        # Relación teórica: 1.28/1.96 ≈ 0.653
        ratio = range_80 / range_95
        assert abs(ratio - 1.28 / 1.96) < 0.01, (
            f"Ratio de intervalos {ratio:.4f} != {1.28/1.96:.4f}"
        )

    def test_generate_forecast_invalid_periods(self, synthetic_df_6m: pd.DataFrame) -> None:
        """Verifica que generate_forecast rechace periods <= 0."""
        model, _ = train_prophet_model(synthetic_df_6m, save_path=None)
        with pytest.raises(ValueError, match="periods debe ser"):
            generate_forecast(model, periods=0)
