"""Tests T05 — Backtesting (walk-forward validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.backtest import _calculate_metrics, walk_forward_backtest


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


class TestCalculateMetrics:
    """Tests unitarios para _calculate_metrics."""

    def test_perfect_prediction(self) -> None:
        """Caso perfecto: y_true == y_pred → MAE=0, RMSE=0, MAPE=0."""
        y_true = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
        y_pred = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])

        metrics = _calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mape"] == 0.0

    def test_known_error(self) -> None:
        """Caso conocido: y_true=[100, 200], y_pred=[110, 190].

        MAE = (10 + 10)/2 = 10
        RMSE = sqrt((100 + 100)/2) = sqrt(100) = 10
        MAPE = (10/100 + 10/200)/2 * 100 = (0.10 + 0.05)/2 * 100 = 7.5%
        """
        y_true = pd.Series([100.0, 200.0])
        y_pred = pd.Series([110.0, 190.0])

        metrics = _calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == pytest.approx(10.0)
        assert metrics["rmse"] == pytest.approx(10.0)
        assert metrics["mape"] == pytest.approx(7.5)

    def test_zero_in_y_true(self) -> None:
        """Caso con cero en y_true: MAPE debe excluir ese punto sin error."""
        y_true = pd.Series([0.0, 100.0, 200.0])
        y_pred = pd.Series([10.0, 110.0, 190.0])

        # No debe lanzar error
        metrics = _calculate_metrics(y_true, y_pred)

        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0

        # MAPE debe calcularse solo sobre y_true != 0
        # Para [100, 200] vs [110, 190]: MAPE = 7.5%
        assert metrics["mape"] == pytest.approx(7.5)
        assert np.isfinite(metrics["mape"])

    def test_all_same_values(self) -> None:
        """Caso donde todos los valores son iguales."""
        y_true = pd.Series([100.0, 100.0, 100.0])
        y_pred = pd.Series([100.0, 100.0, 100.0])

        metrics = _calculate_metrics(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mape"] == 0.0

    def test_large_errors(self) -> None:
        """Caso con errores grandes."""
        y_true = pd.Series([1000.0, 2000.0, 3000.0])
        y_pred = pd.Series([1500.0, 1000.0, 4000.0])

        metrics = _calculate_metrics(y_true, y_pred)

        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mape"] > 0
        assert metrics["smape"] > 0
        assert np.isfinite(metrics["mae"])
        assert np.isfinite(metrics["rmse"])
        assert np.isfinite(metrics["mape"])
        assert np.isfinite(metrics["smape"])

    def test_smape_perfect_prediction(self) -> None:
        """SMAPE debe ser 0 cuando la predicción es perfecta."""
        y_true = pd.Series([100.0, 200.0, 300.0])
        y_pred = pd.Series([100.0, 200.0, 300.0])
        metrics = _calculate_metrics(y_true, y_pred)
        assert metrics["smape"] == 0.0

    def test_smape_known_value(self) -> None:
        """SMAPE con valores conocidos: y_true=[100,200], y_pred=[110,190].

        SMAPE = 200/2 * (|100-110|/(100+110) + |200-190|/(200+190))
             = 100 * (10/210 + 10/390)
             = 100 * (0.04762 + 0.02564)
             = 100 * 0.07326
             = 7.326%
        """
        y_true = pd.Series([100.0, 200.0])
        y_pred = pd.Series([110.0, 190.0])
        metrics = _calculate_metrics(y_true, y_pred)
        expected_smape = 100.0 * (10.0/210.0 + 10.0/390.0)  # ya es mean, sin /n adicional
        assert metrics["smape"] == pytest.approx(expected_smape, abs=0.01)

    def test_smape_with_zeros(self) -> None:
        """SMAPE debe funcionar cuando y_true contiene ceros."""
        y_true = pd.Series([0.0, 100.0, -50.0])
        y_pred = pd.Series([10.0, 110.0, -45.0])
        metrics = _calculate_metrics(y_true, y_pred)

        assert np.isfinite(metrics["smape"]), "SMAPE no debe ser NaN con ceros"
        assert metrics["smape"] > 0, "SMAPE debe ser positivo con errores"
        # SMAPE debe dar un valor razonable (< 100%) incluso con ceros
        assert metrics["smape"] < 100.0, (
            f"SMAPE debe ser < 100% para errores pequeños, got {metrics['smape']}%"
        )

    def test_smape_both_zero(self) -> None:
        """SMAPE debe ser 0 cuando ambos, real y predicción, son cero."""
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = pd.Series([0.0, 0.0, 0.0])
        metrics = _calculate_metrics(y_true, y_pred)
        assert metrics["smape"] == 0.0

    def test_smape_symmetric(self) -> None:
        """SMAPE debe ser simétrico: intercambiar y_true e y_pred da el mismo resultado."""
        y_true = pd.Series([100.0, 200.0, 300.0])
        y_pred = pd.Series([120.0, 180.0, 330.0])
        m1 = _calculate_metrics(y_true, y_pred)["smape"]
        m2 = _calculate_metrics(y_pred, y_true)["smape"]
        assert m1 == pytest.approx(m2, abs=1e-10), "SMAPE debe ser simétrico"

    def test_smape_bounded_by_200(self) -> None:
        """SMAPE máximo es 200% (cuando uno es 0 y el otro no)."""
        y_true = pd.Series([100.0, 100.0])
        y_pred = pd.Series([-100.0, -100.0])  # error máximo relativo
        metrics = _calculate_metrics(y_true, y_pred)
        # |100 - (-100)| = 200, denom = 100 + 100 = 200 → ratio = 1.0 → 200% / 2 * 1 = 100%
        # Wait, SMAPE = 2 * |diff| / (|true| + |pred|) = 2 * 200 / (100 + 100) = 400/200 = 2.0 → 200%
        # En realidad SMAPE = mean(2*|diff|/(|true|+|pred|))
        # Para [100, -100]: 2*200/(100+100) = 400/200 = 2.0
        # mean = 2.0, *100 = 200%
        assert metrics["smape"] == pytest.approx(200.0, abs=1.0), (
            f"SMAPE máximo debe ser ~200%, got {metrics['smape']}%"
        )


class TestWalkForwardBacktest:
    """Tests para walk_forward_backtest."""

    def test_walk_forward_basic(self, synthetic_df_18m: pd.DataFrame) -> None:
        """Ejecuta walk-forward y verifica estructura de resultados."""
        results = walk_forward_backtest(
            synthetic_df_18m,
            train_window=90,
            test_window=30,
            min_iterations=6,
        )

        # Verificar keys esperadas (incluyendo SMAPE)
        expected_keys = {
            "iterations", "global_mae", "global_rmse", "global_mape",
            "global_smape", "mape_per_iteration", "smape_per_iteration",
            "n_iterations", "train_window", "test_window",
            "success_by_smape", "success_by_mae",
        }
        assert expected_keys.issubset(results.keys()), (
            f"Keys faltantes: {expected_keys - set(results.keys())}"
        )

        # Verificar n_iterations >= 6
        assert results["n_iterations"] >= 6, (
            f"n_iterations={results['n_iterations']} < 6"
        )

        # Verificar MAPE global es un float válido (puede ser alto
        # cuando y_true cruza por cero — limitación conocida de MAPE)
        assert isinstance(results["global_mape"], float)
        assert results["global_mape"] > 0.0, (
            f"global_mape={results['global_mape']} debe ser positivo"
        )
        assert np.isfinite(results["global_mape"]), (
            "global_mape no debe ser NaN o Inf"
        )

        # Verificar SMAPE global es un float válido y más pequeño que MAPE
        # (SMAPE es simétrico y no explota con valores cercanos a cero)
        assert isinstance(results["global_smape"], float)
        assert results["global_smape"] > 0.0, (
            f"global_smape={results['global_smape']} debe ser positivo"
        )
        assert np.isfinite(results["global_smape"]), (
            "global_smape no debe ser NaN o Inf"
        )

        # SMAPE debe ser significativamente menor que MAPE en datos sintéticos
        # (porque SMAPE no explota cuando y_true se acerca a cero)
        assert results["global_smape"] < results["global_mape"], (
            f"SMAPE ({results['global_smape']:.2f}) debe ser < MAPE "
            f"({results['global_mape']:.2f}) en datos sintéticos con ruido"
        )

        # Verificar MAE > 0
        assert results["global_mae"] > 0, f"global_mae={results['global_mae']} <= 0"

        # Verificar RMSE > 0
        assert results["global_rmse"] > 0, f"global_rmse={results['global_rmse']} <= 0"

        # Verificar que success_* son bool
        assert isinstance(results["success_by_smape"], bool)
        assert isinstance(results["success_by_mae"], bool)

    def test_backtest_metrics_consistency(self, synthetic_df_18m: pd.DataFrame) -> None:
        """Verifica consistencia de métricas en backtesting incluyendo SMAPE."""
        results = walk_forward_backtest(
            synthetic_df_18m,
            train_window=90,
            test_window=30,
            min_iterations=6,
        )

        # MAE <= RMSE por propiedad matemática
        assert results["global_mae"] <= results["global_rmse"] + 1e-6, (
            f"MAE ({results['global_mae']}) > RMSE ({results['global_rmse']})"
        )

        # Todas las iteraciones deben tener métricas (incluyendo SMAPE)
        assert len(results["iterations"]) == results["n_iterations"]
        for it in results["iterations"]:
            assert "mae" in it
            assert "rmse" in it
            assert "mape" in it
            assert "smape" in it
            assert it["mae"] > 0
            assert it["rmse"] > 0
            assert it["mape"] > 0, f"MAPE debe ser positivo, got {it['mape']}"
            assert it["smape"] > 0, f"SMAPE debe ser positivo, got {it['smape']}"
            assert np.isfinite(it["mape"]), (
                f"MAPE no debe ser NaN o Inf, got {it['mape']}"
            )
            assert np.isfinite(it["smape"]), (
                f"SMAPE no debe ser NaN o Inf, got {it['smape']}"
            )
            # SMAPE debe ser < MAPE (por construcción, SMAPE <= MAPE cuando hay valores cercanos a cero)
            assert it["smape"] <= it["mape"] + 1e-6 or it["mape"] > 100, (
                f"SMAPE ({it['smape']:.2f}) debe ser <= MAPE ({it['mape']:.2f}) "
                f"o MAPE debe ser > 100% (indicador de valores cercanos a cero)"
            )

        # MAPE per iteration debe coincidir
        assert len(results["mape_per_iteration"]) == results["n_iterations"]
        assert abs(
            sum(results["mape_per_iteration"]) / len(results["mape_per_iteration"])
            - results["global_mape"]
        ) < 1e-6

        # SMAPE per iteration debe coincidir
        assert len(results["smape_per_iteration"]) == results["n_iterations"]
        assert abs(
            sum(results["smape_per_iteration"]) / len(results["smape_per_iteration"])
            - results["global_smape"]
        ) < 1e-6

    def test_backtest_insufficient_data(self) -> None:
        """Verifica que rechace datos insuficientes."""
        df_small = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=50, freq="D"),
            "caja_neta": np.random.default_rng(42).normal(15000, 3000, 50),
        })

        with pytest.raises(ValueError, match="demasiado pequeño"):
            walk_forward_backtest(df_small, train_window=90, test_window=30)

    def test_backtest_empty_df(self) -> None:
        """Verifica que rechace DataFrame vacío."""
        df_empty = pd.DataFrame(columns=["fecha", "caja_neta"])
        with pytest.raises(ValueError, match="vacío"):
            walk_forward_backtest(df_empty)

    def test_backtest_missing_columns(self, synthetic_df_18m: pd.DataFrame) -> None:
        """Verifica que rechace columnas faltantes."""
        df_bad = synthetic_df_18m.drop(columns=["caja_neta"])
        with pytest.raises(ValueError, match="Columnas requeridas"):
            walk_forward_backtest(df_bad)
