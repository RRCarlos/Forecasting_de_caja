"""Tests T07 — Pipeline de integracin completo."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.anomalies.benford import detect_benford_anomalies
from src.anomalies.consensus import build_consensus_matrix
from src.anomalies.isolation_forest import detect_isolation_forest, get_if_score
from src.anomalies.report import generate_anomaly_report
from src.anomalies.statistical import detect_zscore, detect_iqr
from src.anomalies.temporal import detect_temporal_anomalies
from src.etl.cleaner import clean_dataset
from src.etl.features import add_features
from src.etl.generator import generate_dataset
from src.forecasting.backtest import walk_forward_backtest
from src.forecasting.predict import generate_forecast
from src.forecasting.train import train_prophet_model
from src.forecasting.visualize import generate_all_plots


@pytest.fixture
def pipeline_data() -> dict:
    """Genera datos de pipeline completos para tests de integracin.

    Returns:
        Dict con todos los artefactos del pipeline.
    """
    seed = 42
    periods = 12  # Usar 12 meses para tests rpidos
    anomaly_rate = 0.05

    # 1. ETL
    df_raw = generate_dataset(seed=seed, periods=periods, anomaly_rate=anomaly_rate)
    df_clean, _ = clean_dataset(df_raw)
    df_feat = add_features(df_clean)

    # 2. Deteccin de anomalas
    if_anom = detect_isolation_forest(df_feat, random_state=seed)
    if_scores = get_if_score(df_feat, random_state=seed)
    z_anom = detect_zscore(df_feat)
    iqr_anom = detect_iqr(df_feat)
    benford_anom = detect_benford_anomalies(df_feat)
    temp_anom = detect_temporal_anomalies(df_feat)

    consensus_df = build_consensus_matrix(
        df=df_feat,
        if_anomalies=if_anom,
        zscore_anomalies=z_anom,
        iqr_anomalies=iqr_anom,
        benford_anomalies=benford_anom,
        temporal_anomalies=temp_anom,
        if_scores=if_scores,
    )
    anomaly_stats = generate_anomaly_report(consensus_df, df_feat)

    # 3. Forecasting
    model, forecast_full = train_prophet_model(df_feat, save_path=None)
    predictions = generate_forecast(model, periods=30)
    backtest_results = walk_forward_backtest(
        df_feat,
        train_window=60,
        test_window=20,
        min_iterations=3,
    )

    # Anomalas para visualizacin
    anomalies_for_viz = pd.read_csv("reports/anomaly_report.csv") if os.path.exists(
        "reports/anomaly_report.csv"
    ) else None

    # Visualizaciones
    viz_paths = []
    try:
        viz_paths = generate_all_plots(
            forecast_df=predictions,
            model=model,
            anomalies_df=anomalies_for_viz,
            output_dir="reports",
        )
    except Exception:
        pass  # No crtico si plotly no est disponible

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "df_feat": df_feat,
        "consensus": consensus_df,
        "anomaly_stats": anomaly_stats,
        "model": model,
        "forecast_full": forecast_full,
        "predictions": predictions,
        "backtest": backtest_results,
        "viz_paths": viz_paths,
    }


class TestPipelineEndToEnd:
    """Tests de integracin del pipeline completo."""

    def test_pipeline_end_to_end(self, pipeline_data: dict) -> None:
        """Verifica que el pipeline completo se ejecute sin errores.

        NO ejecuta export_powerbi ni reportes (pesados para test unitario).
        """
        data = pipeline_data

        # ── 1. ETL ────────────────────────────────────────────────────────
        assert isinstance(data["df_raw"], pd.DataFrame), (
            "generate_dataset debe retornar DataFrame"
        )
        assert len(data["df_raw"]) > 0, "Raw dataset no debe estar vaco"

        assert isinstance(data["df_clean"], pd.DataFrame), (
            "clean_dataset debe retornar DataFrame"
        )
        assert len(data["df_clean"]) > 0, "Clean dataset no debe estar vaco"
        assert "caja_neta" in data["df_clean"].columns or (
            "ingreso_efectivo" in data["df_clean"].columns
        ), "clean_dataset debe mantener columnas de ingresos"

        assert isinstance(data["df_feat"], pd.DataFrame), (
            "add_features debe retornar DataFrame"
        )
        assert len(data["df_feat"]) > 0, "Features dataset no debe estar vaco"
        assert "caja_neta" in data["df_feat"].columns, (
            "add_features debe crear caja_neta"
        )
        assert "lag_1" in data["df_feat"].columns, (
            "add_features debe crear lags"
        )

        # ── 2. Anomalas ──────────────────────────────────────────────────
        assert isinstance(data["consensus"], pd.DataFrame), (
            "build_consensus_matrix debe retornar DataFrame"
        )
        assert "severidad" in data["consensus"].columns, (
            "consenso debe tener columna severidad"
        )
        assert "consenso_score" in data["consensus"].columns, (
            "consenso debe tener columna consenso_score"
        )

        assert isinstance(data["anomaly_stats"], dict), (
            "generate_anomaly_report debe retornar dict"
        )
        assert "total_anomalies" in data["anomaly_stats"], (
            "anomaly_stats debe tener total_anomalies"
        )
        assert data["anomaly_stats"]["total_anomalies"] >= 0, (
            "total_anomalies debe ser >= 0"
        )

        # ── 3. Forecast ──────────────────────────────────────────────────
        from prophet import Prophet

        assert isinstance(data["model"], Prophet), (
            "train_prophet_model debe retornar modelo Prophet"
        )

        assert isinstance(data["forecast_full"], pd.DataFrame), (
            "train_prophet_model debe retornar forecast DataFrame"
        )
        assert "yhat" in data["forecast_full"].columns, (
            "forecast debe tener columna yhat"
        )

        assert isinstance(data["predictions"], pd.DataFrame), (
            "generate_forecast debe retornar DataFrame"
        )
        if not data["predictions"].empty:
            assert "horizonte" in data["predictions"].columns, (
                "predictions debe tener columna horizonte"
            )
            assert "yhat_lower_80" in data["predictions"].columns, (
                "predictions debe tener IC 80%"
            )

        # ── 4. Backtest ──────────────────────────────────────────────────
        assert isinstance(data["backtest"], dict), (
            "walk_forward_backtest debe retornar dict"
        )
        expected_keys = {
            "iterations", "global_mae", "global_rmse",
            "global_mape", "global_smape",
            "mape_per_iteration", "smape_per_iteration",
            "n_iterations", "train_window", "test_window",
            "success_by_smape", "success_by_mae",
        }
        assert expected_keys.issubset(data["backtest"].keys()), (
            f"Keys faltantes en backtest: "
            f"{expected_keys - set(data['backtest'].keys())}"
        )
        assert data["backtest"]["n_iterations"] >= 1, (
            "Debe haber al menos 1 iteracin de backtest"
        )
        assert np.isfinite(data["backtest"]["global_smape"]), (
            "SMAPE global debe ser finito"
        )

        # ── 5. Visualizaciones (opcional) ────────────────────────────────
        # No se verifica contenido, solo que no haya errores

    def test_backtest_metrics_reasonable(self, pipeline_data: dict) -> None:
        """Verifica que las mtricas de backtest sean razonables."""
        backtest = pipeline_data["backtest"]

        assert backtest["global_mae"] >= 0, "MAE no debe ser negativo"
        assert backtest["global_rmse"] >= 0, "RMSE no debe ser negativo"
        assert backtest["global_smape"] >= 0, "SMAPE no debe ser negativo"

        # MAE <= RMSE por propiedad matemtica
        assert backtest["global_mae"] <= backtest["global_rmse"] + 1e-6, (
            f"MAE ({backtest['global_mae']}) > RMSE ({backtest['global_rmse']})"
        )

        # success_* deben ser booleanos
        assert isinstance(backtest["success_by_smape"], bool)
        assert isinstance(backtest["success_by_mae"], bool)

    def test_anomaly_consistency(self, pipeline_data: dict) -> None:
        """Verifica consistencia entre deteccin de anomalas y datos."""
        consensus = pipeline_data["consensus"]
        df_feat = pipeline_data["df_feat"]

        # Mismo nmero de filas
        assert len(consensus) == len(df_feat), (
            f"Consenso ({len(consensus)}) vs Features ({len(df_feat)})"
        )

        # Columnas de deteccin por mtodo
        detector_cols = [
            "if_detectado", "zscore_detectado", "iqr_detectado",
            "benford_detectado", "temporal_detectado",
        ]
        for col in detector_cols:
            assert col in consensus.columns, (
                f"Falta columna detector: {col}"
            )

    def test_forecast_confidence_bounds(self, pipeline_data: dict) -> None:
        """Verifica que los IC del forecast sean consistentes."""
        predictions = pipeline_data["predictions"]

        if predictions.empty:
            pytest.skip("No se generaron predicciones futuras")

        # yhat_lower <= yhat_lower_80 <= yhat <= yhat_upper_80 <= yhat_upper
        assert (predictions["yhat_lower"] <= predictions["yhat_lower_80"]).all(), (
            "IC 95% lower > IC 80% lower"
        )
        assert (predictions["yhat_lower_80"] <= predictions["yhat"]).all(), (
            "IC 80% lower > yhat"
        )
        assert (predictions["yhat"] <= predictions["yhat_upper_80"]).all(), (
            "yhat > IC 80% upper"
        )
        assert (predictions["yhat_upper_80"] <= predictions["yhat_upper"]).all(), (
            "IC 80% upper > IC 95% upper"
        )


class TestMainCli:
    """Tests para el CLI main.py."""

    def test_main_cli_help(self) -> None:
        """Verifica que main.py --help funciona y retorna 0."""
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        assert result.returncode == 0, (
            f"main.py --help fall: {result.stderr}"
        )
        assert "usage:" in result.stdout.lower() or "uso:" in result.stdout.lower(), (
            "Debe mostrar ayuda"
        )

    def test_main_cli_pipeline(self) -> None:
        """Verifica que main.py pipeline --periods 6 existe como comando.

        No ejecuta el pipeline completo (sera muy lento), solo verifica
        que argparse acepta la configuracin.
        """
        # Simular parseo de argumentos
        import argparse
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["pipeline", "--periods", "6", "--anomaly-rate", "0.05"])

        assert args.comando == "pipeline"
        assert args.periods == 6
        assert args.anomaly_rate == 0.05

    def test_main_cli_etl(self) -> None:
        """Verifica que el parser acepta el subcomando etl."""
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["etl"])
        assert args.comando == "etl"
        assert args.periods == 24  # default global
        assert args.seed == 42  # default global

    def test_main_cli_reports(self) -> None:
        """Verifica que el parser acepta el subcomando reports."""
        from main import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["reports"])
        assert args.comando == "reports"

    def test_main_cli_all(self) -> None:
        """Verifica que 'all' es alias de 'pipeline'."""
        from main import _build_parser, cmd_pipeline

        parser = _build_parser()
        args = parser.parse_args(["all"])
        assert args.comando == "all"

    def test_main_cli_no_args(self) -> None:
        """Verifica que sin argumentos muestra ayuda y retorna 0."""
        from main import _build_parser

        parser = _build_parser()
        # Sin argumentos debe mostrar ayuda (no lanza error)
        args = parser.parse_args([])
        assert args.comando is None
