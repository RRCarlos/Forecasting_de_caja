#!/usr/bin/env python
"""
Orquestador CLI del pipeline completo de Forecasting de Caja.

Modo de uso:
    python main.py --help
    python main.py pipeline            # Ejecuta todo: ETL -> anomalas -> forecast -> reportes
    python main.py etl                  # Solo ETL (generar -> limpiar -> features)
    python main.py anomalies            # Solo deteccin de anomalas
    python main.py forecast             # Solo forecasting (entrenar -> predecir -> backtestear)
    python main.py reports              # Solo reportes (narrativa, ejecutivo, QA, PowerBI)
    python main.py visualize            # Solo visualizaciones
    python main.py all                  # pipeline completo

Uso:
    python main.py pipeline --periods 24 --anomaly-rate 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _configurar_logging(verbose: bool = False) -> None:
    """Configura el logging del pipeline.

    Args:
        verbose: Si True, muestra logs en nivel DEBUG.
    """
    nivel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------


def cmd_etl(args: argparse.Namespace) -> int:
    """Ejecuta el pipeline ETL: generacin -> limpieza -> features.

    Args:
        args: Argumentos del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    try:
        logger.info("=== ETL: Generacin de datos ===")
        from src.etl.generator import generate_dataset

        df_raw = generate_dataset(
            seed=args.seed,
            periods=args.periods,
            anomaly_rate=args.anomaly_rate,
        )
        logger.info("Dataset generado: %d filas", len(df_raw))

        logger.info("=== ETL: Limpieza de datos ===")
        from src.etl.cleaner import clean_dataset

        df_clean, report = clean_dataset(df_raw)
        logger.info(
            "Dataset limpio: %d filas, %d columnas",
            len(df_clean),
            len(df_clean.columns),
        )

        logger.info("=== ETL: Feature engineering ===")
        from src.etl.features import add_features

        df_feat = add_features(df_clean)
        logger.info(
            "Features creadas: %d filas, %d columnas",
            len(df_feat),
            df_feat.shape[1],
        )

        logger.info("ETL completado exitosamente.")
        return 0

    except Exception as e:
        logger.error("Error en ETL: %s", e, exc_info=True)
        return 1


def cmd_anomalies(args: argparse.Namespace) -> int:
    """Ejecuta la deteccin de anomalas.

    Args:
        args: Argumentos del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    try:
        # Cargar datos
        df = pd.read_csv("data/curated/dataset_features.csv")
        logger.info("Datos cargados: %d filas", len(df))

        # Detectar con todos los mtodos
        logger.info("=== Anomalas: Isolation Forest ===")
        from src.anomalies.isolation_forest import detect_isolation_forest, get_if_score

        if_anom = detect_isolation_forest(df, random_state=args.seed)
        if_scores = get_if_score(df, random_state=args.seed)

        logger.info("=== Anomalas: Z-score ===")
        from src.anomalies.statistical import detect_zscore

        z_anom = detect_zscore(df)

        logger.info("=== Anomalas: IQR ===")
        from src.anomalies.statistical import detect_iqr

        iqr_anom = detect_iqr(df)

        logger.info("=== Anomalas: Benford ===")
        from src.anomalies.benford import detect_benford_anomalies

        benford_anom = detect_benford_anomalies(df)

        logger.info("=== Anomalas: Temporal ===")
        from src.anomalies.temporal import detect_temporal_anomalies

        temp_anom = detect_temporal_anomalies(df)

        # Consenso
        logger.info("=== Anomalas: Matriz de consenso ===")
        from src.anomalies.consensus import build_consensus_matrix

        consensus_df = build_consensus_matrix(
            df=df,
            if_anomalies=if_anom,
            zscore_anomalies=z_anom,
            iqr_anomalies=iqr_anom,
            benford_anomalies=benford_anom,
            temporal_anomalies=temp_anom,
            if_scores=if_scores,
        )

        # Reporte
        logger.info("=== Anomalas: Generacin de reportes ===")
        from src.anomalies.report import generate_anomaly_report

        stats = generate_anomaly_report(consensus_df, df)
        logger.info(
            "Anomalas detectadas: %d", stats["total_anomalies"],
        )

        logger.info("Detccin de anomalas completada exitosamente.")
        return 0

    except Exception as e:
        logger.error("Error en deteccin de anomalas: %s", e, exc_info=True)
        return 1


def cmd_forecast(args: argparse.Namespace) -> int:
    """Ejecuta el pipeline de forecasting.

    Args:
        args: Argumentos del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    try:
        # Cargar datos
        df = pd.read_csv("data/curated/dataset_features.csv")
        logger.info("Datos cargados: %d filas", len(df))

        # Entrenar
        logger.info("=== Forecast: Entrenamiento Prophet ===")
        from src.forecasting.train import train_prophet_model

        model, forecast = train_prophet_model(df, save_path="models/prophet_model.pkl")

        # Predecir
        logger.info("=== Forecast: Generacin de predicciones ===")
        from src.forecasting.predict import generate_forecast

        predictions = generate_forecast(model, periods=args.periods)
        logger.info("Predicciones generadas: %d das", len(predictions))

        # Backtest
        logger.info("=== Forecast: Walk-forward backtesting ===")
        from src.forecasting.backtest import walk_forward_backtest

        backtest_results = walk_forward_backtest(
            df,
            train_window=60,
            test_window=20,
        )
        logger.info(
            "Backtest: %d iteraciones, SMAPE=%.2f%%",
            backtest_results["n_iterations"],
            backtest_results["global_smape"],
        )

        logger.info("Forecasting completado exitosamente.")
        return 0

    except Exception as e:
        logger.error("Error en forecasting: %s", e, exc_info=True)
        return 1


def cmd_visualize(args: argparse.Namespace) -> int:
    """Genera las visualizaciones del forecast.

    Args:
        args: Argumentos del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    try:
        # Cargar datos
        df = pd.read_csv("data/curated/dataset_features.csv")
        logger.info("Datos cargados: %d filas", len(df))

        # Cargar modelo y forecast
        from src.forecasting.train import train_prophet_model

        model, forecast = train_prophet_model(df, save_path=None)

        from src.forecasting.predict import generate_forecast

        predictions = generate_forecast(model, periods=90)

        # Cargar anomalas para el grfico de anomalas
        anom_path = "reports/anomaly_report.csv"
        anomalies_df = None
        import os

        if os.path.exists(anom_path):
            anomalies_df = pd.read_csv(anom_path)

        logger.info("=== Visualizacin: Generacin de graficos ===")
        from src.forecasting.visualize import generate_all_plots

        paths = generate_all_plots(
            forecast_df=predictions,
            model=model,
            anomalies_df=anomalies_df,
            output_dir="reports",
        )
        logger.info("Graficos generados: %d archivos", len(paths))

        logger.info("Visualizaciones completadas exitosamente.")
        return 0

    except Exception as e:
        logger.error("Error en visualizaciones: %s", e, exc_info=True)
        return 1


def cmd_reports(args: argparse.Namespace) -> int:
    """Genera los reportes de la Fase 4.

    Args:
        args: Argumentos del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    try:
        logger.info("=== Reportes: Power BI Export ===")
        from src.dashboards.export_powerbi import export_powerbi

        powerbi_result = export_powerbi(output_dir="reports")
        logger.info("Power BI export: %s", powerbi_result["powerbi_csv"])

        logger.info("=== Reportes: Smart Narrative ===")
        from src.reports.smart_narrative import generate_smart_narrative

        narrative_path = generate_smart_narrative()
        logger.info("Smart Narrative: %s", narrative_path)

        logger.info("=== Reportes: Resumen Ejecutivo ===")
        from src.reports.executive_summary import generate_executive_summary

        exec_path = generate_executive_summary()
        logger.info("Resumen ejecutivo: %s", exec_path)

        logger.info("=== Reportes: QA Report ===")
        from src.reports.qa_report import generate_qa_report

        qa_path = generate_qa_report()
        logger.info("QA Report: %s", qa_path)

        logger.info("Reportes generados exitosamente.")
        return 0

    except Exception as e:
        logger.error("Error en reportes: %s", e, exc_info=True)
        return 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Ejecuta el pipeline completo: ETL -> anomalas -> forecast -> reportes.

    Args:
        args: Argumentos del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETO — Forecasting de Caja para Logstica")
    logger.info("=" * 60)

    pasos: List[Dict[str, Any]] = [
        {"nombre": "ETL",        "fn": cmd_etl},
        {"nombre": "Anomalas",  "fn": cmd_anomalies},
        {"nombre": "Forecast",   "fn": cmd_forecast},
        {"nombre": "Visualizar", "fn": cmd_visualize},
        {"nombre": "Reportes",   "fn": cmd_reports},
    ]

    for paso in pasos:
        logger.info("")
        logger.info("--- Paso: %s ---", paso["nombre"])
        codigo = paso["fn"](args)
        if codigo != 0:
            logger.error("Paso '%s' fall con codigo %d", paso["nombre"], codigo)
            return codigo

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos del CLI.

    Returns:
        ArgumentParser configurado con subcomandos.
    """
    parser = argparse.ArgumentParser(
        description="Mini-Modelo de Forecasting de Caja para Logstica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py pipeline               # Pipeline completo
  python main.py etl --periods 12       # Solo ETL, 12 meses
  python main.py reports                # Solo reportes
  python main.py pipeline --verbose     # Pipeline con logs detallados
        """,
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar logs detallados (DEBUG)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=24,
        help="Nmero de meses para generar (default: 24)",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
        help="Tasa de anomalas sintticas (default: 0.05)",
    )

    subparsers = parser.add_subparsers(
        dest="comando",
        title="subcomandos",
        description="Subcomandos disponibles",
    )

    # pipeline
    p_pipeline = subparsers.add_parser(
        "pipeline",
        help="Ejecuta pipeline completo (ETL -> anomalas -> forecast -> reportes)",
    )
    p_pipeline.add_argument(
        "--periods", type=int, default=24,
        help="Nmero de meses para generar (default: 24)",
    )
    p_pipeline.add_argument(
        "--anomaly-rate", type=float, default=0.05,
        help="Tasa de anomalas sintticas (default: 0.05)",
    )

    # etl
    p_etl = subparsers.add_parser(
        "etl", help="Solo ETL (generar -> limpiar -> features)",
    )
    p_etl.add_argument(
        "--periods", type=int, default=24,
        help="Nmero de meses para generar (default: 24)",
    )
    p_etl.add_argument(
        "--anomaly-rate", type=float, default=0.05,
        help="Tasa de anomalas sintticas (default: 0.05)",
    )

    # anomalies
    subparsers.add_parser(
        "anomalies", help="Solo deteccin de anomalas",
    )

    # forecast
    p_forecast = subparsers.add_parser(
        "forecast", help="Solo forecasting",
    )
    p_forecast.add_argument(
        "--periods", type=int, default=90,
        help="Dias de forecast a generar (default: 90)",
    )

    # visualize
    subparsers.add_parser(
        "visualize", help="Solo visualizaciones",
    )

    # reports
    subparsers.add_parser(
        "reports", help="Solo reportes (narrativa, ejecutivo, QA, PowerBI)",
    )

    # all (alias de pipeline)
    p_all = subparsers.add_parser(
        "all", help="Alias de 'pipeline' (completo)",
    )
    p_all.add_argument(
        "--periods", type=int, default=24,
        help="Nmero de meses para generar (default: 24)",
    )
    p_all.add_argument(
        "--anomaly-rate", type=float, default=0.05,
        help="Tasa de anomalas sintticas (default: 0.05)",
    )

    return parser


def main() -> int:
    """Punto de entrada principal del CLI.

    Returns:
        Cdigo de salida (0 = xito, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args()

    _configurar_logging(verbose=args.verbose)

    logger.debug("Argumentos: %s", args)

    # Mapa de subcomandos a funciones
    comandos = {
        "etl": cmd_etl,
        "anomalies": cmd_anomalies,
        "forecast": cmd_forecast,
        "visualize": cmd_visualize,
        "reports": cmd_reports,
        "pipeline": cmd_pipeline,
        "all": cmd_pipeline,
    }

    if args.comando is None:
        parser.print_help()
        return 0

    fn = comandos.get(args.comando)
    if fn is None:
        logger.error("Comando desconocido: %s", args.comando)
        return 1

    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
