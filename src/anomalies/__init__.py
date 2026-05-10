"""
Módulo de detección de anomalías financieras.

Proporciona 5 métodos de detección más una matriz de consenso que los
combina:

- Isolation Forest: detección multivariada no supervisada
- Z-score: detección univariada por desviaciones estándar
- IQR: detección por rango intercuartílico
- Benford: prueba de la Ley de Benford sobre montos
- Temporal: detección de patrones temporales anómalos
- Consenso: matriz ponderada que integra los 5 métodos
- Reporte: generación de reportes CSV y resúmenes textuales
"""

from __future__ import annotations

from src.anomalies.isolation_forest import detect_isolation_forest, get_if_score
from src.anomalies.statistical import detect_zscore, detect_iqr
from src.anomalies.benford import benford_test, detect_benford_anomalies
from src.anomalies.temporal import detect_temporal_anomalies
from src.anomalies.consensus import build_consensus_matrix
from src.anomalies.report import generate_anomaly_report

__all__ = [
    "detect_isolation_forest",
    "get_if_score",
    "detect_zscore",
    "detect_iqr",
    "benford_test",
    "detect_benford_anomalies",
    "detect_temporal_anomalies",
    "build_consensus_matrix",
    "generate_anomaly_report",
]
