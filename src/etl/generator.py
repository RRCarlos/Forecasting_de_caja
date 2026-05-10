"""
Generador de datos sintéticos de flujo de caja para una empresa de logística.

Produce un dataset diario con ~730 filas (24 meses × ~30.4 días) que incluye
patrones de ingresos y egresos con estacionalidades semanal, mensual y
trimestral, tendencia anual lineal, y ~5% de anomalías controladas de 6
tipos distintos.

Uso:
    from src.etl.generator import generate_dataset
    df = generate_dataset(seed=42, periods=24, anomaly_rate=0.05)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

# ── Festivos mexicanos ──────────────────────────────────────────────────────
# Se incluyen 8 días festivos oficiales + 4 fechas comúnmente observadas.
_FESTIVOS_MEXICANOS: dict[int, list[str]] = {
    2024: [
        "2024-01-01",  # Año Nuevo
        "2024-02-05",  # Día de la Constitución (1er lunes de feb)
        "2024-03-18",  # Natalicio de Benito Juárez (3er lunes de mar)
        "2024-05-01",  # Día del Trabajo
        "2024-09-16",  # Día de la Independencia
        "2024-10-01",  # Transmisión del Poder Ejecutivo
        "2024-11-18",  # Día de la Revolución (3er lunes de nov)
        "2024-12-25",  # Navidad
        "2024-01-06",  # Día de Reyes (no oficial, comúnmente observado)
        "2024-05-05",  # Batalla de Puebla
        "2024-11-02",  # Día de Muertos
        "2024-12-12",  # Día de la Virgen de Guadalupe
    ],
    2025: [
        "2025-01-01",  # Año Nuevo
        "2025-02-03",  # Día de la Constitución (1er lunes de feb)
        "2025-03-17",  # Natalicio de Benito Juárez (3er lunes de mar)
        "2025-05-01",  # Día del Trabajo
        "2025-09-16",  # Día de la Independencia
        "2025-10-01",  # Transmisión del Poder Ejecutivo
        "2025-11-17",  # Día de la Revolución (3er lunes de nov)
        "2025-12-25",  # Navidad
        "2025-01-06",  # Día de Reyes
        "2025-05-05",  # Batalla de Puebla
        "2025-11-02",  # Día de Muertos
        "2025-12-12",  # Día de la Virgen de Guadalupe
    ],
}


def _build_holiday_mask(dates: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
    """Construye máscara booleana para días festivos mexicanos.

    Busca cada fecha en el diccionario de festivos según su año.

    Args:
        dates: Índice de fechas del dataset.

    Returns:
        Arreglo booleano: True si la fecha es festivo mexicano.
    """
    holiday_set: set[str] = set()
    for year in dates.year.unique():
        holiday_set.update(_FESTIVOS_MEXICANOS.get(int(year), []))
    return np.array(
        [d.strftime("%Y-%m-%d") in holiday_set for d in dates],
        dtype=np.bool_,
    )


def _asignar_anomalias(
    rng: np.random.Generator,
    n: int,
    anomaly_rate: float,
    ingreso_efectivo: npt.NDArray[np.float64],
    ingreso_tarjeta: npt.NDArray[np.float64],
    ingreso_transferencia: npt.NDArray[np.float64],
    gasto_operativo: npt.NDArray[np.float64],
    month: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[np.bool_],
    list[Optional[str]],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Inserta anomalías controladas en ~anomaly_rate de las filas.

    Distribuye equitativamente entre 6 tipos de anomalía y modifica in-place
    los arreglos de montos. Los saldos se recalculan externamente después.

    Tipos de anomalía:
        1. duplicado — copia montos de un día reciente (3-7 días atrás)
        2. madrugada — infla ingreso_efectivo (simula error nocturno)
        3. monto_atipico — un monto > 5σ de la media del mes
        4. proveedor_fantasma — infla gasto_operativo 3-6×
        5. monto_redondo — redondea montos a múltiplos de 10,000
        6. benford_violation — fuerza primer dígito a {5, 6, 7}

    Args:
        rng: Generador aleatorio con semilla.
        n: Número total de filas.
        anomaly_rate: Proporción de filas a marcar como anómalas.
        ingreso_efectivo: Arreglo de ingresos en efectivo (modificado in-place).
        ingreso_tarjeta: Arreglo de ingresos con tarjeta (modificado in-place).
        ingreso_transferencia: Arreglo de transferencias (modificado in-place).
        gasto_operativo: Arreglo de gastos operativos (modificado in-place).
        month: Arreglo con el número de mes (1-12) de cada fila.

    Returns:
        Tupla (tiene_anomalia, tipo_anomalia, ingreso_efectivo, ingreso_tarjeta,
               ingreso_transferencia, gasto_operativo) con las anomalías aplicadas.
    """
    tiene_anomalia: npt.NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
    tipo_anomalia: list[Optional[str]] = [None] * n  # type: ignore[assignment]

    n_anomalies = max(1, int(n * anomaly_rate))
    anomalias_idx = rng.choice(n, size=n_anomalies, replace=False)
    rng.shuffle(anomalias_idx)

    tipos = [
        "duplicado",
        "madrugada",
        "monto_atipico",
        "proveedor_fantasma",
        "monto_redondo",
        "benford_violation",
    ]
    n_tipos = len(tipos)
    per_type = n_anomalies // n_tipos

    for i, idx in enumerate(anomalias_idx):
        tipo = tipos[min(i // per_type, n_tipos - 1)]
        tipo_anomalia[idx] = tipo
        tiene_anomalia[idx] = True

        if tipo == "duplicado" and idx >= 3:
            # Copia exacta de montos de un día 3-7 días atrás
            src = idx - int(rng.integers(3, 8))
            ingreso_efectivo[idx] = ingreso_efectivo[src]
            ingreso_tarjeta[idx] = ingreso_tarjeta[src]
            ingreso_transferencia[idx] = ingreso_transferencia[src]
            gasto_operativo[idx] = gasto_operativo[src]

        elif tipo == "madrugada":
            # Simula transacción errónea de madrugada: infla ingreso_efectivo
            factor = rng.uniform(2.0, 5.0)
            ingreso_efectivo[idx] *= factor

        elif tipo == "monto_atipico":
            # Monto > 5σ de la media del mes
            mask_mes = month == month[idx]
            # Usamos ingreso_tarjeta como variable objetivo del spike
            valores_mes = ingreso_tarjeta[mask_mes]
            mean_mes = float(np.mean(valores_mes))
            std_mes = float(np.std(valores_mes)) or 1.0
            ingreso_tarjeta[idx] = mean_mes + rng.uniform(5.5, 8.0) * std_mes

        elif tipo == "proveedor_fantasma":
            # Infla gasto operativo drásticamente (proveedor fantasma)
            factor = rng.uniform(3.0, 6.0)
            gasto_operativo[idx] *= factor

        elif tipo == "monto_redondo":
            # Infla gasto operativo 3-4x y redondea a múltiplos de 10,000
            # Esto crea una anomalía detectable en caja_neta diaria
            factor_mr = rng.uniform(3.0, 4.0)
            gasto_operativo[idx] = (
                np.round(gasto_operativo[idx] * factor_mr / 10_000) * 10_000
            )

        elif tipo == "benford_violation":
            # Fuerza un monto extremadamente alto (primer dígito 9) que
            # NO corresponde a la distribución normal de la empresa.
            # Esto es detectable en caja_neta diaria como un spike.
            gasto_operativo[idx] = rng.uniform(150_000, 250_000)

    return (
        tiene_anomalia,
        tipo_anomalia,
        ingreso_efectivo,
        ingreso_tarjeta,
        ingreso_transferencia,
        gasto_operativo,
    )


def generate_dataset(
    seed: int = 42,
    periods: int = 24,
    anomaly_rate: float = 0.05,
) -> pd.DataFrame:
    """Genera dataset sintético de flujo de caja diario para logística.

    Crea ~periods × 30.4 filas con estacionalidades semanal, mensual y
    trimestral, tendencia lineal anual, ruido gaussiano, y ≈anomaly_rate
    de filas anómalas de 6 tipos distintos.

    Args:
        seed: Semilla del generador aleatorio (reproducibilidad).
        periods: Número de meses a generar.
        anomaly_rate: Proporción de filas que contendrán anomalías.

    Returns:
        DataFrame con ~730 filas y columnas:
            fecha, ingreso_efectivo, ingreso_tarjeta, ingreso_transferencia,
            gasto_operativo, gasto_extraordinario, saldo_diario,
            saldo_acumulado, es_festivo, es_finde, tiene_anomalia,
            tipo_anomalia.

    Raises:
        ValueError: Si periods < 1 o anomaly_rate fuera de [0, 1].
    """
    if periods < 1:
        raise ValueError("`periods` debe ser >= 1")
    if not 0.0 <= anomaly_rate <= 1.0:
        raise ValueError("`anomaly_rate` debe estar en [0, 1]")

    rng = np.random.default_rng(seed)

    # ── Eje temporal ────────────────────────────────────────────────────────
    n = int(periods * 30.4)  # ~730 filas para 24 meses
    start = pd.Timestamp("2024-01-01")
    dates: pd.DatetimeIndex = pd.date_range(start=start, periods=n, freq="D")
    year: npt.NDArray[np.int32] = dates.year.values    # type: ignore[assignment]
    month: npt.NDArray[np.int32] = dates.month.values   # type: ignore[assignment]
    day: npt.NDArray[np.int32] = dates.day.values       # type: ignore[assignment]
    dow: npt.NDArray[np.int32] = dates.dayofweek.values  # type: ignore[assignment]

    logger.info(
        "Generando %d filas desde %s hasta %s",
        n,
        dates[0].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d"),
    )

    # ── Ingresos ────────────────────────────────────────────────────────────
    # Base diaria ~50,000 con distribución log-normal
    base_income = rng.lognormal(mean=np.log(50_000), sigma=0.20, size=n)

    # Estacionalidad semanal
    wk_factor: npt.NDArray[np.float64] = np.ones(n)
    wk_factor[dow >= 5] *= 0.70    # sáb/dom −30%
    wk_factor[dow == 4] *= 1.05    # viernes +5%
    wk_factor[dow == 0] *= 1.03    # lunes +3%

    # Estacionalidad mensual
    mo_factor: npt.NDArray[np.float64] = np.ones(n)
    mo_factor[day <= 7] *= 0.95    # primera semana −5%
    mo_factor[day >= 24] *= 1.10   # última semana +10%

    # Estacionalidad trimestral
    qt_factor: npt.NDArray[np.float64] = np.ones(n)
    qt_factor[(month >= 10) & (month <= 12)] *= 1.20  # Q4 +20%
    qt_factor[(month >= 1) & (month <= 3)] *= 0.90    # Q1 −10%

    # Tendencia lineal: +4% anual
    trend: npt.NDArray[np.float64] = 1.0 + 0.04 * np.arange(n) / 365.0

    # Ruido gaussiano ~5% del valor base
    noise = 1.0 + rng.normal(loc=0.0, scale=0.05, size=n)

    total_income = base_income * wk_factor * mo_factor * qt_factor * trend * noise

    # Reparto proporcional entre 3 medios de pago con ruido individual
    p_efectivo = np.clip(0.30 + rng.normal(0, 0.02, size=n), 0.05, 0.60)
    p_tarjeta = np.clip(0.50 + rng.normal(0, 0.02, size=n), 0.10, 0.80)
    p_transferencia = np.clip(1.0 - p_efectivo - p_tarjeta, 0.05, 0.60)
    # Re-normalizar a suma 1
    p_sum = p_efectivo + p_tarjeta + p_transferencia

    ingreso_efectivo: npt.NDArray[np.float64] = (
        total_income * (p_efectivo / p_sum)
    )
    ingreso_tarjeta: npt.NDArray[np.float64] = (
        total_income * (p_tarjeta / p_sum)
    )
    ingreso_transferencia: npt.NDArray[np.float64] = (
        total_income * (p_transferencia / p_sum)
    )

    # ── Egresos ─────────────────────────────────────────────────────────────
    # Nómina: quincenal (~15,000 los días 15 y fin de mes)
    nomina: npt.NDArray[np.float64] = np.zeros(n)
    is_end_month = dates.is_month_end
    quincena_mask: npt.NDArray[np.bool_] = (day == 15) | is_end_month
    n_nomina = int(quincena_mask.sum())
    nomina[quincena_mask] = rng.normal(loc=15_000, scale=500, size=n_nomina)
    nomina = np.clip(nomina, 0, None)

    # Proveedores: ~20,000/día con autocorrelación (simula retraso N(37,7))
    rho = 0.6
    prov: npt.NDArray[np.float64] = np.full(n, 20_000, dtype=np.float64)
    innov = rng.normal(loc=0.0, scale=4_000, size=n)
    for i in range(1, n):
        prov[i] = 20_000 + rho * (prov[i - 1] - 20_000) + innov[i]
    prov = np.clip(prov, 5_000, 50_000)

    # Combustible: ~8,000/día con alta volatilidad (σ=0.40 en log)
    combustible = rng.lognormal(mean=np.log(8_000), sigma=0.40, size=n)

    # Mantenimiento: ~3,000/día con picos en cierre de trimestre
    mantenimiento = rng.lognormal(mean=np.log(3_000), sigma=0.20, size=n)
    is_quarter_end = dates.is_quarter_end
    n_qpeak = int(is_quarter_end.sum())
    q_peak = rng.uniform(low=1.3, high=1.8, size=n_qpeak)
    mantenimiento[is_quarter_end] = (
        mantenimiento[is_quarter_end] * q_peak
    )

    # Servicios: ~2,000/día (agua, luz, internet, telefonía)
    servicios = rng.lognormal(mean=np.log(2_000), sigma=0.15, size=n)

    gasto_operativo = nomina + prov + combustible + mantenimiento + servicios

    # Gastos extraordinarios: media baja, varianza alta (~5% de los días)
    gasto_extraordinario: npt.NDArray[np.float64] = np.zeros(n)
    n_extra = max(1, int(n * 0.05))
    extra_idx = rng.choice(n, size=n_extra, replace=False)
    gasto_extraordinario[extra_idx] = rng.exponential(scale=50_000, size=n_extra)

    # ── Anomalías ───────────────────────────────────────────────────────────
    (
        tiene_anomalia,
        tipo_anomalia,
        ingreso_efectivo,
        ingreso_tarjeta,
        ingreso_transferencia,
        gasto_operativo,
    ) = _asignar_anomalias(
        rng=rng,
        n=n,
        anomaly_rate=anomaly_rate,
        ingreso_efectivo=ingreso_efectivo,
        ingreso_tarjeta=ingreso_tarjeta,
        ingreso_transferencia=ingreso_transferencia,
        gasto_operativo=gasto_operativo,
        month=month,
    )

    # ── Saldos (recalculados tras anomalías) ────────────────────────────────
    total_gasto = gasto_operativo + gasto_extraordinario
    total_ingreso = (
        ingreso_efectivo + ingreso_tarjeta + ingreso_transferencia
    )
    saldo_diario = total_ingreso - total_gasto
    saldo_acumulado = np.cumsum(saldo_diario)

    # ── Flags temporales ────────────────────────────────────────────────────
    es_festivo = _build_holiday_mask(dates)
    es_finde: npt.NDArray[np.bool_] = dow >= 5

    # ── DataFrame final ─────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "fecha": dates,
            "ingreso_efectivo": ingreso_efectivo,
            "ingreso_tarjeta": ingreso_tarjeta,
            "ingreso_transferencia": ingreso_transferencia,
            "gasto_operativo": gasto_operativo,
            "gasto_extraordinario": gasto_extraordinario,
            "saldo_diario": saldo_diario,
            "saldo_acumulado": saldo_acumulado,
            "es_festivo": es_festivo,
            "es_finde": es_finde,
            "tiene_anomalia": tiene_anomalia,
            "tipo_anomalia": tipo_anomalia,
        }
    )

    # Persistir a CSV
    out_path = "data/raw/dataset_raw.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "Dataset guardado en %s (%d filas, %d anomalías — %.1f%%)",
        out_path,
        len(df),
        int(tiene_anomalia.sum()),
        100.0 * float(tiene_anomalia.mean()),
    )

    return df
