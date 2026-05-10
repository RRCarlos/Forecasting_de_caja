"""
Prueba de la Ley de Benford para detección de anomalías en montos financieros.

La Ley de Benford establece que en conjuntos de datos naturales, la
probabilidad del primer dígito d es P(d) = log10(1 + 1/d). Desviaciones
significativas pueden indicar manipulación o anomalías.

Uso:
    from src.anomalies.benford import benford_test, detect_benford_anomalies
    resultado = benford_test(df)
    anomalias = detect_benford_anomalies(df)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Distribución esperada de Benford para dígitos 1-9
_DIGITOS = list(range(1, 10))
_BENFORD_ESPERADO: dict[int, float] = {
    d: np.log10(1.0 + 1.0 / d) for d in _DIGITOS
}


def _primer_digito(serie: pd.Series) -> pd.Series:
    """Extrae el primer dígito de cada valor absoluto en la serie.

    Solo considera valores > 0. Los valores <= 0 o nulos producen NaN.

    Args:
        serie: Serie numérica con montos.

    Returns:
        Serie con el primer dígito (1-9) o NaN si el valor es inválido.
    """
    # Tomar valor absoluto
    abs_vals = serie.abs()
    # Evitar warnings por log10(0) o valores negativos
    abs_vals = abs_vals.replace(0, np.nan)
    # Solo operar sobre valores válidos (> 0)
    mascara_valida = abs_vals.notna() & (abs_vals > 0)

    resultado = pd.Series(np.nan, index=serie.index, dtype=float)

    if mascara_valida.any():
        vals = abs_vals[mascara_valida]
        # Primer dígito: floor(log10(x)) → cuántos dígitos - 1
        # Luego x / 10^n dígitos → primer dígito
        ordem = np.floor(np.log10(vals))
        primeiro = np.floor(vals / (10**ordem))
        # Solo 1-9 son válidos
        mascara_digito = (primeiro >= 1) & (primeiro <= 9)
        resultado.loc[mascara_valida] = primeiro.where(mascara_digito, np.nan)

    return resultado


def _calcular_benford_para_grupo(
    valores: pd.Series,
    min_transactions: int,
    alpha: float,
) -> dict:
    """Calcula estadísticos de Benford para un grupo de valores.

    Args:
        valores: Serie de montos del grupo.
        min_transactions: Mínimo de transacciones requeridas.
        alpha: Nivel de significancia para chi-cuadrado.

    Returns:
        Dict con keys: chi2_stat, p_value, mad, sospechoso, n.
        Si no hay suficientes datos, chi2_stat, p_value y mad son NaN.
    """
    n_total = len(valores)
    if valores.empty or n_total < min_transactions:
        return {
            "chi2_stat": float("nan"),
            "p_value": float("nan"),
            "mad": float("nan"),
            "sospechoso": False,
            "n": n_total,
        }

    digitos = _primer_digito(valores).dropna()
    n = len(digitos)

    if n < min_transactions:
        return {
            "chi2_stat": float("nan"),
            "p_value": float("nan"),
            "mad": float("nan"),
            "sospechoso": False,
            "n": n,
        }

    # Frecuencias observadas
    observadas = np.array(
        [float((digitos == d).sum()) for d in _DIGITOS], dtype=np.float64
    )

    # Frecuencias esperadas según Benford
    esperadas = np.array(
        [_BENFORD_ESPERADO[d] * n for d in _DIGITOS], dtype=np.float64
    )

    # Chi-cuadrado
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_stat = float(np.nansum((observadas - esperadas) ** 2 / esperadas))
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=8)

    # MAD (Mean Absolute Deviation)
    proporciones_obs = observadas / n
    proporciones_esp = esperadas / n
    mad = float(np.mean(np.abs(proporciones_obs - proporciones_esp)))

    # Criterio: MAD > 0.015 es sospechoso
    sospechoso = bool(mad > 0.015)

    return {
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "mad": mad,
        "sospechoso": sospechoso,
        "n": n,
    }


def benford_test(
    df: pd.DataFrame,
    amount_column: str = "caja_neta",
    group_column: str | None = None,
    min_transactions: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aplica la prueba de la Ley de Benford sobre montos financieros.

    Calcula la distribución del primer dígito y la compara con la
    distribución esperada de Benford usando chi-cuadrado y MAD.

    Args:
        df: DataFrame con los datos financieros.
        amount_column: Columna con los montos a analizar.
        group_column: Columna para agrupar (ej: tipo de transacción,
            proveedor). Si es None, trata todo como un solo grupo.
        min_transactions: Mínimo de transacciones requeridas por grupo
            para aplicar la prueba. Grupos con menos se omiten (NaN).
        alpha: Nivel de significancia para la prueba chi-cuadrado.

    Returns:
        DataFrame con columnas:
            - grupo: identificador del grupo (o 'global' si sin agrupar)
            - chi2_stat: estadístico chi-cuadrado
            - p_value: valor p de la prueba
            - mad: Mean Absolute Deviation vs Benford
            - sospechoso: True si MAD > 0.015
            - n: número de transacciones válidas en el grupo
    """
    if df.empty:
        logger.warning("DataFrame vacío — no se puede aplicar Benford")
        return pd.DataFrame(
            columns=["grupo", "chi2_stat", "p_value", "mad", "sospechoso", "n"]
        )

    if amount_column not in df.columns:
        raise ValueError(
            f"Columna de montos '{amount_column}' no encontrada en el DataFrame"
        )

    if group_column is not None and group_column not in df.columns:
        raise ValueError(
            f"Columna de agrupación '{group_column}' no encontrada en el DataFrame"
        )

    valores = df[amount_column].dropna()
    if valores.empty:
        logger.warning("Benford: no hay valores no nulos en '%s'", amount_column)
        return pd.DataFrame(
            columns=["grupo", "chi2_stat", "p_value", "mad", "sospechoso", "n"]
        )

    resultados: List[dict] = []

    if group_column is None:
        # Un solo grupo global
        res = _calcular_benford_para_grupo(valores, min_transactions, alpha)
        res["grupo"] = "global"
        resultados.append(res)
    else:
        # Agrupar por la columna indicada
        for grupo_name, grupo_df in df.groupby(group_column, observed=True):
            grupo_valores = grupo_df[amount_column].dropna()
            res = _calcular_benford_para_grupo(
                grupo_valores, min_transactions, alpha
            )
            res["grupo"] = str(grupo_name)
            resultados.append(res)

    df_resultado = pd.DataFrame(resultados)
    df_resultado = df_resultado[["grupo", "chi2_stat", "p_value", "mad", "sospechoso", "n"]]

    n_sospechosos = df_resultado["sospechoso"].sum()
    logger.info(
        "Benford: %d/%d grupos evaluados, %d sospechosos (MAD > 0.015)",
        (~df_resultado["mad"].isna()).sum(),
        len(df_resultado),
        n_sospechosos,
    )

    return df_resultado


def detect_benford_anomalies(
    df: pd.DataFrame,
    amount_column: str = "caja_neta",
    group_column: str | None = None,
    min_transactions: int = 30,
    alpha: float = 0.05,
) -> pd.Series:
    """Detecta registros que pertenecen a grupos que violan la Ley de Benford.

    Aplica benford_test y retorna una máscara booleana donde True indica
    que el registro pertenece a un grupo sospechoso.

    Args:
        df: DataFrame con los datos financieros.
        amount_column: Columna con los montos a analizar.
        group_column: Columna para agrupar.
        min_transactions: Mínimo de transacciones por grupo.
        alpha: Nivel de significancia.

    Returns:
        Serie booleana con mismo índice que df: True si el registro
        pertenece a un grupo que viola Benford.
    """
    if df.empty:
        return pd.Series([], dtype=bool, name="benford_anomalia")

    resultado_benford = benford_test(
        df=df,
        amount_column=amount_column,
        group_column=group_column,
        min_transactions=min_transactions,
        alpha=alpha,
    )

    # Si no hay grupos sospechosos, todo es False
    grupos_sospechosos = set(
        resultado_benford.loc[resultado_benford["sospechoso"], "grupo"].tolist()
    )

    if not grupos_sospechosos:
        logger.info("Benford: ningún grupo sospechoso detectado")
        return pd.Series(False, index=df.index, name="benford_anomalia", dtype=bool)

    if group_column is None:
        if "global" in grupos_sospechosos:
            # Si el grupo global es sospechoso, NO marcamos todo.
            # En lugar de eso, marcamos solo los registros individuales
            # cuyo primer dígito es poco probable según Benford.
            # Esto evita falsos positivos masivos cuando el análisis
            # global no es aplicable (ej: caja_neta como flujo neto).
            digitos = _primer_digito(df[amount_column])
            # Umbral: marcar dígitos con probabilidad Benford < 8%
            # (dígitos 7=5.8%, 8=5.1%, 9=4.6% son los más raros)
            umbral_digito = 0.08
            anomalias = digitos.apply(
                lambda d: _BENFORD_ESPERADO.get(int(d), 0.0) < umbral_digito
                if pd.notna(d) else False
            )
        else:
            anomalias = pd.Series(False, index=df.index)
    else:
        anomalias = df[group_column].astype(str).isin(grupos_sospechosos)

    anomalias = anomalias.fillna(False).astype(bool)

    n_anomalias = anomalias.sum()
    logger.info(
        "Benford: %d registros en grupos sospechosos detectados",
        n_anomalias,
    )

    return anomalias
