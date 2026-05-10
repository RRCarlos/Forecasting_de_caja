"""
Exporta datos optimizados para Power BI Dashboard.

Genera:
- reports/forecast_powerbi.csv: 17 columnas con datos históricos y forecast
- docs/dashboard_design.md: documento de diseño del dashboard

Uso:
    from src.dashboards.export_powerbi import export_powerbi
    result = export_powerbi()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Columnas del CSV de salida
_COLUMNAS_EXPORT: List[str] = [
    "fecha",
    "caja_neta",
    "ingreso_efectivo",
    "ingreso_tarjeta",
    "ingreso_transferencia",
    "gasto_operativo",
    "gasto_extraordinario",
    "saldo_diario",
    "yhat",
    "yhat_lower_80",
    "yhat_upper_80",
    "yhat_lower",
    "yhat_upper",
    "es_anomalia",
    "severidad",
    "tipo_anomalia",
    "horizonte",
]

_RUTAS: Dict[str, str] = {
    "features": "data/curated/dataset_features.csv",
    "anomaly_report": "reports/anomaly_report.csv",
    "consensus": "reports/consensus_matrix.csv",
    "forecast": "reports/forecast_results.csv",
}


def _cargar_datasets() -> Dict[str, pd.DataFrame]:
    """Carga todos los datasets necesarios desde CSV.

    Returns:
        Dict con DataFrames: features, anomaly_report, consensus, forecast.

    Raises:
        FileNotFoundError: Si algún archivo no existe.
    """
    cargados: Dict[str, pd.DataFrame] = {}
    for nombre, ruta in _RUTAS.items():
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"Archivo {ruta} no encontrado. "
                f"Ejecuta el pipeline completo primero."
            )
        cargados[nombre] = pd.read_csv(ruta)
        logger.info("Cargado %s: %d filas", ruta, len(cargados[nombre]))
    return cargados


def _preparar_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepara el DataFrame base con datos históricos.

    Toma dataset_features.csv y selecciona las columnas de origen.

    Args:
        datasets: Dict con DataFrames cargados.

    Returns:
        DataFrame base con columnas históricas.
    """
    df_feat = datasets["features"].copy()
    df_feat["fecha"] = pd.to_datetime(df_feat["fecha"])

    cols_base: List[str] = [
        "fecha", "caja_neta", "ingreso_efectivo", "ingreso_tarjeta",
        "ingreso_transferencia", "gasto_operativo", "gasto_extraordinario",
        "saldo_diario",
    ]
    return df_feat[cols_base].copy()


def _mergear_anomalias(
    df_base: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Mergea información de anomalías al DataFrame base.

    Args:
        df_base: DataFrame base con datos históricos.
        datasets: Dict con DataFrames cargados.

    Returns:
        DataFrame con columnas de anomalías añadidas.
    """
    df_anom = datasets["anomaly_report"].copy()
    df_anom["fecha"] = pd.to_datetime(df_anom["fecha"])
    df_anom = df_anom.rename(columns={"metodos_detectores": "tipo_anomalia"})
    df_anom["es_anomalia"] = df_anom["severidad"].notna()

    # Merge por fecha (left join para conservar filas del features)
    df_merged = df_base.merge(
        df_anom[["fecha", "es_anomalia", "severidad", "tipo_anomalia"]],
        on="fecha",
        how="left",
    )
    return df_merged


def _mergear_forecast(
    df_base: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Mergea predicciones del forecast al DataFrame base.

    Las filas históricas tienen valores NaN en yhat.
    Las filas futuras tienen valores NaN en caja_neta.

    Args:
        df_base: DataFrame con datos históricos + anomalías.
        datasets: Dict con DataFrames cargados.

    Returns:
        DataFrame con forecast añadido.
    """
    df_fc = datasets["forecast"].copy()
    df_fc = df_fc.rename(columns={"ds": "fecha"})
    df_fc["fecha"] = pd.to_datetime(df_fc["fecha"])

    # Merge por fecha (outer para incluir futuras)
    df_merged = df_base.merge(
        df_fc[["fecha", "yhat", "yhat_lower_80", "yhat_upper_80",
               "yhat_lower", "yhat_upper", "horizonte"]],
        on="fecha",
        how="outer",
    )

    # Ordenar por fecha
    df_merged = df_merged.sort_values("fecha").reset_index(drop=True)
    return df_merged


def _completar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Completa valores por defecto en columnas del export.

    Args:
        df: DataFrame parcialmente mergeado.

    Returns:
        DataFrame con columnas completadas.
    """
    result = df.copy()

    # Completar es_anomalia (filas futuras → False)
    if "es_anomalia" in result.columns:
        result["es_anomalia"] = result["es_anomalia"].fillna(False)

    # Completar severidad y tipo_anomalia
    for col in ["severidad", "tipo_anomalia"]:
        if col in result.columns:
            result[col] = result[col].fillna("")

    # Asegurar columnas de forecast para filas históricas
    for col in ["yhat", "yhat_lower_80", "yhat_upper_80",
                "yhat_lower", "yhat_upper"]:
        if col not in result.columns:
            result[col] = float("nan")

    # Completar horizonte para filas históricas
    if "horizonte" not in result.columns:
        result["horizonte"] = ""
    else:
        result["horizonte"] = result["horizonte"].fillna("")

    return result


def _seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y ordena las columnas del export final.

    Args:
        df: DataFrame completo.

    Returns:
        DataFrame solo con columnas de exportación.
    """
    cols_disponibles = [c for c in _COLUMNAS_EXPORT if c in df.columns]
    return df[cols_disponibles].copy()


def generate_dashboard_design(
    output_path: str = "docs/dashboard_design.md",
) -> str:
    """Genera documento de diseño del dashboard Power BI.

    Incluye descripción de 4+ visualizaciones, medidas DAX sugeridas,
    layout y carga de datos.

    Args:
        output_path: Ruta del archivo .md a generar.

    Returns:
        Ruta del archivo generado.
    """
    content = r"""# Dashboard Power BI — Forecasting de Caja Netas

## 1. Carga de Datos

### Origen
- **Archivo**: `reports/forecast_powerbi.csv`
- **Formato**: CSV con 17 columnas, ~789 filas (~699 históricas + ~90 forecast)
- **Frecuencia**: Diaria

### Instrucciones de Carga
1. En Power BI Desktop: *Obtener datos* → *Text/CSV*
2. Seleccionar `reports/forecast_powerbi.csv`
3. Asegurar que `fecha` se detecte como tipo *Date*
4. Verificar que `es_anomalia` se detecte como *True/False*
5. Click en *Cargar*

### Columnas
| Columna | Tipo | Descripción |
|---------|------|-------------|
| fecha | Date | Fecha del registro |
| caja_neta | Decimal | Caja neta real (histórico) |
| ingreso_efectivo | Decimal | Ingreso en efectivo |
| ingreso_tarjeta | Decimal | Ingreso con tarjeta |
| ingreso_transferencia | Decimal | Ingreso por transferencia |
| gasto_operativo | Decimal | Gasto operativo total |
| gasto_extraordinario | Decimal | Gasto extraordinario |
| saldo_diario | Decimal | Saldo del día |
| yhat | Decimal | Predicción de caja neta |
| yhat_lower_80 | Decimal | IC inferior 80% |
| yhat_upper_80 | Decimal | IC superior 80% |
| yhat_lower | Decimal | IC inferior 95% |
| yhat_upper | Decimal | IC superior 95% |
| es_anomalia | Bool | Si el día es anómalo |
| severidad | Text | crítico/alto/medio/bajo |
| tipo_anomalia | Text | Método(s) detector(es) |
| horizonte | Text | 30d/60d/90d |

---

## 2. Visualizaciones Propuestas

### V1 — KPI Cards (Fila superior)
**Tipo**: 4 tarjetas de KPI

Métricas:
- **Caja Neta Actual**: Último valor real del período
  ```
  Caja Neta Actual = 
  VAR UltimaFecha = MAX(forecast_powerbi[fecha])
  VAR UltimoValorReal = CALCULATE(
      SUM(forecast_powerbi[caja_neta]),
      forecast_powerbi[fecha] = UltimaFecha,
      forecast_powerbi[caja_neta] <> BLANK()
  )
  RETURN UltimoValorReal
  ```

- **Promedio 30d**: Media móvil de caja neta últimos 30 días
  ```
  Promedio 30d Caja = 
  CALCULATE(
      AVERAGE(forecast_powerbi[caja_neta]),
      DATESINPERIOD(forecast_powerbi[fecha], MAX(forecast_powerbi[fecha]), -30, DAY),
      forecast_powerbi[caja_neta] <> BLANK()
  )
  ```

- **Forecast Próximo 30d**: Suma de predicciones a 30 días
  ```
  Forecast 30d = 
  CALCULATE(
      SUM(forecast_powerbi[yhat]),
      forecast_powerbi[horizonte] = "30d"
  )
  ```

- **Varianza vs Forecast**: Diferencia real vs predicho (últimos 30d históricos)
  ```
  Varianza Forecast = 
  VAR Real30d = [Promedio 30d Caja]
  VAR Forecast30d = [Forecast 30d] / 30
  RETURN Real30d - Forecast30d
  ```

### V2 — Time Series: Caja Neta Histórica + Forecast (Centro)
**Tipo**: Gráfico de líneas combinado

Elementos:
- Lnea de caja_neta real (azul, histórica)
- Lnea de yhat (roja punteada, forecast)
- Banda de IC 80% (sombreado más oscuro)
- Banda de IC 95% (sombreado más claro)
- Lnea vertical divisoria histórico/futuro
- Tooltip con valores completos

Eje X: fecha (continuo)
Eje Y: Caja neta ($)

### V3 — Anomaly Heatmap (Lateral derecho)
**Tipo**: Mapa de calor (Heatmap)

Eje X: Mes
Eje Y: Severidad (crítico, alto, medio, bajo)

```
Anomaly Heatmap = 
SUMMARIZE(
    forecast_powerbi,
    forecast_powerbi[mes],
    forecast_powerbi[severidad],
    "Conteo", COUNTROWS(forecast_powerbi)
)
```

Nota: Crear columna `mes` calculada:
```
mes = MONTH(forecast_powerbi[fecha])
```

### V4 — Tabla de Alertas (Fila inferior)
**Tipo**: Tabla con formato condicional

Columnas: fecha, severidad, caja_neta, yhat, tipo_anomalia

Filtros: Top 20 anomalías del período (severidad no vacía)
Formato condicional: Severidad → color (rojo=crítico, naranja=alto, amarillo=medio, verde=bajo)

---

## 3. Medidas DAX Adicionales

Además de las ya incluidas en las visualizaciones:

### Total de Anomalías
```
Total Anomalías = 
CALCULATE(
    COUNTROWS(forecast_powerbi),
    forecast_powerbi[es_anomalia] = TRUE()
)
```

### Tasa de Anomalías
```
Tasa Anomalías = 
DIVIDE(
    [Total Anomalías],
    COUNTROWS(forecast_powerbi)
)
```

### Caja Neta Acumulada (YTD)
```
Caja Neta YTD = 
TOTALYTD(
    SUM(forecast_powerbi[caja_neta]),
    forecast_powerbi[fecha]
)
```

### Predicción vs Real (últimos 7 días)
```
Precision 7d = 
VAR Real7d = CALCULATE(
    SUM(forecast_powerbi[caja_neta]),
    DATESINPERIOD(forecast_powerbi[fecha], MAX(forecast_powerbi[fecha]), -7, DAY),
    forecast_powerbi[caja_neta] <> BLANK()
)
VAR Pred7d = CALCULATE(
    SUM(forecast_powerbi[yhat]),
    DATESINPERIOD(forecast_powerbi[fecha], MAX(forecast_powerbi[fecha]), -7, DAY),
    forecast_powerbi[yhat] <> BLANK()
)
RETURN DIVIDE(Real7d - Pred7d, Pred7d)
```

### Máximo Drawdown (30d)
```
Max Drawdown 30d = 
MINX(
    DATESINPERIOD(forecast_powerbi[fecha], MAX(forecast_powerbi[fecha]), -30, DAY),
    forecast_powerbi[caja_neta]
)
```

### SMAPE Estimado
```
SMAPE Estimado = 
VAR Real = SUM(forecast_powerbi[caja_neta])
VAR Pred = SUM(forecast_powerbi[yhat])
VAR Numerador = 2 * ABS(Real - Pred)
VAR Denominador = ABS(Real) + ABS(Pred)
RETURN DIVIDE(Numerador, Denominador)
```

---

## 4. Diseño de Layout (Descripción Textual)

```
+--------------------------------------------------+
|  [Logo]  Dashboard de Caja — Forecasting Logística |
+--------------------------------------------------+
|  [KPI: Caja] | [KPI: Prom30d] | [KPI: Fcst30d] | [KPI: Varianza] |
+--------------------------------------------------+
|                                          |  [Heatmap: Anomalías] |
|  [Time Series: Histórico + Forecast]     |  por Mes y Severidad  |
|                                          |                        |
|                                          +------------------------+
|                                          |  [Filtros: Rango Fecha]|
+--------------------------------------------------+
|  [Tabla: Top 20 Anomalías del Período]            |
|  (fecha, severidad, caja, yhat, tipo)             |
+--------------------------------------------------+
```

### Layout Responsivo
- **Escritorio**: 4 tarjetas arriba, time series 70% + heatmap 30% centro, tabla 100% abajo
- **Tablet/Móvil**: KPIs en 2×2, time series 100%, heatmap debajo, tabla al final
- **Segmentadores**: Rango de fechas, severidad, horizonte

---

## 5. Refresco y Mantenimiento

- **Frecuencia**: Diario (programar en Power BI Service)
- **Alertas**: Configurar alerta por correo si `Tasa Anomalías` supera 10%
- **Histórico**: Los datos históricos son estáticos; solo se agregan nuevos días

---

*Documento generado automáticamente por export_powerbi.py*
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Dashboard design guardado: %s", output_path)
    return output_path


def export_powerbi(output_dir: str = "reports") -> Dict[str, Any]:
    """Exporta datos optimizados para Power BI Dashboard.

    Carga dataset_features.csv, anomaly_report.csv, consensus_matrix.csv
    y forecast_results.csv. Mergea todo en un solo DataFrame por fecha y
    guarda el resultado en {output_dir}/forecast_powerbi.csv.

    También genera docs/dashboard_design.md.

    Args:
        output_dir: Directorio donde guardar el CSV de exportación.

    Args:
        output_dir: Directorio donde guardar los archivos.

    Returns:
        Dict con rutas de archivos generados:
            - powerbi_csv: ruta del CSV
            - dashboard_design: ruta del documento de diseño
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== Exportando datos para Power BI ===")

    # 1. Cargar datasets
    datasets = _cargar_datasets()

    # 2. Preparar datos base
    df_base = _preparar_features(datasets)
    logger.info("Base preparada: %d filas", len(df_base))

    # 3. Mergear anomalías
    df_con_anomalias = _mergear_anomalias(df_base, datasets)

    # 4. Mergear forecast
    df_completo = _mergear_forecast(df_con_anomalias, datasets)
    logger.info(
        "Merge completado: %d filas (históricas + forecast)",
        len(df_completo),
    )

    # 5. Completar columnas
    df_completo = _completar_columnas(df_completo)

    # 6. Seleccionar columnas finales
    df_export = _seleccionar_columnas(df_completo)

    # 7. Guardar CSV
    csv_path = os.path.join(output_dir, "forecast_powerbi.csv")
    df_export.to_csv(csv_path, index=False)
    logger.info(
        "Power BI CSV guardado: %s (%d filas, %d columnas)",
        csv_path,
        len(df_export),
        df_export.shape[1],
    )

    # 9. Generar documento de diseño (siempre en docs/)
    md_path = generate_dashboard_design(
        output_path="docs/dashboard_design.md",
    )

    logger.info("=== Exportación Power BI completada ===")

    return {
        "powerbi_csv": csv_path,
        "dashboard_design": md_path,
    }
