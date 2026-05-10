# Dashboard Power BI — Forecasting de Caja Netas

## 1. Carga de Datos

### Origen
- **Archivo**: `dashboards/forecast_powerbi.csv`
- **Formato**: CSV con 17 columnas, ~789 filas (~699 históricas + ~90 forecast)
- **Frecuencia**: Diaria

### Instrucciones de Carga
1. En Power BI Desktop: *Obtener datos* → *Text/CSV*
2. Seleccionar `dashboards/forecast_powerbi.csv`
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
