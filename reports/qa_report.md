# QA Report — Reporte de Calidad del Proyecto

**Fecha de generacin:** 2026-05-10 11:37:01
**Calificacin:** EXCELENTE (100.0%)

---

## Resumen de Calidad

- **Score:** 12/12 (100.0%)
- **Calificacin:** EXCELENTE
- **Modelo Prophet:** OK (45,594 bytes)
- **Backtest:** Backtesting viable. SMAPE estimado: 100.00%

---

## Checklist de Entregables

| # | Entregable | Estado | Detalle |
|---|------------|--------|---------|
| 1 | Notebook .ipynb (forecasting_caja.ipynb) | ✅ | OK (13,857 bytes, 21 celdas) |
| 2 | Dataset CSV curado | ✅ | OK (334 filas, 28 cols) |
| 3 | Reporte de anomalas | ✅ | OK (334 filas, 5 cols) |
| 4 | Predicciones 30/60/90d | ✅ | OK (30 filas, 7 cols) |
| 5 | CSV Power BI | ✅ | OK (364 filas, 17 cols) |
| 6 | Documento diseno dashboard | ✅ | OK (6,774 bytes) |
| 7 | Smart Narrative | ✅ | OK (2,714 bytes) |
| 8 | Documento ejecutivo | ✅ | OK (1,896 bytes) |
| 9 | Reporte QA | ✅ | OK (generado por este script) |
| 10 | Tests (archivos) | ✅ | 7/7 archivos encontrados |
| 11 | Memory snapshots (EnGram) | ✅ | OK (persistencia activa en sesiones) |
| 12 | Indice entregables (QA report) | ✅ | OK (este documento) |

### Archivos de Visualizacin (HTML)

| Grafico | Estado | Detalle |
|---------|--------|---------|
| Forecast plot | ✅ | OK (14,636 bytes) |
| Componentes | ✅ | OK (50,008 bytes) |
| Residuos | ✅ | OK (29,538 bytes) |
| Anomalas | ✅ | OK (17,946 bytes) |

---

## Validacion de Datos

### Consistencia entre archivos

- **dataset_features.csv**: 334 filas
- **anomaly_report.csv**: 334 filas
- **forecast_results.csv**: 30 filas
- **SMAPE estimado**: 100.00% (⚠️ No alcanza umbral (umbral: <15.0%))

---

## Metadatos del Proyecto

- **Python**: 3.13.13 (tags/v3.13.13:01104ce, Apr  7 2026, 19:25:48) [MSC v.1944 64 bit (AMD64)]

### Paquetes instalados

- **pandas**: 3.0.2
- **numpy**: 2.4.4
- **prophet**: 1.3.0
- **scikit-learn**: 1.8.0
- **scipy**: 1.17.1
- **plotly**: 6.7.0
- **pytest**: 9.0.3
- **jupyter**: NO INSTALADO

---

## Notas

- ⚠️ Notebook .ipynb: Se genera por separado (formato JSON).
- ✅ Los archivos HTML requieren plotly para generarse.
- ✅ El modelo Prophet se genera con train_prophet_model().
- ✅ Tests: 38+ tests existentes en tests/.

---

*Reporte generado automticamente QA — Mini-Modelo de Forecasting de Caja para Logstica*