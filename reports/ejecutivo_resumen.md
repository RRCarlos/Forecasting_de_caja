# Resumen Ejecutivo — Forecasting de Caja

**Lectura estimada:** 60 segundos

---

## Indicadores Clave (KPIs)

| Indicador | Valor |
|-----------|-------|
| **Caja neta actual** | $3,804.75 |
| **Promedio 30 das** | $16,997.23 |
| **Forecast prximos 30d** | $400,637.69 |
| **Anomalas detectadas** | 162 |
| **SMAPE del modelo** | 16.75% |

---

## Estado del Forecasting

- **Tendencia:** Decreciente
- **Estacionalidad dominante:** Noviembre (mayor caja neta promedio)
- **Confianza del modelo:** Baja

---

## Anomalas

**Total:** 162 en el perodo
- Alto: 16
- Medio: 103
- Bajo: 30

### Top 3 Anomalas

| Fecha | Caja Neta | Severidad | Mtodo |
|-------|-----------|-----------|-------|
| 2024-05-17 | $-50,332.69 | alto | Isolation Forest, IQR, Benford |
| 2024-06-03 | $-64,678.03 | alto | Isolation Forest, IQR, Benford |
| 2024-11-20 | $60,572.60 | alto | Isolation Forest, IQR, Benford |

---

## Prximos 30 Das

- **Valor mnimo esperado:** $-36,475.92
- **Valor mximo esperado:** $51,794.86
- **Perodo del forecast:** 2024-12-30 — 2025-01-28
- **Confianza:** Baja

---

## Recomendaciones

1. La caja neta actual se encuentra por debajo del valor proyectado (diferencia de $9,549.84). Monitorear de cerca la ejecucin presupuestal.

2. No se detectaron anomalas crticas en el perodo. Los procesos de control operan dentro de lo esperado.

3. La proyeccin de caja neta muestra tendencia decreciente. Se recomienda revisar la estructura de gastos y evaluar medidas preventivas de optimizacin.

---

### Metadata

- **Fecha de generacin:** 2026-05-10 11:33
- **Periodo analizado:** 2024-01-31 — 2024-12-29
- **Registros analizados:** 334
- **Modelo:** Prophet (seasonality_mode=additive, changepoint_prior_scale=0.05)

---

*Documento generado automticamente — Mini-Modelo de Forecasting de Caja para Logstica*