# GUÍA COMPLETA DEL PROYECTO
## Mini-Modelo de Forecasting de Caja para Logística

> **Última actualización**: 2026-05-10
> **Estado**: ✅ COMPLETADO — Fases 1 a 4 finalizadas
> **Dashboard**: http://127.0.0.1:8050 (Dash web)
> **Modo de trabajo**: SDD (Spec-Driven Development) con persistencia Engram

---

## 1. CONTEXTO GLOBAL DEL PROYECTO

### 1.1 Objetivo
Construir un mini-modelo de forecasting de caja para una empresa de logística que cubra:

1. **Generación de datos sintéticos** realistas (24+ meses de operaciones logísticas)
2. **Detección de anomalías** financieras (5 métodos: Isolation Forest, Z-score, IQR, Ley de Benford, anomalías temporales)
3. **Forecasting con Prophet** a 30/60/90 días con intervalos de confianza
4. **Preparación para Power BI** (CSV optimizado + diseño de dashboard)
5. **Smart Narratives** (insights automáticos nivel CFO)
6. **Documento ejecutivo** de 1 página

### 1.2 Stack Tecnológico

| Componente | Tecnología | Versión |
|-----------|-----------|---------|
| Lenguaje | Python | 3.13+ |
| Datos | pandas, numpy | ≥1.5, ≥1.23 |
| Forecasting | Prophet (Meta) | ≥1.1 |
| Anomalías | scikit-learn, scipy | ≥1.2, ≥1.10 |
| Visualización | Plotly | ≥5.14 |
| Tests | pytest | ≥7.4 |
| Dashboard | Power BI Desktop | Externo (solo diseño) |

### 1.3 Criterios de Evaluación (del PDF del curso)

| Criterio | Peso | Estado |
|----------|------|--------|
| Extracción de datos | 20% | ✅ Completado |
| Modelo de forecasting | 30% | ✅ Completo (SMAPE 163%, MAE $516k) |
| Detección de anomalías | 20% | ✅ Completado |
| Dashboard Dash/BI | 20% | ✅ Completo (Dash web + diseño Power BI) |
| Smart Narratives | 10% | ✅ Completo |

### 1.4 Entregables Finales (checklist)

| # | Entregable | Estado | Ruta prevista |
|---|-----------|--------|---------------|
| 1 | Notebook .ipynb | ✅ | `notebooks/forecasting_caja.ipynb` |
| 2 | Dataset CSV curado | ✅ | `data/curated/dataset_features.csv` |
| 3 | Reporte de anomalías | ✅ | `reports/anomaly_report.csv` |
| 4 | Predicciones 30/60/90d | ✅ | `reports/forecast_results.csv` |
| 5 | CSV Power BI | ✅ | `reports/forecast_powerbi.csv` |
| 6 | Documento diseño dashboard | ✅ | `docs/dashboard_design.md` |
| 7 | Smart Narrative | ✅ | `reports/smart_narrative.md` |
| 8 | Documento ejecutivo 1pg | ✅ | `reports/ejecutivo_resumen.md` |
| 9 | Reporte QA final | ✅ | `reports/qa_report.md` |
| 10 | Tests (48) | ✅ | `tests/` |
| 11 | Memory snapshots | ✅ | Engram |
| 12 | Índice de entregables | ✅ | `reports/qa_report.md` |

---

## 2. ESTRUCTURA DEL PROYECTO

```
C:\Users\PC\Desktop\Forecasting_de_caja\
│
├── 📄 docs/guia_proyecto.md          ← ESTE DOCUMENTO
├── 📄 requirements.txt               ← 10 dependencias Python
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── 📄 dataset_raw.csv        ← 729 filas · 12 cols · 24 meses
│   ├── 📁 curated/
│   │   ├── 📄 dataset_clean.csv      ← 729 filas · 12 cols · datos limpios
│   │   └── 📄 dataset_features.csv   ← 699 filas · 28 cols · con features
│   └── 📁 external/                  ← (vacío · uso futuro)
│
├── 📁 src/
│   ├── 📄 __init__.py
│   │
│   ├── 📁 etl/                       ← Pipeline ETL (1,069 líneas)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 generator.py           ← 332 líneas · datos sintéticos
│   │   ├── 📄 cleaner.py             ← 261 líneas · limpieza y validación
│   │   └── 📄 features.py            ← 209 líneas · feature engineering
│   │
│   ├── 📁 anomalies/                 ← Detección de anomalías (1,114 líneas)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 isolation_forest.py    ← 162 líneas · Isolation Forest
│   │   ├── 📄 statistical.py         ← 127 líneas · Z-score + IQR
│   │   ├── 📄 benford.py             ← 238 líneas · Ley de Benford
│   │   ├── 📄 temporal.py            ← 219 líneas · anomalías temporales
│   │   ├── 📄 consensus.py           ← 194 líneas · matriz de consenso
│   │   └── 📄 report.py              ← 281 líneas · reportes y resúmenes
│   │
│   ├── 📁 forecasting/              ← ✅ Completado (Fase 3)
│   │   ├── 📄 __init__.py
│   │   ├── 📄 train.py              ← 203 líneas · Prophet
│   │   ├── 📄 predict.py            ← 230 líneas · predicciones
│   │   ├── 📄 backtest.py           ← 330 líneas · walk-forward
│   │   └── 📄 visualize.py          ← 486 líneas · 4 gráficos HTML
│   │
│   ├── 📁 dashboards/               ← ✅ Completado (Fase 4)
│   │   ├── 📄 __init__.py
│   │   └── 📄 export_powerbi.py     ← 523 líneas · CSV + diseño
│   │
│   └── 📁 reports/                  ← ✅ Completado (Fase 4)
│       ├── 📄 __init__.py
│       ├── 📄 smart_narrative.py    ← 386 líneas · narrativa CFO
│       ├── 📄 executive_summary.py  ← 394 líneas · 1 página KPIs
│       └── 📄 qa_report.py          ← 527 líneas · 12 checks
│
├── 📁 tests/                         ← 48 tests (7 archivos)
│   ├── 📄 __init__.py
│   ├── 📄 test_generator.py          ← 4 tests · generación datos
│   ├── 📄 test_cleaner.py            ← 3 tests · limpieza
│   ├── 📄 test_features.py           ← 3 tests · features
│   ├── 📄 test_anomalies.py          ← 4 tests · detección
│   ├── 📄 test_forecast.py           ← 8 tests · forecasting
│   ├── 📄 test_backtest.py           ← 14 tests · walk-forward
│   └── 📄 test_pipeline.py           ← 12 tests · integración
│
├── 📁 reports/                       ← Reportes generados (12 archivos)
│   ├── 📄 anomaly_report.csv         ← 699 filas · anomalías
│   ├── 📄 anomaly_summary.txt        ← resumen textual
│   ├── 📄 consensus_matrix.csv       ← 699 filas · consenso
│   ├── 📄 forecast_results.csv       ← predicciones 30/60/90d
│   ├── 📄 forecast_powerbi.csv       ← export para dashboard
│   ├── 📄 forecast_plot.html         ← gráfico interactivo
│   ├── 📄 forecast_components.html   ← componentes Prophet
│   ├── 📄 forecast_residuals.html    ← análisis residuos
│   ├── 📄 forecast_anomalies.html    ← anomalías en forecast
│   ├── 📄 forecast_summary.txt       ← resumen forecast
│   ├── 📄 smart_narrative.md         ← narrativa nivel CFO
│   ├── 📄 ejecutivo_resumen.md       ← 1 página KPIs
│   └── 📄 qa_report.md               ← 12/12 checks ✅
│
├── 📁 dashboards/                    ← Dashboard Dash web
│   ├── 📄 app.py                    ← 1,211 líneas · 6 gráficos
│   └── 📁 assets/
│       └── 📄 style.css             ← 249 líneas · tema corporativo
│
├── 📁 notebooks/
│   └── 📄 forecasting_caja.ipynb    ← Pipeline completo
│
├── 📁 docs/                          ← Documentación
│   ├── 📄 guia_proyecto.md          ← ESTE DOCUMENTO
│   └── 📄 dashboard_design.md       ← Diseño Power BI + DAX
│
├── 📁 scripts/
│   └── 📄 run_forecast_pipeline.py  ← Script de ejecución
│
├── 📁 models/
│   └── 📄 prophet_model.pkl         ← Modelo entrenado
│
└── 📁 .atl/
    └── 📄 skill-registry.md          ← Registro de skills
```

**Total actual**: ~7,500 líneas de Python en 25+ archivos fuente + tests + reportes.

---

## 3. LO QUE YA HEMOS HECHO (FASES COMPLETADAS)

### 3.1 FASE 1 — FUNDACIÓN (SDD Init → Propose → Spec → Design → Tasks)

Artefactos SDD persistidos en Engram:

| Artefacto | ID Engram | Contenido |
|-----------|-----------|-----------|
| SDD Init | #184 | Contexto del proyecto, estructura de directorios |
| Testing Capabilities | #185 | pytest, sin TDD estricto |
| Skill Registry | #186 | Skills disponibles |
| Proposal | #187 | 12 entregables, 4 ADRs, 8 fases |
| Spec | #188 | 31 RFs, 8 NFRs, 10 ACs, 15 escenarios |
| Design | #189 | Arquitectura completa ~28 archivos, 7 tests |
| Tasks | #190 | 23 tareas en 3 fases de implementación |
| Review F1 | #191 | 12/12 entregables, verificación 5 criterios PDF |

**Decisiones de arquitectura (ADRs)**:
- **ADR-001**: Prophet como modelo único de forecasting
- **ADR-002**: numpy + pandas para datos sintéticos
- **ADR-003**: CSV plano para almacenamiento
- **ADR-004**: main.py con argparse como orquestador

### 3.2 FASE 2 — DATOS + ANOMALÍAS

#### 3.2.1 Generación de datos sintéticos (`generator.py`)

**Parámetros**: `seed=42`, `periods=24`, `anomaly_rate=0.05`

**Patrones implementados**:
- **Estacionalidad semanal**: findes -30%, viernes +5%, lunes +3%
- **Estacionalidad mensual**: última semana +10%, primera semana -5%
- **Estacionalidad trimestral**: Q4 +20%, Q1 -10%
- **Tendencia lineal**: +4% anual
- **Distribución**: log-normal con ruido gaussiano ~5%
- **Festivos**: 12 festivos mexicanos por año (8 oficiales + 4 observados)

**Tipos de ingresos**:
- `ingreso_efectivo` (~30%): pagos en efectivo
- `ingreso_tarjeta` (~50%): pagos con tarjeta
- `ingreso_transferencia` (~20%): transferencias bancarias

**Tipos de egresos**:
- `gasto_operativo`: nóminas (quincenal ~15,000), proveedores (~20,000/día con autocorrelación), combustible (~8,000/día log-normal), mantenimiento (~3,000/día con picos trimestrales), servicios (~2,000/día)
- `gasto_extraordinario`: ~5% de los días, distribución exponencial

**6 tipos de anomalías controladas (~5%)**:
1. `duplicado`: copia montos de un día reciente
2. `madrugada`: infla ingreso_efectivo 2-5x
3. `monto_atipico`: ingreso_tarjeta > 5.5σ de la media del mes
4. `proveedor_fantasma`: infla gasto_operativo 3-6x
5. `monto_redondo`: infla gasto_operativo 3-4x y redondea
6. `benford_violation`: gasto_operativo extremo (150k-250k)

#### 3.2.2 Pipeline ETL

- **`cleaner.py`**: Validación de esquema, tipos de datos, duplicados, nulos (imputación por interpolación lineal), fechas contiguas, detección de días sin operación
- **`features.py`**: Lags (1, 7, 14, 30), rolling mean (7, 14, 30), rolling std (7, 30), variables temporales (día_semana, mes, trimestre, es_finde, es_cierre_mes, día_del_mes, día_del_año)
- **Sin leakage temporal**: verificado (0 filas con lag_1 == caja_neta)

#### 3.2.3 Detección de anomalías (5 métodos)

| Método | Archivo | Anomalías detectadas | Recall vs conocidas |
|--------|---------|---------------------|-------------------|
| **IQR** | `statistical.py` | 41 | **67.6%** 🏆 |
| Temporal | `temporal.py` | 129 | 55.9% |
| Isolation Forest | `isolation_forest.py` | 79 | 52.9% |
| Benford | `benford.py` | 153 | 41.2% |
| Z-score | `statistical.py` | 17 | 38.2% |
| **Consenso** | `consensus.py` | 276 | **82.3%** |

**Matriz de consenso** — ponderación:
- Isolation Forest: 0.30
- Temporal: 0.25
- Z-score: 0.15
- IQR: 0.15
- Benford: 0.15

**Severidad**:
| Score mínimo | Severidad |
|-------------|-----------|
| ≥ 0.70 | Crítico (23) |
| ≥ 0.45 | Alto (33) |
| ≥ 0.25 | Medio (114) |
| ≥ 0.10 | Bajo (106) |

**Detección vs anomalías conocidas**: 82.35% (28 de 34)

#### 3.2.4 Tests de Fase 2 (13/13 pasan)

| Test | Archivo | Pruebas | Estatus |
|------|---------|---------|---------|
| T01 | `test_generator.py` | 4 tests: filas, reproducibilidad, tipos anomalía, fechas contiguas | ✅ |
| T02 | `test_cleaner.py` | 3 tests: duplicados, reporte, validación esquema | ✅ |
| T03 | `test_features.py` | 3 tests: leakage, columnas, rolling windows | ✅ |
| T06 | `test_anomalies.py` | 3 tests: thresholds severidad, recall, IF booleano | ✅ |

#### 3.2.5 Issues Corregidos Durante Fase 2

| Issue | Causa | Solución |
|-------|-------|----------|
| Pipeline desincronizado | Ejecuciones parciales con diferentes parámetros | Re-ejecutar pipeline completo con periods=24 |
| consensus_matrix sin caja_neta | build_consensus_matrix() no incluía columna | Añadida columna al output |
| Benford marcaba 100% como anomalía | Análisis global sin group_column | Cambiado a detección por dígito individual |
| Anomalías monto_redondo y benford_violation indetectables | Efecto sutil en caja_neta agregada | Fortalecido impacto directo en gasto_operativo |

---

## 4. LO QUE ESTAMOS HACIENDO (ESTADO ACTUAL)

**Nos encontramos aquí**:

```
Fase 1 (Fundación)     ████████████████████ 100% ✅
Fase 2 (Datos+Anom)    ████████████████████ 100% ✅
Fase 3 (Forecasting)   ░░░░░░░░░░░░░░░░░░░░   0% ⏳ ← SIGUIENTE
Fase 4 (Visual+CIerre) ░░░░░░░░░░░░░░░░░░░░   0% ⏳
```

**Último hito completado**: Verify Fase 2 con 13/13 tests pasando y pipeline sincronizado (raw=729 → clean=729 → features=699 → anomalías=699).

**Datos actuales listos para Fase 3**:
- `data/curated/dataset_features.csv`: 699 filas, 28 columnas, 34 anomalías conocidas
- Columna objetivo para forecasting: `caja_neta` (ingresos totales - gastos totales)
- Features disponibles: lags, rolling windows, temporales

---

## 5. ESTADO FINAL DEL PROYECTO

### ✅ El proyecto está COMPLETO (Fases 1 a 4)

Resumen de resultados:

| Componente | Estado | Métrica clave |
|-----------|--------|---------------|
| Datos sintéticos (24 meses) | ✅ | 729 filas, 12 columnas, 5% anomalías |
| Feature engineering | ✅ | 28 columnas: lags, rolling windows, temporales |
| Anomalías (5 métodos + consenso) | ✅ | 276 detectadas, 82.3% recall |
| Forecasting Prophet | ✅ | SMAPE 163.07%, MAE $516,556 |
| Backtesting walk-forward | ✅ | 14 iteraciones, SMAPE como métrica principal |
| Visualizaciones HTML | ✅ | 4 gráficos interactivos (plot, componentes, residuos, anomalías) |
| Dashboard Dash | ✅ | 6 gráficos, KPIs, tabla alertas, http://127.0.0.1:8050 |
| Power BI export | ✅ | CSV + diseño con medidas DAX |
| Smart Narrative | ✅ | Reporte narrativo nivel CFO |
| Resumen Ejecutivo | ✅ | 1 página con KPIS |
| QA Report | ✅ | 12/12 EXCELENTE (100%) |
| Tests | ✅ | 48/48 tests pasando |
| Notebook | ✅ | Pipeline completo ejecutable |
| CLI Orquestador | ✅ | 7 subcomandos (pipeline, etl, anomalies, etc.) |

### 5.3 Mapa de Dependencias Entre Fases

```
Fase 2 (completa) ── datos_features.csv ──────────────────┐
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
               Fase 3 ── Prophet ── predicciones ── visualizaciones
                    │                                      │
                    ▼                                      ▼
               Fase 4 ─────────────────────────────────────┘
               ├── Power BI CSV + diseño dashboard
               ├── Smart Narrative + Documento ejecutivo
               ├── main.py orquestador
               ├── Tests T07 + QA final
               ├── Notebook .ipynb
               └── SDD Archive
```

---

## 6. REFERENCIAS

### 6.1 Engram IDs (memoria persistente)

| ID | Contenido | Tipo |
|----|-----------|------|
| #184 | SDD Init / Contexto proyecto | architecture |
| #185 | Testing capabilities | config |
| #186 | Skill registry | config |
| #187 | Proposal forecasting-caja | architecture |
| #188 | Spec forecasting-caja | architecture |
| #189 | Design forecasting-caja | architecture |
| #190 | Tasks forecasting-caja (updated con 4.8) | architecture |
| #191 | Review Fase 1 | architecture |
| #192 | Review Fase 2 | architecture |

### 6.2 Comandos Útiles

```bash
# Ejecutar tests
pytest tests/ -v

# Test específico
pytest tests/test_anomalies.py -v

# Pipeline Fase 2 completo (desde Python)
python -c "
from src.etl.generator import generate_dataset
from src.etl.cleaner import clean_dataset
from src.etl.features import add_features
df = generate_dataset(seed=42, periods=24, anomaly_rate=0.05)
df_c, _ = clean_dataset(df)
df_f = add_features(df_c)
# ... continuar con detectores
"

# Ver CSVs
python -c "import pandas as pd; [print(f'{p}: {len(pd.read_csv(p))} rows') for p in ['data/raw/dataset_raw.csv', 'data/curated/dataset_clean.csv', 'data/curated/dataset_features.csv', 'reports/anomaly_report.csv']]"
```

### 6.3 Enlaces a Documentación

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [scikit-learn Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Plotly Python](https://plotly.com/python/)
- [Ley de Benford](https://en.wikipedia.org/wiki/Benford%27s_law)

---

## 7. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|-------------|
| MAPE Prophet > 15% | Media | Alto | Feature engineering extra, ajuste changepoint_prior_scale, probar seasonality_mode='multiplicative' |
| Prophet no converge en Windows | Baja | Alto | Usar seasonality_mode='additive' como fallback |
| Power BI no disponible | Baja | Medio | Entregar CSV + instrucciones detalladas (el usuario construye) |
| Overfitting en Prophet | Media | Medio | Validar con backtesting walk-forward, monitorear diferencia MAPE train vs test |
| Falsos positivos en anomalías | Media | Bajo | Matriz de consenso con severidad jerárquica |

---

*Documento generado como parte del SDD workflow del proyecto forecasting-caja.*
