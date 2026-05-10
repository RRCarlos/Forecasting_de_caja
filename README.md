# Forecasting de Caja para Logística

[![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-48%2F48-brightgreen)](#)
[![QA](https://img.shields.io/badge/QA-12%2F12%20%E2%80%94%20EXCELENTE-success)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

> **Mini-modelo de forecasting de caja para una empresa de logística.**  
> Pipeline completo: generación de datos sintéticos → detección de anomalías → forecasting con Prophet → dashboard Dash web → reportes ejecutivos.

---

## Tabla de Contenidos

- [Descripción General](#descripcion-general)
- [Stack Tecnológico](#stack-tecnologico)
- [Arquitectura del Proyecto](#arquitectura-del-proyecto)
- [Instalación y Configuración](#instalacion-y-configuracion)
- [Uso — CLI](#uso--cli)
- [Dashboard Web](#dashboard-web)
- [Pipeline de Datos](#pipeline-de-datos)
  - [1. Generación de Datos Sintéticos](#1-generacion-de-datos-sinteticos)
  - [2. Limpieza y Feature Engineering](#2-limpieza-y-feature-engineering)
  - [3. Detección de Anomalías](#3-deteccion-de-anomalias)
  - [4. Forecasting con Prophet](#4-forecasting-con-prophet)
  - [5. Visualizaciones](#5-visualizaciones)
- [Reportes Generados](#reportes-generados)
- [Tests](#tests)
- [Métricas de Rendimiento](#metricas-de-rendimiento)
- [Estructura de Archivos](#estructura-de-archivos)
- [Licencia](#licencia)

---

## Descripción General

Este proyecto implementa un sistema completo de forecasting de caja neta para una empresa de logística. Los datos sintéticos generados simulan 24 meses de operaciones con:

- **Estacionalidades**: semanal (findes -30%), mensual (última semana +10%), trimestral (Q4 +20%)
- **Tendencia lineal**: +4% anual
- **Festivos mexicanos**: 12 días festivos por año como regresores
- **Anomalías controladas**: ~5% con 6 tipos distintos (picos, caídas, quiebres estructurales, etc.)

El pipeline completo va desde la generación de datos crudos hasta un dashboard web interactivo con KPIs, gráficos de evolución temporal, estacionalidad, análisis de residuos, scatter real-vs-predicho y tabla de alertas.

---

## Stack Tecnológico

| Componente | Tecnología | Versión |
|-----------|-----------|---------|
| Lenguaje | Python | 3.13+ |
| Datos | pandas, numpy | ≥1.5, ≥1.23 |
| Forecasting | Prophet (Meta) | ≥1.1 |
| Anomalías | scikit-learn, scipy | ≥1.2, ≥1.10 |
| Visualización | Plotly | ≥5.14 |
| Dashboard web | Dash + Bootstrap | ≥2.18 |
| Tests | pytest | ≥7.4 |
| Notebook | Jupyter | ≥1.0 |

---

## Arquitectura del Proyecto

```
                     ┌──────────────────┐
                     │   main.py (CLI)  │
                     │   Orquestador    │
                     └──────┬───────────┘
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                    ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │  ETL     │    │  Anomalías   │    │  Forecasting │
   │ generator│──▶ │  Isolation   │──▶ │  Prophet     │
   │ cleaner  │    │  Forest      │    │  predict     │
   │ features │    │  Z-score/IQR │    │  backtest    │
   └──────────┘    │  Benford     │    │  visualize   │
                   │  Temporal    │    └──────┬───────┘
                   │  Consenso    │           │
                   └──────┬───────┘           │
                          ▼                   ▼
                   ┌─────────────────────────────┐
                   │      Reportes (Fase 4)      │
                   │  ┌───────┐ ┌──────┐ ┌────┐ │
                   │  │Smart  │ │Ejec. │ │ QA │ │
                   │  │Narrat.│ │Resum.│ │    │ │
                   │  └───────┘ └──────┘ └────┘ │
                   │  ┌──────────────────────┐  │
                   │  │   Dashboard Dash     │  │
                   │  │  http://127.0.0.1:8050│  │
                   │  └──────────────────────┘  │
                   └─────────────────────────────┘
```

---

## Instalación y Configuración

### Requisitos

- Python 3.13 o superior
- pip (gestor de paquetes)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/RRCarlos/Forecasting_de_caja.git
cd Forecasting_de_caja

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python -m pytest tests/ -v
```

> **Nota**: Prophet requiere compilación de C++. Si tienes problemas, instálalo con `pip install prophet` — en Windows puede necesitar Microsoft C++ Build Tools.

---

## Uso — CLI

El orquestador principal es `main.py` con 7 subcomandos:

```bash
# Pipeline completo (ETL → anomalías → forecast → reportes)
python main.py pipeline

# Ejecutar paso a paso
python main.py etl              # Solo ETL: generar → limpiar → features
python main.py anomalies        # Solo detección de anomalías
python main.py forecast         # Solo forecasting (train → predict → backtest)
python main.py visualize        # Solo visualizaciones HTML
python main.py reports          # Solo reportes (narrativa, ejecutivo, QA, PowerBI)

# Pipeline completo (alias)
python main.py all

# Parámetros opcionales
python main.py pipeline --periods 24 --anomaly-rate 0.05 --verbose
```

| Subcomando | Descripción |
|-----------|-------------|
| `pipeline` / `all` | Ejecuta todo: ETL → anomalías → forecast → reportes |
| `etl` | Genera datos sintéticos, limpia y crea features |
| `anomalies` | Detecta anomalías con 5 métodos + matriz de consenso |
| `forecast` | Entrena Prophet, genera predicciones y backtesting |
| `visualize` | Genera 4 gráficos HTML interactivos |
| `reports` | Smart Narrative, Resumen Ejecutivo, QA Report, Power BI |

### Ayuda

```bash
python main.py --help
```

---

## Dashboard Web

El dashboard web corre con **Dash** (Plotly) en `http://127.0.0.1:8050`:

```bash
python dashboards/app.py
# Abrir http://127.0.0.1:8050 en el navegador
```

### Componentes del Dashboard

| Sección | Descripción |
|---------|-------------|
| **KPIs** | Caja neta actual, promedio 30d, forecast 30d, SMAPE, MAE, min/max |
| **Gráfico 1** | Evolución temporal: real + forecast + bandas de confianza (80% y 95%) |
| **Gráfico 2** | Ingresos/Gastos mensuales (barras apiladas) |
| **Gráfico 3** | Estacionalidad semanal + residuales |
| **Gráfico 4** | Scatter Real vs Predicho + línea de referencia |
| **Gráfico 5** | Distribución de errores (histograma) |
| **Tabla** | Alertas: anomalías del período con severidad y tipo |

### Diseño

- Tema corporativo: azul primario `#1a237e`, cards blancas con sombra
- Tipografía: Inter (Google Fonts)
- Layout responsive: 3 filas de subplots con rangeslider interactivo
- Filtros por rango de fechas

---

## Pipeline de Datos

### 1. Generación de Datos Sintéticos

`src/etl/generator.py`

Genera ~730 filas (24 meses) con:

| Patrón | Comportamiento |
|--------|---------------|
| Estacionalidad semanal | Findes -30%, viernes +5%, lunes +3% |
| Estacionalidad mensual | Última semana +10%, primera semana -5% |
| Estacionalidad trimestral | Q4 +20%, Q1 -10% |
| Tendencia anual | +4% lineal |
| Distribución | Log-normal con ruido gaussiano ~5% |
| Festivos mexicanos | 12 por año (8 oficiales + 4 observados) |
| Anomalías | 6 tipos: pico, caída, quiebre, estacional, drift, cero |

**Parámetros**: `seed=42` (reproducible), `periods=24` meses, `anomaly_rate=0.05`

### 2. Limpieza y Feature Engineering

`src/etl/cleaner.py` + `src/etl/features.py`

- Remueve duplicados temporales
- Normaliza tipos de datos
- Reporte de limpieza con estadísticas
- 28 columnas finales:

| Tipo de Feature | Columnas |
|----------------|----------|
| Originales | fecha, caja_neta, ingresos (3), gastos (2), saldo |
| Lags | 1, 7, 14, 30 días |
| Rolling windows | Media y std de 7, 14, 30 días |
| Temporales | día_semana, mes, trimestre, año, finde, etc. |

### 3. Detección de Anomalías

`src/anomalies/` — 5 métodos de detección + matriz de consenso:

| Método | Archivo | Descripción |
|--------|---------|-------------|
| **Isolation Forest** | `isolation_forest.py` | Basado en aislamiento, random_state=42 |
| **Z-score** | `statistical.py` | Desviaciones > 3σ |
| **IQR** | `statistical.py` | Fuera de [Q1-1.5*IQR, Q3+1.5*IQR] |
| **Ley de Benford** | `benford.py` | Distribución anómala de primer dígito |
| **Temporal** | `temporal.py` | Desviaciones en patrones estacionales |

La **matriz de consenso** (`consensus.py`) pondera los 5 métodos y clasifica la severidad:
- **Crítico**: 5/5 detectores coinciden
- **Alto**: 4/5
- **Medio**: 3/5
- **Bajo**: 2/5

**Resultado**: 276 anomalías detectadas, **82.3% recall** vs anomalías conocidas.

### 4. Forecasting con Prophet

`src/forecasting/` — Todo el pipeline de forecasting:

| Módulo | Descripción |
|--------|-------------|
| `train.py` | Prophet con `seasonality_mode='additive'`, `changepoint_prior_scale=0.05`, estacionalidades semanal y anual, festivos mexicanos |
| `predict.py` | Predicciones a 30/60/90 días con intervalos de confianza 80% y 95% |
| `backtest.py` | Walk-forward: ventana train=90d, test=30d, mínimo 6 iteraciones |
| `visualize.py` | 4 gráficos HTML interactivos: forecast, componentes, residuos, anomalías |

**Métrica principal**: **SMAPE** (Symmetric Mean Absolute Percentage Error). Se eligió SMAPE sobre MAPE porque `caja_neta` cruza por cero, lo que hace que MAPE sea infinito/indefinido.

**Resultados del backtest**:
| Métrica | Valor |
|---------|-------|
| SMAPE global | 163.07% |
| MAE global | $516,556 |
| Iteraciones | 14 |

### 5. Visualizaciones

`src/forecasting/visualize.py` — 4 gráficos HTML interactivos:

1. **Forecast Plot**: Serie temporal con histórico, predicción e intervalos de confianza
2. **Componentes**: Tendencia, estacionalidad semanal y anual (descompuestas por Prophet)
3. **Residuos**: Análisis de residuos vs tiempo
4. **Anomalías**: Anomalías marcadas sobre la serie de forecast

---

## Reportes Generados

| Reporte | Archivo | Descripción |
|---------|---------|-------------|
| **Smart Narrative** | `reports/smart_narrative.md` | Reporte narrativo nivel CFO con insights, tendencias y alertas |
| **Resumen Ejecutivo** | `reports/ejecutivo_resumen.md` | 1 página con KPIs, anomalías críticas y recomendaciones |
| **QA Report** | `reports/qa_report.md` | 12/12 checks — score EXCELENTE (100%) |
| **Power BI CSV** | `reports/forecast_powerbi.csv` | 17 columnas optimizadas para Power BI |
| **Diseño Dashboard** | `docs/dashboard_design.md` | Layout, medidas DAX, instrucciones de carga |
| **Anomalías** | `reports/anomaly_report.csv` | 334 filas con severidad y tipo |
| **Matriz de Consenso** | `reports/consensus_matrix.csv` | Scores de 5 detectores por fecha |
| **Resumen Forecast** | `reports/forecast_summary.txt` | Estadísticas de predicciones |

---

## Tests

48 tests organizados en 7 archivos:

| Archivo | Tests | Cobertura |
|---------|-------|-----------|
| `test_generator.py` | 4 | Generación de datos sintéticos |
| `test_cleaner.py` | 3 | Limpieza y validación de esquema |
| `test_features.py` | 3 | Feature engineering sin leakage |
| `test_anomalies.py` | 4 | Detección y consenso |
| `test_forecast.py` | 8 | Prophet, festivos, predicciones |
| `test_backtest.py` | 14 | Walk-forward, SMAPE, métricas |
| `test_pipeline.py` | 12 | Integración end-to-end + CLI |

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Con cobertura
python -m pytest tests/ --cov=src/ --cov-report=term
```

---

## Métricas de Rendimiento

| Métrica | Valor | Notas |
|---------|-------|-------|
| SMAPE global | 163.07% | Simétrico, acotado [0, 200], funciona con negativos |
| MAE global | $516,556 | Error absoluto medio en USD |
| Anomalías detectadas | 276 | 5 métodos + consenso |
| Recall anomalías | 82.3% | Vs anomalías conocidas sintéticas |
| Tests | 48/48 ✅ | Todos pasando |
| QA Score | 12/12 — EXCELENTE | 100% de entregables verificados |

> **Nota sobre SMAPE vs MAPE**: MAPE no es fiable cuando `caja_neta` cruza por cero (valores negativos o cero en el denominador). SMAPE es simétrico, acotado y maneja correctamente ceros y negativos.

---

## Estructura de Archivos

```
Forecasting_de_caja/
│
├── main.py                          ← Orquestador CLI (7 subcomandos)
├── requirements.txt                 ← Dependencias Python
├── .gitignore                       ← Cache y logs ignorados
│
├── src/                             ← Código fuente Python
│   ├── etl/                         ← Pipeline ETL
│   │   ├── generator.py             ← Datos sintéticos (332 líneas)
│   │   ├── cleaner.py               ← Limpieza y validación (261 líneas)
│   │   └── features.py              ← Feature engineering (209 líneas)
│   │
│   ├── anomalies/                   ← Detección de anomalías
│   │   ├── isolation_forest.py      ← Isolation Forest (162 líneas)
│   │   ├── statistical.py           ← Z-score + IQR (127 líneas)
│   │   ├── benford.py               ← Ley de Benford (238 líneas)
│   │   ├── temporal.py              ← Anomalías temporales (219 líneas)
│   │   ├── consensus.py             ← Matriz de consenso (194 líneas)
│   │   └── report.py                ← Reportes de anomalías (281 líneas)
│   │
│   ├── forecasting/                 ← Pipeline de forecasting
│   │   ├── train.py                 ← Prophet (203 líneas)
│   │   ├── predict.py               ← Predicciones (230 líneas)
│   │   ├── backtest.py              ← Walk-forward (330 líneas)
│   │   └── visualize.py             ← Gráficos Plotly (486 líneas)
│   │
│   ├── dashboards/                  ← Exportación Power BI
│   │   └── export_powerbi.py        ← CSV + diseño (523 líneas)
│   │
│   └── reports/                     ← Generadores de reportes
│       ├── smart_narrative.py       ← Narrativa CFO (386 líneas)
│       ├── executive_summary.py     ← Resumen ejecutivo (394 líneas)
│       └── qa_report.py             ← QA checklist (527 líneas)
│
├── tests/                           ← Tests (48 tests, 7 archivos)
│
├── dashboards/                      ← Dashboard Dash web
│   ├── app.py                       ← 1,211 líneas, 6 gráficos
│   └── assets/
│       └── style.css                ← 249 líneas, tema corporativo
│
├── data/
│   ├── raw/dataset_raw.csv          ← 729 filas generadas
│   └── curated/
│       ├── dataset_clean.csv        ← 729 filas limpias
│       └── dataset_features.csv     ← 334 filas con features
│
├── reports/                         ← Reportes generados
│   ├── anomaly_report.csv
│   ├── consensus_matrix.csv
│   ├── forecast_results.csv
│   ├── forecast_powerbi.csv
│   ├── forecast_summary.txt
│   ├── *.html                       ← 4 gráficos interactivos
│   ├── smart_narrative.md
│   ├── ejecutivo_resumen.md
│   └── qa_report.md
│
├── docs/
│   ├── guia_proyecto.md             ← Guía completa del proyecto
│   └── dashboard_design.md          ← Diseño Power BI + DAX
│
├── notebooks/
│   └── forecasting_caja.ipynb       ← Pipeline completo ejecutable
│
├── models/
│   └── prophet_model.pkl            ← Modelo entrenado (45KB)
│
└── scripts/
    └── run_forecast_pipeline.py     ← Script de ejecución directa
```

---

## Licencia

Este proyecto se distribuye bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

**Autor**: Equipo de Data Science  
**Seed**: 42 (reproducibilidad garantizada)  
**Última actualización**: Mayo 2026
