# Football Match Result Prediction

Pipeline completo de Minería de Datos para predecir el resultado de partidos de fútbol europeo (2000–2025) usando técnicas de Machine Learning.

**Curso:** Minería de Datos (CC442) — UNI
**Docente:** Dr. Marcos Antonio Alania Vicente
**Ciclo:** 2026-I

## Dataset

**Club Football Match Data (2000–2025)** — [Kaggle](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025)

- ~230K partidos, 48 columnas; múltiples ligas internacionales
- Incluye Elo ratings, estadísticas del partido y cuotas de apuestas
- **Objetivo:** `FTResult` ∈ {H, D, A} (victoria local / empate / visitante), multiclase
- Se modela con información **estrictamente pre-partido** y **sin cuotas de mercado**
  (versión honesta): 62 *features* tras el *feature engineering* (Elo, forma, goles
  recientes, h2h, descanso, rachas, banderas de liga)
- Partición **temporal**: train ≤ 2021 · val 2022–2023 · test 2024–2025

## Requisitos

- Python ≥ 3.10
- Dependencias en `requirements.txt`

## Instalación

```bash
git clone <repo-url>
cd football-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Los notebooks en `notebooks/` deben ejecutarse en orden:

1. `1.0-eda.ipynb` — Análisis exploratorio
2. `2.0-preprocessing.ipynb` — Limpieza y transformación
3. `3.0-modeling-evaluation.ipynb` — Modelado y evaluación

O directamente desde los scripts modulares en `src/` (usar el intérprete del entorno):

```bash
# Fase 1 — EDA (genera reportes en results/ y figuras en figures/)
for s in src/eda/*.py; do python "$s"; done

# Fase 2 — Preprocesamiento (genera data/processed/)
for s in src/preprocessing/*.py; do python "$s"; done

# Fase 3 — Modelado y evaluación (en orden; vuelca a results/modeling/ y figures/modeling/)
for s in src/modeling/0*.py src/modeling/1*.py; do python "$s"; done
```

Cada script de `src/modeling/` es autocontenido y comparte `src/modeling/_common.py`
(semillas, carga, `TimeSeriesSplit`, métricas y figuras). El entrenamiento con GPU
(RTX 3060) usa `device='cuda'` en XGBoost y `task_type='GPU'` en CatBoost.

## Resultados Clave

Evaluación en **test (2024–2025)**, optimizando `f1_macro` (el empate pesa igual). El
modelo propuesto es el **Stacking** (meta-LogReg sobre LogReg + RF + XGBoost).

| Modelo | f1_macro | Accuracy | Kappa | ROC-AUC (OvR) | recall (empate) |
|--------|:--------:|:--------:|:-----:|:-------------:|:---------------:|
| **Stacking (propuesto)** | **0.419** | 0.438 | 0.146 | 0.610 | 0.264 |
| LightGBM | 0.419 | 0.444 | 0.149 | 0.613 | 0.235 |
| Random Forest | 0.417 | 0.446 | 0.149 | 0.614 | 0.213 |
| XGBoost | 0.416 | 0.439 | 0.143 | 0.609 | 0.245 |
| CatBoost | 0.415 | 0.442 | 0.146 | 0.615 | 0.221 |
| SVM-RBF | 0.413 | 0.424 | 0.134 | 0.606 | **0.314** |
| Regresión Logística | 0.412 | 0.439 | 0.142 | 0.607 | 0.214 |
| Naive Bayes (base) | 0.392 | 0.412 | 0.103 | 0.576 | 0.235 |
| Dummy (piso) | 0.202 | 0.434 | 0.000 | 0.500 | 0.000 |

El Stacking **supera a las líneas base con significancia estadística** (prueba de
Wilcoxon por bloque mensual, *p* < 0.05 frente a Dummy y a Naive Bayes). El empate (D)
es la clase difícil —sin región propia en el espacio— y su *recall* tiene un techo
estructural; el SVM-RBF, con frontera no lineal, es el que más se le acerca. La
variable dominante es `elo_diff`, confirmada por importancia por permutación y SHAP.

## Estructura del Repositorio

```
.
├── data/               ← Datos crudos y procesados
│   ├── raw/
│   └── processed/
├── notebooks/          ← Jupyter/Colab notebooks
├── src/                ← Código fuente modular
├── models/             ← Modelos serializados (.pkl)
├── figures/            ← Gráficos para el artículo
├── results/            ← Métricas y reportes CSV/JSON
├── docs/               ← Artículo IEEE + Presentación Beamer
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Créditos

- Sergio Pezo
- Luis Trujillo
- Arbues Perez

**Licencia:** MIT
