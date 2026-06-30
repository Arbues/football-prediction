# Football Match Result Prediction

Pipeline completo de Minería de Datos para predecir el resultado de partidos de fútbol europeo (2000–2025) usando técnicas de Machine Learning.

**Curso:** Minería de Datos (CC442) — UNI  
**Docente:** Dr. Marcos Antonio Alania Vicente  
**Ciclo:** 2026-I

## Dataset

**Club Football Match Data (2000–2025)** — [Kaggle](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025)

- ~230K partidos, 48 columnas
- Cubre múltiples ligas internacionales
- Incluye Elo ratings, estadísticas del partido, cuotas de apuestas

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

O directamente desde los scripts en `src/`:

```bash
python src/data_loader.py
python src/preprocessing.py
python src/models.py
python src/evaluation.py
```

## Resultados Clave

| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| TBD    | —        | —        | —       |

*(resultados preliminares — completar tras experimentación)*

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
- [Tu nombre]

**Licencia:** MIT
