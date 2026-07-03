# `src/` — Código fuente modular

Código de producción organizado **por fase** (no por tipo técnico). Cada script es
autocontenido y ejecutable de forma independiente; los de una misma fase comparten un
núcleo común `_common.py` (semillas, carga de datos, esquema de validación temporal,
métricas y utilidades de figuras).

## Estructura

| Subcarpeta | Contenido | Salidas |
|---|---|---|
| `eda/` | 12 scripts numerados (`00_overview` → `11_draw_analysis`) + `_common.py`. Perfilado, faltantes, balance, correlación, outliers, VIF, análisis del empate. | `results/*.txt`, `figures/*.png` |
| `preprocessing/` | 6 scripts (`01_clean`, `02_elo_join`, `03_feature_engineering`, `04_temporal_split`, `05_pipeline_transform`, `06_pca`) + `_common.py`. | `data/processed/*.parquet` |
| `modeling/` | Baselines (`00`), 8 modelos (`01`–`08`), desbalance (`09`), ensambles (`10`), evaluación (`11`), SHAP (`12`) + `_common.py`. Incluye `predict_demo.py` y `simulate_bracket.py` (demo Mundial 2026). | `results/modeling/*`, `models/*.pkl`, `figures/modeling/*.png` |

## Cómo ejecutar

```bash
# Fase 1 — EDA
for s in src/eda/*.py; do python "$s"; done

# Fase 2 — Preprocesamiento (genera data/processed/)
for s in src/preprocessing/*.py; do python "$s"; done

# Fase 3 — Modelado y evaluación (en orden numérico)
for s in src/modeling/0*.py src/modeling/1*.py; do python "$s"; done
```

Los scripts numerados deben correr en orden dentro de cada fase. El entrenamiento con GPU
(RTX 3060) usa `device='cuda'` en XGBoost y `task_type='GPU'` en CatBoost; también corre
en CPU. Toda aleatoriedad está fijada con `random_state = 42`.
