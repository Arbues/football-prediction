# `results/` — Reportes y métricas

Salidas de texto/CSV de cada paso del pipeline. Toda cifra del artículo IEEE se regenera
desde aquí (trazabilidad completa entre repo y paper).

## Raíz — EDA y preprocesamiento

Reportes numerados que acompañan a cada script de `src/eda/` y `src/preprocessing/`:

- `00_overview` … `11_draw_analysis` → hallazgos del EDA (faltantes, balance, correlación,
  VIF, outliers multivariados, anatomía del empate).
- `12_clean` … `17_pca` → trazas del preprocesamiento (limpieza, unión Elo, *features*,
  partición, transformación, PCA).

## `modeling/` — Resultados del modelado

| Archivo | Qué contiene |
|---|---|
| `00_dummy.txt` … `10_stacking.txt` | Métricas por modelo (Precision/Recall/F1 por clase, Kappa, AUC, log-loss). |
| `11_test_comparison.txt` | **Tabla maestra**: los 13 modelos ordenados por F1 macro en test. |
| `11_test_metrics.csv` | Las mismas métricas en formato CSV (fuente de las tablas del paper). |
| `11_wilcoxon.txt` | Prueba de significancia (Wilcoxon por bloque mensual) del modelo propuesto vs. líneas base. |
| `12_shap.txt` | Importancia global por valores SHAP. |
| `grid_search_results.csv` | Resultados de la búsqueda de hiperparámetros. |
| `predict_demo.txt`, `simulate_bracket.txt` | Salidas de la demo de predicción del Mundial 2026. |

**Resultado principal:** el ensamble por apilamiento (Stacking) alcanza F1 macro 0,419 en
test y supera a las líneas base con significancia estadística (*p* < 0,001).
