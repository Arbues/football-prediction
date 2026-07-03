# `notebooks/` — Cuadernos del proyecto (entregable interactivo)

Tres cuadernos Jupyter que reproducen el pipeline completo de principio a fin. Son la
**evidencia ejecutable** del proyecto: alternan explicación teórica (Markdown), código y
interpretación de resultados.

## Orden de ejecución (obligatorio)

Ejecutar en secuencia; cada uno consume la salida del anterior:

| # | Cuaderno | Qué hace | Genera |
|---|---|---|---|
| 1 | `1.0-eda.ipynb` | Análisis exploratorio: distribuciones, correlación, VIF, outliers, anatomía del empate. | Figuras en `figures/`, reportes en `results/`. |
| 2 | `2.0-preprocessing.ipynb` | Limpieza anti-fuga, reconstrucción de Elo, ingeniería de *features*, partición temporal, estandarización y PCA. | Matrices en `data/processed/`. |
| 3 | `3.0-modeling-evaluation.ipynb` | Entrenamiento y ajuste de 13 modelos, evaluación (ROC, matriz de confusión), significancia (Wilcoxon) e interpretabilidad (SHAP). | Métricas en `results/modeling/`, modelos en `models/`. |

## Cómo ejecutarlos

```bash
# desde la raíz del repo, con el entorno instalado (ver README principal)
jupyter notebook            # o abrir en VS Code / JupyterLab
```

- **Datos:** los cuadernos `1.0` y `2.0` descargan el dataset automáticamente con
  `kagglehub` (con *fallback* a la API de Kaggle). No hace falta cargar archivos a mano.
- **Semillas:** todos fijan `random_state = 42` en cada componente estocástico.
- **Reproducibilidad:** ver la sección *Reproducibilidad* del README principal.

> El cuaderno `3.0` requiere que `2.0` se haya ejecutado antes (lee `data/processed/*.parquet`).
