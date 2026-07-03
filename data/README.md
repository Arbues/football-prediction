# `data/` — Datos

Almacenamiento de datos del proyecto. Se separa en datos **crudos** (solo lectura) y
datos **procesados** (generados por el pipeline).

## `raw/` — Datos originales (solo lectura)

Dataset público **Club Football Match Data (2000–2025)**
([Kaggle](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025)):

| Archivo | Qué es |
|---|---|
| `Matches.csv` | ~230K partidos de múltiples ligas: fechas, equipos, marcador, estadísticas y cuotas. |
| `EloRatings.csv` | Serie histórica de *ratings* Elo por equipo (para la unión temporal anti-fuga). |

No se modifican. Los cuadernos `1.0`/`2.0` los descargan automáticamente si no están presentes.

## `processed/` — Datos transformados (generados por `src/preprocessing/`)

Artefactos intermedios (prefijo `_`) y matrices finales para el modelo:

| Archivo | Qué es |
|---|---|
| `_matches_clean.parquet` | Partidos tras limpieza anti-fuga. |
| `_matches_elo.parquet` | Con Elo reconstruido por `merge-asof` (fecha ≤ partido). |
| `_matches_features.parquet` | Con las 62 *features* de ingeniería. |
| `_matches_split.parquet` | Con la etiqueta de partición temporal. |
| `X_train / X_val / X_test .parquet` | Matrices de diseño por partición (train ≤ 2021 · val 2022–2023 · test 2024–2025). |
| `y_train / y_val / y_test .parquet` | Etiquetas `FTResult` ∈ {H, D, A}. |
| `X_*_pca.parquet` | Versiones reducidas por PCA (95 % de varianza, 13 componentes). |
| `feature_names.json` | Nombres de las columnas de la matriz final. |
| `index_ref.parquet` | Índice de referencia para trazar cada fila a su partido original. |

> Regla de la guía: los datos > 50 MB no deben subirse a Git. Si el `.gitignore` excluye
> alguno, se regenera ejecutando `src/preprocessing/` en orden.
