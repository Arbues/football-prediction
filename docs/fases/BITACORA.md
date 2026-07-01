# Bitácora de avances — Football Match Prediction

> Orden cronológico **inverso** (lo más reciente arriba). Plantilla en `README.md`.

---

## [2026-07-01 ~00:35] — Arbués — Fase 1 (EDA) en scripts `src/eda/`

**Qué hice:**
- Descargué el dataset a `data/raw/` (`Matches.csv`, `EloRatings.csv`) vía el endpoint público de Kaggle con `curl` (kagglehub quedó con install roto: falta `kagglesdk.competitions.legacy`; el endpoint directo `api/v1/datasets/download/...` funciona sin credenciales).
- Construí el EDA como **9 scripts `.py` numerados y autónomos** en `src/eda/` (no notebook aún, por decisión del equipo). Cada uno vuelca un `.txt` legible en `results/` y las figuras en `figures/`. `_common.py` centraliza rutas, carga y la taxonomía anti-leakage. **Run All secuencial: 0 errores, stderr limpio.**

**Archivos tocados (nuevos):**
- `src/eda/_common.py` + `00_overview` … `08_drop_decisions` (9 scripts).
- `results/00..08_*.txt` (9 reportes) y `figures/*.png` (12 figuras: `class_balance`, `correlation_matrix`, `correlation_spearman`, `missingness_bar`, `dist_{goals,elo,elo_diff,form}`, `boxplots_{outliers,odds}`, `elo_by_result`, `form_by_result`).

**Hallazgos clave:**
- **N/D confirmado: 230 557 × 48** (era la cifra de PLANNING; los kernels decían 226 755×42 = versión vieja del CSV). 38 ligas, 1214 equipos, 2000-07-28 → 2025-06-01.
- **Balance FTResult**: H 44.6% / A 28.9% / D 26.5%. IR=1.68 (**desbalance moderado**, no severo). Baseline mayoritario a superar = 0.446. D es la clase difícil.
- **CORRECCIÓN a `referencia-kernels.md`**: la nota "Elo solo desde ~2006" es **falsa** para esta versión. El nulo de Elo (~38.6%) es ~10-25% en 2000-2011 y **sube a 42-49% en 2012-2024** (ligas menores sin cobertura Elo). Filtrar por año NO resuelve el nulo.
- **Señal pre-partido confirmada**: `elo_diff` separa H/D/A monótonamente (medias H=+44, D=−10, A=−61); `form5_diff` igual (H=+0.64, D=−0.41, A=−1.32). D queda en el medio → difícil.
- **Cuotas ≈ "leak del sabio"**: Spearman `elo_diff~Odds ≈ 0.89`, `OddHome~OddAway = −0.98`. Confirma modelar CON y SIN odds.
- **Elo simétrico** (elo_diff media≈0): no hay sesgo de emparejamiento por sede; la ventaja de localía es efecto de sede, no de Elo.
- **Calidad de datos en cuotas**: centinelas/imposibles (`OddHome==0` en 7 filas, `HandiSize==−99.9`, `Max*` hasta 301) → limpiar/winsorizar. Elo/Form sin valores imposibles.
- **Decisión de columnas**: 24 conservar (6 pre-seguras + 13 odds condicionales + contexto + target) / **24 descartar** (17 leakage post-partido + 6 `C_*` >51% nulo + `MatchTime` 57% nulo). 3 filas con FTResult nulo se descartan.

**Para el siguiente (Fase 2 — Preprocessing):**
- Insumo listo en `results/08_drop_decisions.txt` (tabla keep/drop + estrategia de nulos).
- Elo: decidir entre (a) filtrar a partidos con Elo, (b) imputar mediana liga/temporada, (c) join temporal con `EloRatings.csv` (`date <= MatchDate`). Form: imputar 0. Odds: mediana o descartar sub-bloques >35% nulo.
- Split **temporal** por `MatchDate` (no aleatorio), reportar F1 por clase.

**Bloqueos / dudas abiertas:**
- Notebook `1.0-eda.ipynb` pendiente (se arma cuando lo pida el usuario, portando estos scripts).

---

## [2026-06-30 ~23:40] — Sergio — Fase 0 cerrada + esqueleto Fase 3

**Qué hice:**
- Generé los dos docs de referencia (papers + kernels) — ver detalle abajo.
- Construí el esqueleto plug-and-play del notebook de modelado `notebooks/3.0-modeling-evaluation.ipynb` (17 celdas). Verificado con `Run All` headless: corre limpio sin errores (todas las celdas dependientes de datos están protegidas con `DATA_READY`).

**Archivos tocados:**
- `notebooks/3.0-modeling-evaluation.ipynb` — nuevo. Seeds, detección de libs, carga flexible de `data/processed`, baseline Dummy, registro de 6 modelos (LogReg/SVM/RF/XGB/LGBM/CatBoost), GridSearch/RandomizedSearch con CV estratificada 5-fold, presupuesto de 8 min/modelo con submuestreo automático, GPU para boosting con fallback, guardado a `models/` y `results/grid_search_results.csv`.
- `requirements.txt` — agregado `scipy` y `optuna`.
- `docs/fases/referencia-papers.md`, `docs/fases/referencia-kernels.md` — nuevos.

**Decisiones tomadas (y por qué):**
- Dataset: arrancamos con el COMPLETO (~226k) usando GPU; si un modelo tarda >8 min en un fit de prueba, el notebook baja a submuestra estratificada del 30% automáticamente (decisión de Sergio).
- Métrica de optimización: `f1_weighted` (multiclase desbalanceado). Se reportarán F1w y F1m.
- `class_weight="balanced"` en LogReg/SVM/RF/LGBM para compensar desbalance sin remuestrear.
- Imports condicionales: si falta lightgbm/catboost/optuna, se omite ese modelo en vez de romper el Run All.

**Para el siguiente que trabaje acá:**
- **Arbués (Fase 2)**: el notebook de modelado espera `data/processed/X_train.{parquet|npy}` + `y_train...` (y val/test). Preferible **parquet** (conserva nombres de columnas → SHAP y feature importance en Fase 4). Si usas otro formato, solo hay que ajustar la celda `load_split`.
- **Luis (Fase 4)**: tu mitad continúa DEBAJO de la última celda del mismo notebook. Ya tendrás `models/*.pkl` y `results/grid_search_results.csv`. Tu insumo teórico está en `referencia-papers.md` (tabla de brechas lista para el .tex).
- Instalar libs faltantes en la máquina de entrenamiento: `pip install lightgbm catboost optuna`.

**Bloqueos / dudas abiertas:**
- Confirmar en Fase 1 el N/D exacto: kernels dicen 226,755×42, PLANNING dice 230,557×48. Probable diferencia de versión del CSV.
- Confirmar qué representan `Form3*/Form5*` (¿pre-partido? valen 0–13, parecen puntos en últimos N).

---

## [2026-06-30 ~23:00] — Sergio — Fase 0: research + andamiaje de bitácora

**Qué hice:**
- Verifiqué el contenido real de `pc5/EXPO/`: son DOS recursos distintos, no un mapeo paper↔notebook.
  - `EXPO/referecnias/` → 10 papers académicos (PDF) → alimentan Trabajos Relacionados (Luis).
  - `EXPO/kagles anteriores/` → 5 kernels de Kaggle (mismo dataset que usamos) → alimentan feature engineering / modelado (Sergio).
- Monté el sistema de bitácora en `docs/fases/`.
- Disparé la extracción estructurada de los 15 documentos para generar los dos docs de referencia.

**Archivos tocados:**
- `docs/fases/README.md` — convención de la bitácora (nuevo)
- `docs/fases/BITACORA.md` — este log (nuevo)
- `docs/fases/referencia-papers.md` — resumen de los 10 papers (en generación)
- `docs/fases/referencia-kernels.md` — resumen de los 5 kernels (en generación)

**Decisiones tomadas (y por qué):**
- `docs/paper/paper.tex` sigue siendo la plantilla IEEE cruda (sin tocar). Lo arranca Luis en Fase 0.5.

**Para el siguiente que trabaje acá:**
- **Luis**: `referencia-papers.md` es tu insumo directo para la tabla comparativa de brechas.
- **Arbués**: `referencia-kernels.md` lista features que ya probaron otros — úsalo en Fase 1/2.
- Ojo con **data leakage**: los kernels que usan `HomeShots`/`AwayShots`/cards del propio partido están mal para nuestro caso (esas stats no existen ANTES del partido). Ver la sección de leakage en `referencia-kernels.md`.

**Bloqueos / dudas abiertas:**
- Falta decidir en Fase 1: ¿ventana completa 2000–2025 (~230k) o últimas N temporadas por costo de entrenamiento?
