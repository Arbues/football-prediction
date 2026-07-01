# Bitácora de avances — Football Match Prediction

> Orden cronológico **inverso** (lo más reciente arriba). Plantilla en `README.md`.

---

## [2026-07-01 ~01:35] — Arbués — Nota de METODOLOGÍA del EDA (para paper + expo)

**Pregunta:** ¿de qué paper tomamos la metodología del EDA? ¿En qué nos basamos?

**Respuesta honesta (verificada contra `referencia-papers.md` y `referencia-kernels.md`):**

1. **Ningún paper aporta una "metodología de EDA" para copiar.** Los 10 papers son de *modelado* (features, algoritmos, validación), no de análisis exploratorio. No hay un EDA de referencia citable como plantilla directa.

2. **Marco metodológico que SÍ adoptamos y podemos citar: CRISP-DM, fase "Data Understanding".** Nuestro EDA corresponde exactamente a esa fase. Es citable vía **Bunker & Thabtah (2019)**, que proponen **SRP-CRISP-DM** (Sport Result Prediction CRISP-DM), una especialización de CRISP-DM para predicción deportiva. → Es nuestra ancla metodológica principal para el EDA y para el pipeline completo. **Citar en Metodología y en la expo.**

3. **Plantilla de EDA que extendemos (kernels de Kaggle, no papers):** los kernels #1 (`club-football-match-data`) y #2 (`football-match-data-analysis`) de `referencia-kernels.md` son EDA-puro y comparten nuestro dataset. De ellos tomamos la estructura básica: `df.info()`, mapa de nulos, boxplots, histogramas/KDE. **PERO los superamos**: añadimos (a) taxonomía anti-leakage de columnas, (b) outliers multivariados (Mahalanobis), (c) VIF/multicolinealidad, (d) análisis condicional del empate, (e) correlación Pearson+Spearman comparadas. Ninguno de los kernels hizo nada de esto → es parte de nuestra contribución/rigor y hay que decirlo así en el paper.

4. **Metodologías de PAPERS que sí adoptamos, pero para Fase 2/3/4 (no EDA):**
   - **Validación temporal (split por fecha, anti-CV estándar):** Bunker & Thabtah (2019), Berrar et al. (2019), Bunker/Yeung/Fujii (2024). ← decisión ya tomada en el EDA (por la deriva temporal + COVID detectada).
   - **Desbalance + F1 por clase + balanced sampling:** Choi et al. (2023), Atta Mills et al. (2024). ← justifica reportar F1 del empate aparte.
   - **Feature engineering recency + rating (setup casi idéntico, 216k):** Berrar et al. (2019).
   - **Prueba de significancia (Wilcoxon):** Stübinger et al. (2020).
   - **Interpretabilidad SHAP:** Ren & Susnjak (2022).
   - **CatBoost/GBT + pi-ratings como punteros:** Ren & Susnjak (2022), Bunker/Yeung/Fujii (2024).

**Frase lista para el paper/expo:** "El análisis exploratorio sigue la fase de *Data Understanding* del marco CRISP-DM, en su especialización SRP-CRISP-DM para predicción deportiva (Bunker & Thabtah, 2019), extendiendo los análisis exploratorios previos sobre este dataset con una taxonomía anti-fuga de datos, detección de atípicos multivariados (Mahalanobis), diagnóstico de multicolinealidad (VIF) y un estudio condicional de la clase empate."

**Para Luis (paper):** añadir Bunker & Thabtah (2019) a la sección de Metodología (no solo a Trabajos Relacionados). Verificar que CRISP-DM/SRP-CRISP-DM quede citado.

---

## [2026-07-01 ~01:10] — Arbués — EDA ampliado + notebook `1.0-eda.ipynb`

**Qué hice:**
- Amplié el EDA con 3 análisis nuevos (investigación previa con evidencia): `09_multivariate_outliers.py` (Mahalanobis, D_M² vs χ²), `10_multicollinearity_vif.py` (VIF = diag(R⁻¹)), `11_draw_analysis.py` (anatomía del empate). 12 scripts en total, Run All 0 errores.
- Construí el **notebook `notebooks/1.0-eda.ipynb`** (41 celdas, estilo académico de PCs anteriores: Fundamentación → Código → Interpretación). Autocontenido y portable a Colab (carga: local `data/raw` → kagglehub → descarga directa ZIP). **Ejecutado con nbconvert de inicio a fin: 16 celdas de código, 0 errores, 8 figuras inline.**

**Hallazgos nuevos:**
- **Mahalanobis**: 78 outliers MV (0.06%), 28 invisibles al IQR univariado (anomalías por combinación, p.ej. Elo alto + forma 0). Cola algo pesada vs χ²(df=6) → leve no-normalidad MV.
- **VIF**: máx 8.85 (OddAway), ninguno grave (>10); 4 moderados (cuotas + ambos Elo). Nota: diffs derivados dan VIF→∞ (usar diffs XOR componentes en modelos lineales; árboles inmunes).
- **Empate**: tasa máx 29.8% (partidos parejos |elo_diff|<25), cae a 18.3% en desnivel >200. **D nunca es mayoritario en ninguna región** → recall con techo estructural bajo (explica la literatura). Composición H/D/A por elo_diff monótona (A 61%→7%, H 17%→79%).
- **(De investigación previa, no incluido en scripts pero anotado)** Deriva temporal + COVID: home-win cae de ~47% (2000s) a 41.7% en 2020 / 42.4% en 2021. Cobertura Elo por liga: E3=0%, E2=12% vs top-5=100% → el nulo de Elo es por liga, no por año. Join nombres Matches↔EloRatings solo 57% exacto. (Candidatos para Fase 2 o ampliación futura.)

**Archivos nuevos:** `src/eda/09,10,11_*.py`, `results/09,10,11_*.txt`, `figures/{mahalanobis_outliers,vif_barplot,draw_analysis}.png`, `notebooks/1.0-eda.ipynb`. 15 figuras y 12 reportes en total.

**Para el siguiente (Fase 2):** el notebook cierra en la sección 11 (contrato de entrada) y 12 (puente). Split temporal por `MatchDate`, imputación Elo por liga/temporada o join temporal, limpieza de centinelas en cuotas.

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
