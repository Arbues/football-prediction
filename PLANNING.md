# Plan de Acción — Proyecto Final Minería de Datos

> **Predicción del Resultado de Partidos de Fútbol mediante Machine Learning**
> Club Football Match Data (2000–2025)
> Sustentación: **Miércoles 1 Julio 2026 — 8:00 AM**

---

## Integrantes

| Integrante | Rol principal | También apoya |
|---|---|---|
| **Arbués Pérez** | Fase 1 (EDA), Fase 2 (Preprocessing) | Fase 6 (Beamer slides intro), Fase 7 |
| **Sergio Pezo** | Fase 0 (kernels Kaggle), Fase 3 (Modeling) | Fase 6 (Beamer slides modelos), Fase 7 |
| **Luis Trujillo** | Fase 0 (papers), Fase 0.5 + 5 (Artículo IEEE), Fase 4 (Evaluación) | Fase 6 (Beamer conclusiones), Fase 7 |

---

## Diagrama de Dependencias

```
Fase 0 (Research — todos en paralelo)
   │
   ├──> Fase 1 (EDA — Arbués)
   │         │
   │         └──> Fase 2 (Preprocessing — Arbués)
   │                  │
   │                  └──> Fase 3 (Modeling — Sergio)
   │                           │
   │                           └──> Fase 4 (Evaluation — Luis)
   │                                    │
   │                                    ├──> Fase 5 (Artículo IEEE — Luis + todos)
   │                                    │
   │                                    └──> Fase 6 (Beamer — Sergio + Arbués)
   │                                             │
   │                                             └──> Fase 7 (Ensayo — todos)
   │
   └──> Fase 0.5 (Draft Paper — Luis, en paralelo con Fase 1 y 2)
```

---

## Fase 0: Research (18:00–19:30)

**Objetivo**: No empezar a codificar ciegamente. Investigar enfoques existentes para el mismo dataset.

### Sergio — Kernels de Kaggle
- Buscar kernels en Kaggle que usen `club-football-match-data-2000-2025`
- Extraer:
  - Features engineering que más usan
  - Modelos que mejor rinden
  - Métricas que reportan
  - Preprocesamiento que aplican
- Resumir en un doc compartido los hallazgos clave

### Luis — Papers relacionados
- Buscar 5–10 artículos en Scopus/IEEE Xplore/ACM sobre predicción de resultados de fútbol con ML
- Armar tabla comparativa: autor, dataset, modelo, métricas, hallazgos
- Esto alimenta la sección "Trabajos Relacionados" del artículo IEEE

### Arbués — Dataset deep dive
- Ejecutar el código de PC05.txt completamente
- Explorar a fondo las 48 columnas de `matches` y 4 de `elo`
- Identificar: nulos por columna, tipos, rangos, valores únicos en categóricas
- Decisión sobre target: FTResult (H/D/A) multiclase
- Decisión sobre columnas a descartar desde el inicio (e.g., columnas de apuestas si no se usan, o columnas con >50% nulos)

**Output de Fase 0**: Brief técnico de 1 página con enfoque consensuado.

---

## Fase 1: EDA (19:30–21:30)

**Responsable**: Arbués
**Notebook**: `notebooks/1.0-eda.ipynb`

### Actividades
1. Estadística descriptiva (mean, median, std, percentiles) para numéricas
2. Análisis de valores nulos por columna (tabla con %)
3. Distribuciones:
   - Goles local/visitante
   - Elo ratings
   - Diferencia de Elo
   - Forma reciente (Form3Home, Form5Home, etc.)
4. Balance de clases en FTResult (H/D/A)
   - Si desbalance severo → planear SMOTE/class_weight
5. Matriz de correlación (Pearson + Spearman) — heatmap
6. Outliers — boxplots + IQR
7. Análisis de columnas candidatas a descartar:
   - Columnas con >50% nulos
   - Columnas redundantes (e.g., FTHome/FTAway vs FTResult)
   - Columnas con data leakage (estadísticas del partido como HomeShots, AwayShots — solo disponibles post-partido)
8. Visualizaciones para el artículo (guardar en `figures/`):
   - Heatmap de correlación (`figures/correlation_matrix.png`)
   - Distribución de clases (`figures/class_balance.png`)
   - Pairplot de variables top (`figures/pairplot.png`)
   - Boxplots de Elo por resultado (`figures/elo_by_result.png`)

**Output**: Notebook EDA completo + figuras en `figures/`.

---

## Fase 0.5: Draft del Artículo IEEE (en paralelo con Fase 1 y 2)

**Responsable**: Luis
**Archivo**: `docs/paper/main.tex`

### Secciones que NO dependen de resultados (pueden escribirse ya)
- **Portada**: Título, autores, afiliación, curso, fecha
- **Abstract** (borrador): 150–250 palabras, estructura: contexto → brecha → método → resultados → conclusión
- **Keywords**: 4–6 palabras ordenadas alfabéticamente
- **Introducción**: Contexto, motivación, problema formal, contribuciones listadas
- **Trabajos Relacionados**: Tabla comparativa de 10+ papers con brecha identificada
- **Descripción del Dataset**: Origen, tamaño (N=230557, D=48), diccionario de variables

### Secciones que esperan resultados (template + placeholders)
- Preprocesamiento: dejar estructura, llenar después
- Metodología Propuesta: dejar estructura con ecuaciones, llenar detalles después
- Resultados: tabla vacía, gráficos placeholder
- Discusión: estructura
- Conclusiones: estructura

### Configuración del template
- Clase IEEEtran de doble columna
- Paquetes: graphicx, amsmath, booktabs, algorithm, hyperref
- Compilar y verificar que genera PDF sin errores

---

## Fase 2: Preprocessing + Feature Engineering (21:30–23:30)

**Responsable**: Arbués
**Notebook**: `notebooks/2.0-preprocessing.ipynb`

### Actividades
1. **Manejo de nulos**:
   - Columnas con >50% nulos → descartar (MatchTime, estadísticas del partido si aplica)
   - Columnas con <50% nulos → imputación con mediana (o KNNImputer si tiempo)
   - Documentar justificación de cada decisión
2. **Feature Engineering**:
   - `elo_diff` = HomeElo - AwayElo
   - `form_diff` = Form5Home - Form5Away
   - `avg_goals_scored_last5` (de últimas 5 fechas por equipo)
   - `avg_goals_conceded_last5`
   - `win_streak_home`, `win_streak_away`
   - `days_rest` (diferencia entre MatchDate y último partido de cada equipo)
   - `is_top_league` (binaria: ligas top 5 europeas)
   - `h2h_avg_goals` (promedio de goles en enfrentamientos previos)
   - Revisar kernels de Kaggle (Fase 0) para features adicionales
3. **Pérdida de datos (data leakage check)**:
   - No usar estadísticas post-partido (HomeShots, AwayShots, etc.)
   - Solo usar información disponible ANTES del partido
4. **División**: Train/Val/Test estratificada (70/15/15)
5. **Pipeline de transformaciones** con `ColumnTransformer`:
   - Numéricas: StandardScaler (fit en train, transform en val/test)
   - Categóricas: OneHotEncoder o TargetEncoding
6. **PCA** (opcional): scree plot, varianza acumulada
7. **Guardar datos procesados**: `data/processed/X_train.npy`, etc. (o parquet)

**Output**: Notebook preprocessing completo + datos limpios en `data/processed/`.

---

## Fase 3: Modeling (23:30–02:30)

**Responsable**: Sergio
**Notebook**: `notebooks/3.0-modeling-evaluation.ipynb` (primera mitad)

### Modelos a implementar

| Modelo | Hiperparámetros a buscar | Prioridad |
|---|---|---|
| LogisticRegression | C, penalty, class_weight | Alta (baseline) |
| SVM (LinearSVC) | C | Alta |
| RandomForest | n_estimators, max_depth, min_samples_split | Alta |
| XGBoost | n_estimators, learning_rate, max_depth, gamma, subsample | Alta |
| LightGBM | n_estimators, learning_rate, num_leaves, subsample | Alta |
| CatBoost | iterations, learning_rate, depth | Media |
| MLP (opcional) | hidden_layer_sizes, alpha | Baja |

### Estrategia de GridSearch
- **Stratified K-Fold** con 5 folds para todos los modelos
- RandomizedSearchCV para modelos más pesados (XGBoost, LightGBM, CatBoost)
- GridSearchCV para modelos livianos (LogisticRegression, SVM)
- F1-weighted como métrica de optimización (por el multiclase)
- Fijar `random_state=42` en TODOS los modelos

### Reproducibilidad
- `np.random.seed(42)`
- `random.seed(42)`
- `sklearn.set_config(transform_output="default")`

### Almacenamiento
- Guardar mejor modelo de cada tipo en `models/` como `.pkl`
- Guardar resultados del grid en `results/grid_search_results.csv`

### Si el tiempo de entrenamiento es muy largo
- Opción 1: Entrenar solo con últimas 5 temporadas (~80K filas)
- Opción 2: Reducir grid a 2–3 valores por hiperparámetro
- Opción 3: Aumentar early_stopping_rounds en boosting

### Registro de métricas por modelo
| Modelo | Best Params | Train Time | Accuracy (val) | F1 (val) |
|---|---|---|---|---|
| LogisticRegression | ... | ... | ... | ... |
| SVM | ... | ... | ... | ... |
| RandomForest | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... |
| LightGBM | ... | ... | ... | ... |
| CatBoost | ... | ... | ... | ... |

---

## Fase 4: Evaluation (02:30–04:00)

**Responsable**: Luis
**Notebook**: `notebooks/3.0-modeling-evaluation.ipynb` (segunda mitad)

### Métricas por modelo (en test set)
- Accuracy
- Precision (macro, weighted)
- Recall (macro, weighted)
- F1-Score (macro, weighted)
- Cohen's Kappa
- Log Loss
- ROC-AUC (One-vs-Rest macro)

### Visualizaciones (guardar en `figures/`)
- Matriz de confusión para mejor modelo (`figures/confusion_matrix.png`)
- Curvas ROC-AUC comparativas (`figures/roc_auc_comparison.png`)
- Curvas Precision-Recall (`figures/pr_curves.png`)
- Comparación de Accuracy entre modelos (barplot) (`figures/model_comparison.png`)

### Análisis estadístico
- **Wilcoxon signed-rank test** entre el mejor modelo y cada baseline
- Reportar p-values y significancia (α = 0.05)
- Tabla: "Nuestro modelo supera significativamente a XGBoost? p = 0.003 → Sí"

### Interpretabilidad
- **SHAP values** para el mejor modelo:
  - Summary plot (`figures/shap_summary.png`)
  - Bar plot de importancia global (`figures/shap_importance.png`)
- **Permutation Importance** (comparación)
- **Feature Importance** nativa del modelo (RandomForest/XGBoost)

---

## Fase 5: Finalizar Artículo IEEE (04:00–05:30)

**Responsable**: Luis (con datos de todos)

### Secciones a completar ahora que hay resultados
1. **Abstract** (versión final)
2. **Metodología Propuesta**:
   - Pipeline diagrama (BPMN o diagrama de bloques)
   - Ecuaciones matemáticas de modelos seleccionados
   - Tabla de hiperparámetros optimizados
3. **Resultados Experimentales**:
   - Tabla comparativa con métricas de TODOS los modelos
   - Figuras del EDA y evaluación embedidas
   - Resultados de prueba estadística
4. **Discusión**:
   - ¿Por qué unos modelos superaron a otros?
   - Amenazas a la validez (interna y externa)
5. **Conclusiones y Trabajo Futuro**
6. **Referencias**: 15+ mínimo, formato IEEE

### Checklist final del artículo
- [ ] Formato IEEE doble columna
- [ ] 8–10 páginas
- [ ] Tablas con título arriba (TABLE I, TABLE II, ...)
- [ ] Figuras con leyenda abajo (Fig. 1, Fig. 2, ...)
- [ ] Ecuaciones numeradas
- [ ] Sin primera persona
- [ ] Keywords ordenadas alfabéticamente
- [ ] Referencias en estilo IEEE

---

## Fase 6: Beamer Presentation (05:30–07:00)

**Responsable**: Sergio + Arbués

### Estructura (22 diapositivas)

| Slide | Contenido | Responsable |
|---|---|---|
| 1 | Portada: título, integrantes, curso, UNI, fecha | Arbués |
| 2 | Agenda | Arbués |
| 3 | Motivación y Contexto | Arbués |
| 4 | Definición Formal del Problema | Arbués |
| 5 | Objetivos | Arbués |
| 6 | Descripción del Dataset | Arbués |
| 7 | EDA — Parte I (distribuciones) | Arbués |
| 8 | EDA — Parte II (correlaciones) | Arbués |
| 9 | Pipeline de Preprocesamiento | Sergio |
| 10 | Limpieza e Imputación | Sergio |
| 11 | Feature Engineering | Sergio |
| 12 | Arquitectura de la Metodología | Sergio |
| 13 | Modelo A — Fundamentos | Sergio |
| 14 | Modelo B — Fundamentos | Sergio |
| 15 | Resultados Experimentales (tabla) | Luis |
| 16 | Visualizaciones de Rendimiento (ROC, matrices) | Luis |
| 17 | Análisis Estadístico (Wilcoxon) | Luis |
| 18 | Discusión Crítica | Luis |
| 19 | Limitaciones y Amenazas a la Validez | Luis |
| 20 | Conclusiones | Luis |
| 21 | Trabajo Futuro | Luis |
| 22 | Cierre + GitHub + Preguntas | Todos |

### Reglas de la presentación
- **21 min total** (7 min exactos por integrante) — NO EXCEDER
- Diapositivas con diagramas, ecuaciones y gráficos (NO texto denso)
- Transiciones fluidas: "Habiendo visto los hallazgos del EDA, cedo la palabra a Sergio..."
- Prohibido leer diapositivas

---

## Fase 7: Ensayo + Preparación Final (07:00–08:00)

**Responsable**: Todos

### Actividades
1. Cada integrante practica sus 7 min con cronómetro
2. Verificar transiciones entre expositores
3. Preparar respuestas a preguntas probables:
   - "¿Por qué usaron esa métrica y no otra?"
   - "¿Cómo evitaron data leakage?"
   - "¿Qué significa que el p-value sea menor a 0.05?"
   - "¿Por qué XGBoost rindió mejor que Random Forest aquí?"
4. Verificar que el Colab corre de inicio a fin (Run All)
5. Subir commit final al repo
6. Tener el PDF del artículo listo
7. Tener el PDF del Beamer listo

---

## Reglas clave durante todo el proyecto

### Reproducibilidad
- `random_state=42` en absolutamente todo
- Data leakage: fit SOLO en train, transform en val/test
- Pipeline de sklearn para evitar fugas

### Git
- Commits incrementales (no un solo commit masivo)
- Cada integrante debe tener commits propios
- Mensajes de commit descriptivos

### Comunicación
- Grupo de WhatsApp/Telegram activo
- Cualquier bloqueo >15 min → pedir ayuda al grupo
- Si un modelo no converge, pasar al siguiente y reportar

---

## Checklist de entregables

- [ ] Repositorio GitHub con estructura completa
- [ ] README.md completo
- [ ] `notebooks/1.0-eda.ipynb`
- [ ] `notebooks/2.0-preprocessing.ipynb`
- [ ] `notebooks/3.0-modeling-evaluation.ipynb`
- [ ] `docs/paper/main.pdf` (Artículo IEEE)
- [ ] `docs/presentation/presentation.pdf` (Beamer)
- [ ] `figures/` con todas las imágenes
- [ ] `results/` con métricas en CSV/JSON
- [ ] `models/` con modelos serializados
- [ ] `requirements.txt`
- [ ] `.gitignore`
- [ ] Licencia MIT

---

*Generado el 30 Junio 2026 — Plan de emergencia para sustentación 8:00 AM del 1 Julio 2026*
