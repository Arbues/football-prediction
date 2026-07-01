Contexto: Proyecto Final de Minería de Datos CC442 — predicción de resultado de
partidos de fútbol (FTResult H/D/A multiclase), dataset club-football-match-data-2000-2025.
Estoy en /home/sergi/Documents/semesters/eight/dm. Ya cerramos la Fase 1 (EDA) y la
Fase 2 (PREPROCESAMIENTO + FEATURE ENGINEERING). Ahora toca la Fase 3: MODELADO
(responsable Sergio). El fit se hará EN ESTA PC (tengo una NVIDIA RTX 3060, GPU
verificada con XGBoost device='cuda'), así que vamos con todo: dataset completo.

PRIMERO recupera TODO el contexto desde memoria (engram). Llama a mem_search con estas
consultas y LEE COMPLETO lo que salga (usa mem_get_observation si algo sale truncado):
  - "football-prediction fase3 handoff modelado"   (topic_key: football-prediction/fase3-handoff — LÉELA ENTERA: estado completo del preprocesamiento, decisiones A1-B6, resultados del pipeline, precisiones de las clases)
  - "football-prediction fase2 handoff preprocesamiento"  (topic_key: football-prediction/fase2-handoff — el estado al cerrar el EDA)
  - "football-prediction EDA leakage FTResult"
  - "football-prediction metodologia EDA CRISP-DM"

LUEGO lee estos documentos del repo (ya destilan todo, NO releas PDFs de kernels):
  - pc5/football-prediction/docs/fases/plan-modelado.md    (EL PLAN: modelos, fórmulas ancladas al curso, validación, selección de hiperparámetros, métricas — es tu hoja de ruta, síguela)
  - pc5/football-prediction/docs/fases/BITACORA.md         (log inverso: qué se hizo en Fase 1 y 2 y por qué — la entrada de arriba es la de preprocesamiento)
  - pc5/football-prediction/PLANNING.md                     (Fase 3: tabla de modelos + estrategia de GridSearch)
  - pc5/football-prediction/docs/fases/referencia-papers.md (estado del arte: techo realista ~52-57%, empate difícil, boosting/RF ganadores, Wilcoxon)
  - pc5/football-prediction/docs/fases/referencia-kernels.md (espacio Optuna del kernel #3 como base para XGB/LGBM)
  - pc5/guia_proyecto_final_mineria_datos.md                 (rúbrica: 3.7 metodología con ecuaciones e hiperparámetros; 3.8 implementación /src; 3.9 métricas Precision/Recall/F1/Kappa/ROC-AUC + Wilcoxon; Sección 12 matriz de errores; 11 reproducibilidad)
  - pc5/football-prediction/notebooks/1.0-eda.ipynb          (EDA completo)
  - pc5/football-prediction/notebooks/2.0-preprocessing.ipynb (preprocesamiento completo, para no repetir y entender las 62 features)
  - pc5/football-prediction/notebooks/3.0-modeling-evaluation.ipynb (ESQUELETO plug-and-play que dejó Sergio: seeds, carga flexible de data/processed, registro de 6 modelos, GridSearch/RandomizedSearch, presupuesto por modelo, guardado a models/)

Y CARGA los datos ya procesados (salida de Fase 2, en data/processed/):
  - X_{train,val,test}.parquet + y_{train,val,test}.parquet  (MATRIZ CRUDA con nombres: 62 features. Úsala para árboles/boosting — SHAP necesita los nombres)
  - X_{train,val,test}_pca.parquet                            (VARIANTE PCA: 13 PC + flags + one-hot, para modelos lineales/distancia)
  - feature_names.json                                        (nombres de las 62 columnas)
  - models/preprocessor.joblib                                (ColumnTransformer ajustado en train, por si hay que transformar datos nuevos)
  - index_ref.parquet                                         (match_id, MatchDate, Division, split — para trazabilidad y el Wilcoxon por bloque temporal)

ENTORNO: el .venv está en la RAÍZ del repo (/home/sergi/Documents/semesters/eight/dm/.venv).
Desde pc5/football-prediction ejecuta con ../../.venv/bin/python. Ya están pandas/numpy/
sklearn/scipy/matplotlib/seaborn/shap/xgboost/pyarrow. FALTAN e instálalas con
'VIRTUAL_ENV=/home/sergi/Documents/semesters/eight/dm/.venv uv pip install lightgbm catboost imbalanced-learn optuna'
(pip normal está roto; usar uv). GPU: RTX 3060 6GB, XGBoost device='cuda' verificado
(LightGBM/CatBoost tienen soporte GPU análogo).

Tarea de esta sesión — MODELADO (Fase 3), siguiendo plan-modelado.md al pie:
  - Baselines: DummyClassifier (piso 0.439 acc) + Naive Bayes gaussiano (alineado al sílabo).
  - Modelos VISTOS EN CLASE (base obligatoria, con su fundamentación matemática):
    Árbol de Decisión (Clase-6: entropía/IG/Gini/GainRatio), SVM (Clase-7: margen/dual/KKT
    + kernel RBF vía Bishop), Random Forest y Boosting (Clase-9: bagging/OOB/descorrelación,
    AdaBoost α_m=½ln((1-ε)/ε)), + PCA (Clase-3) como reducción para lineales/distancia.
  - Modelos de VALOR AÑADIDO: Regresión Logística multinomial (softmax+entropía cruzada,
    L2=Tikhonov de Clase-2), XGBoost/LightGBM/CatBoost (gradient boosting, Hastie Cap.10,
    GPU), Voting/Stacking (la contribución del trabajo). MLP y t-SNE/UMAP opcionales.

Reglas (idénticas a Fases 1 y 2):
  - random_state=42 en TODO (estimadores, búsqueda, folds).
  - fit SOLO en train (anti-leakage); la matriz ya viene estandarizada/fit-en-train.
  - VALIDACIÓN TEMPORAL: TimeSeriesSplit (K=5) como CV interna para elegir hiperparámetros
    DENTRO de train; NADA de K-Fold aleatorio (los modelos siguen siendo clasificadores
    estándar, no series de tiempo). val 2022-23 para selección final, test 2024-25 intacto.
  - OPTIMIZAR f1_macro (el empate D pesa igual). REPORTAR: Precision/Recall/F1 (macro,
    weighted y POR CLASE), Cohen's Kappa, ROC-AUC OvR, matriz de confusión, log-loss.
  - DESBALANCE: class_weight='balanced' primario + SMOTE solo-en-train como experimento
    controlado para el empate (comparar sin balanceo vs class_weight vs SMOTE).
  - SIGNIFICANCIA: prueba de Wilcoxon (α=0.05) entre el mejor modelo y cada baseline.
  - HIPERPARÁMETROS "sin números mágicos" (estilo PC-4): barrer un rango con curva de
    validación sobre TimeSeriesSplit, elegir por el criterio (codo/estabilización de la
    CV, control del gap E_in-E_out), REPORTAR los números reales. Ver tabla en plan-modelado.md.
  - WORKFLOW script-first: primero archivos .py en src/modeling/ que corran y vuelquen
    outputs en results/ (txt: métricas por modelo, best_params, tiempos) y figures/ (png:
    matrices de confusión, ROC, curvas de validación) para que yo interprete; los lees e
    iteras; y SOLO cuando te lo diga completas el notebook 3.0-modeling-evaluation.ipynb
    (estilo académico: Fundamentación→Código→Interpretación→Discusión, español poco
    detectable como IA, Run All limpio verificado con nbconvert). Usa el .venv del proyecto.
    NO hagas commits salvo que lo pida. Al terminar, agrega entrada nueva en BITACORA.md y
    guarda un mem_save con las decisiones y resultados.

Hallazgos de Fase 1/2 que condicionan el modelado (el "hilo", ya en plan-modelado.md):
  desbalance IR=1.68 moderado; empate D difuso sin región propia (recall techo <35%);
  elo_diff/form5_diff señal monótona con D en el medio (frontera no lineal → favorece
  árboles); 5 diffs colineales exactos (VIF→∞, 5 autovalores PCA nulos → lineales usan L2
  o PCA k=13); Elo faltante estructural por liga (flag elo_missing); deriva temporal+COVID
  (→ split temporal); sin cuotas (techo honesto ~50-55% acc). Sonda ya medida: LogReg
  balanced sin tunear da acc 0.448, f1_macro 0.420, recall D 0.22.

ANTES de escribir código, hazme TODAS las preguntas necesarias para proceder: qué modelos
priorizar y en qué orden dado el presupuesto de cómputo; qué espacios de hiperparámetros
concretos usar por modelo; si SVM-RBF se corre sobre el dataset completo o submuestra
estratificada (coste cuadrático); si CatBoost/LightGBM van en GPU; si el ensamble final es
Voting o Stacking (o ambos); si comparo la ruta cruda vs PCA para algún lineal; qué modelo
se declara "el propuesto" para el Wilcoxon; y cualquier duda de la rúbrica (3.7/3.9). No
asumas nada crítico: pregunta, espera mi respuesta, y recién entonces procede. Vamos con todo.
