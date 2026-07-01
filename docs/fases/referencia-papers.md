# Referencia — 10 papers académicos (Estado del Arte)

> Insumo directo para la sección **Trabajos Relacionados** del artículo IEEE (responsable: Luis).
> Fuente: `pc5/EXPO/referecnias/`. Nuestro proyecto: dataset `club-football-match-data-2000-2025`
> (~226k partidos, target `FTResult` H/D/A multiclase, comparación LogReg/SVM/RF/XGBoost/LightGBM/CatBoost).

---

## ⚠️ Advertencia metodológica (leer antes de comparar cifras)

Los papers **NO son comparables entre sí en números crudos** porque usan métricas y formulaciones distintas:

- **Comparables con nosotros** (accuracy/F1 de clasificación 3-vías real): **Yeung 2023** (F1=0.47, Acc=0.57) y **Atta Mills 2024** (Acc=0.83 con voting + features in-game).
- **NO comparables directos**: Stübinger 2020 (Acc=81.8% pero es *regresión* de diferencia de goles con empates excluidos), Berrar 2019/2024 (métrica **RPS** ≈0.20–0.21, probabilística).
- **Techo realista de accuracy pre-partido en 3 clases: ~52–57%.** Números de 65%+ requieren feature engineering agresivo, subconjuntos "fáciles", o features in-game (medio tiempo) que nosotros probablemente no usaremos.

Al redactar, **explicitar esta diferencia de métricas** para no comparar peras con manzanas. Es exactamente el tipo de rigor que pide la rúbrica.

---

## Tabla comparativa de brechas (borrador para el paper — Sección Trabajos Relacionados)

| # | Ref | Dataset (N, ligas) | Features clave | Modelos | Mejor resultado | Validación | Métrica |
|---|---|---|---|---|---|---|---|
| 1 | Rodrigues & Pinto 2022 | EPL, 1900 (5 temp.) | odds, ratings sofifa, forma | NB, KNN, RF, SVM, C5.0, XGB, MLR, ANN | **RF Acc 65.3%** | split temporal 4/1 temp. | Acc, profit |
| 2 | Bunker & Thabtah 2019 | *survey* (no exp.) | discute Elo, odds, forma | *revisión* (foco ANN) | — | propone SRP-CRISP-DM, **anti-CV** | — |
| 3 | Ren & Susnjak 2022 | EPL, 1140 (2019–21) | Elo, ODM, Streak, PCA, Kelly Index | LogReg, DT, RF, KNN, GB, XGB, **CatBoost**, Voting, Stacking | **CatBoost Acc 51.9%** (all), 70% en "fáciles" | ventana temporal extendida | Acc, P/R/F1 |
| 4 | Bunker, Yeung & Fujii 2024 | *survey/book* | Elo, pi-ratings, GAP, odds | *revisión* | GBT+pi-ratings SOTA | temporal + **RPS** | RPS, Acc |
| 5 | Choi, Foo & Chua 2023 | EPL, 3297 (10 temp.) | diffs, FIFA, odds, streak, Boruta→8 | RF, LogReg, LinearSVC, XGB | LogReg multiclase Acc 53.8%; RF binario 66.6% | 80/20 + 5-fold CV | Acc, F1, AUC |
| 6 | Stübinger et al. 2020 | 5 ligas EU + 2ª div, 47856 | 40 skills FIFA/jugador | RF, Boosting, SVM, LinReg, ensemble | **ensemble Acc 81.8%** ⚠️*regresión* | rolling temporal | Acc, profit |
| 7 | Yeung, Bunker & Fujii 2023 | EPL, ~1700 (2011–16) | formación, FIFA ratings (interpretable) | LinReg + **XGBoost** | **XGB F1=0.47, Acc=0.57** | split por temporada | F1, AUC, Acc |
| 8 | Berrar et al. 2019 | 216,743, 52 ligas | recency + rating learning (tipo Elo) | **k-NN, XGBoost** | **kNN+rating RPS 0.2054** (ganó Challenge) | 85/15 + 3-fold | **RPS** |
| 9 | Berrar et al. 2024 | >300,000, 51 ligas | recency, super/meta league | k-NN, ANN, ordinal forest, NB | kNN2 RMSE 1.62; ANN RPS 0.2113 (bookmaker ganó) | LOO-CV, 85/15 | RPS, RMSE |
| 10 | Atta Mills et al. 2024 | Eredivisie+2, 1411 | in-game (HT), 28 feats, SMOTE | LogReg, XGB, RF, SVM, NB, FNN, RNN, **Voting** | **Voting(RF+XGB) Acc 0.83, F1 0.81** | 7-fold CV | Acc, F1, AUC |

**Nuestra brecha / contribución** (fila "propuesta" de la tabla): ningún trabajo compara **los 6 algoritmos** (LogReg/SVM/RF/XGBoost/**LightGBM/CatBoost**) simultáneamente sobre un dataset de **~226k partidos multi-liga (2000–2025)** con métricas de clasificación por clase (Acc/F1/Kappa/AUC) + prueba de significancia (Wilcoxon). Los que tienen esa escala (Berrar) usan RPS y k-NN; los que usan nuestras métricas (Yeung, Atta Mills) tienen datasets pequeños (~1.5k).

---

## Fichas por paper

### 1. Rodrigues & Pinto (2022) — *Procedia Computer Science* 204
- **Dataset**: EPL, 1900 partidos (2013/14–2018/19). H 45.3% / D 24.7% / A 29.9%.
- **Features**: 31→18 (Boruta). Odds Bet365, ratings ataque/medio/defensa (sofifa), goles marcados/concedidos, tiros, corners, faltas, tarjetas, wins últimos 5.
- **Modelos**: NB, KNN, RF, SVM, C5.0, XGBoost, MLR, ANN.
- **Mejor**: RF **Acc 65.26%**, profit 203€ (margen 26.78%). Draw recall solo 29.76% (clase difícil).
- **Validación**: split temporal (4 temp. train / 2018-19 test). Sin k-fold.
- **Para nosotros**: casi mismo setup (H/D/A, LogReg/SVM/RF/XGB). Nos diferenciamos en escala (1900 vs 226k) y añadimos LightGBM/CatBoost. Su empate débil es contrastable.

### 2. Bunker & Thabtah (2019) — *Applied Computing and Informatics* 15(1)
- **Tipo**: survey/framework, no experimental.
- **Aporte**: framework **SRP-CRISP-DM**. Recomienda **NO usar CV estándar** (rompe orden temporal); split round-by-round; accuracy solo si balanceado.
- **Para nosotros**: soporte metodológico para justificar validación temporal y la advertencia sobre accuracy en datos desbalanceados (empate minoritario). Cita clave para defender nuestro protocolo.

### 3. Ren & Susnjak (2022) — arXiv:2211.15734
- **Dataset**: EPL 2019–2021, 1140 partidos. Odds de 6 bookmakers.
- **Features**: 52 → Elo, ODM (ofensivo/defensivo), Streak Index, win%/draw%, PCA. **Kelly Index** para clasificar dificultad.
- **Modelos**: LogReg, DT, RF, KNN, GB, XGBoost, CatBoost, Voting, Stacking. Selección con SHAP+RFE.
- **Mejor**: CatBoost **Acc 51.9%** (todos), 70% en partidos "fáciles", 41% en "difíciles".
- **Validación**: ventana temporal extendida, RandomizedSearchCV. Anti-CV estándar (cita a Bunker).
- **Para nosotros**: comparte casi todos nuestros modelos + target. Aporta SHAP (que Luis usará en Fase 4) y features Elo/Streak. Nos diferenciamos en escala.

### 4. Bunker, Yeung & Fujii (2024) — book chapter (arXiv:2403.07669)
- **Tipo**: survey de referencia. Cataloga datasets (incl. **Open International Soccer DB, 216k partidos** — escala como la nuestra).
- **Aporte**: SOTA = **gradient boosting (CatBoost) sobre ratings específicos (pi-ratings)**. Recomienda temporal split + **RPS**. Odds del bookmaker = baseline muy difícil de batir.
- **Para nosotros**: valida incluir CatBoost/GBT como punteros. Señala como PENDIENTE la comparación empírica amplia multi-modelo → esa es literalmente nuestra contribución.

### 5. Choi, Foo & Chua (2023) — *MENDEL* 29(2)
- **Dataset**: EPL 2012/13–2021/22, 3297 partidos tras FE. FIFA ratings, venue, referee.
- **Modelos**: RF, LogReg, LinearSVC, XGBoost. **Stratified vs balanced sampling**; multiclase vs binario.
- **Mejor**: LogReg multiclase Acc ~53.8% (F1 0.498, AUC 0.718); RF binario Acc 66.6%. Profit: LogReg binario +7.57%.
- **Hallazgo**: **balanced sampling > stratified** (permite predecir empates); binario (Win/Non-Win) rinde mejor que multiclase.
- **Para nosotros**: directamente aplicable el tema sampling/desbalance y el problema del empate. Añadimos LightGBM/CatBoost.

### 6. Stübinger, Mangold & Knoll (2020) — *Applied Sciences* 10(1):46
- **Dataset**: 5 ligas EU + 2ª div, **47,856 partidos** (2006–2018). 40 skills FIFA/jugador.
- **Modelos**: RF, Boosting, SVM, LinReg, ensemble. ⚠️ **Regresión** (diferencia de goles), no clasificación.
- **Mejor**: ensemble **Acc 81.77%** (derivada, empates excluidos), profit +1.58%/apuesta (Wilcoxon p<0.0001).
- **Para nosotros**: NO comparar su 81.8% con nuestra Acc (formulación distinta). Tomar idea de features de calidad de jugador (FIFA) y del ensemble + del test de Wilcoxon para profit.

### 7. Yeung, Bunker & Fujii (2023) — *PLOS ONE* 18(4):e0284318
- **Dataset**: EPL 2011/12–2015/16, ~1700 partidos. Formación + FIFA ratings (enfoque interpretable, "controlable por el coach").
- **Modelos**: LinReg + **XGBoost**. Baselines: odds, team-rating, ANN.
- **Mejor**: XGBoost **F1=0.47, AUC=0.54, Acc=0.57** — supera a las odds (F1 0.39).
- **Para nosotros**: ⭐ referencia de **techo realista** de F1/Acc en clasificación H/D/A (directamente comparable). Nos diferenciamos por escala y por 6 modelos.

### 8. Berrar, Lopes & Dubitzky (2019) — *Machine Learning* 108
- **Dataset**: 2017 Soccer Prediction Challenge, **216,743 partidos**, 52 ligas. Datos "simples" (fecha/equipos/liga/marcador) — **casi idéntico a nuestro setup**.
- **Features**: recency feature extraction + rating feature learning (alternativa a Elo).
- **Modelos**: k-NN, XGBoost. **Mejor**: kNN+rating **RPS 0.2054** (ganó el Challenge); XGB+rating después 0.2023.
- **Hallazgo**: el feature engineering (conocimiento de dominio) importa más que el algoritmo; k-NN simple ganó a boosted trees.
- **Para nosotros**: ⭐ escala comparable (216k) y datos simples. Tomar recency/rating features. Diferencia: ellos usan RPS, nosotros Acc/F1 con 6 modelos.

### 9. Berrar, Lopes & Dubitzky (2024) — *Machine Learning* 113
- **Dataset**: 2023 Challenge, **>300,000 partidos**, 51 ligas. Datos simples.
- **Modelos**: k-NN, ANN, ordinal forests, NB. **Mejor**: kNN2 RMSE 1.62 (Task 1 score); en Task 2 (resultado) **ni el mejor ML batió al bookmaker** (ANN RPS 0.2113 vs bookmaker 0.2063).
- **Para nosotros**: escala casi idéntica. Advertencia: empates difíciles, odds imbatibles. No comparan nuestros 6 modelos ni reportan Acc/F1 → brecha.

### 10. Atta Mills et al. (2024) — *Journal of Big Data* 11:170
- **Dataset**: Eredivisie + Scottish + Belgian, 1411 partidos. Features **in-game (medio tiempo)** + 28 feats, SMOTE.
- **Modelos**: LogReg, XGB, RF, SVM, NB, FNN, RNN, **Voting(RF+XGB)**. **7-fold CV**.
- **Mejor**: **Voting Acc 0.83, F1 0.81**. Draw sigue siendo la clase difícil (F1 0.19–0.47 en individuales).
- **Para nosotros**: ⭐ el más alineado en métricas/diseño (Acc/F1/P/R/AUC, CV, desbalance). CUIDADO: su 0.83 usa features in-game (HT) que nosotros no tendremos → no comparar directo. Tomar plantilla de tablas por clase + idea de voting.

---

## Síntesis accionable para el paper (Luis)

1. **Narrativa de brecha**: "Los estudios previos o bien alcanzan escala (Berrar, 216k–300k) usando métricas probabilísticas (RPS) y pocos algoritmos, o bien comparan clasificadores con métricas estándar (Yeung, Atta Mills) sobre datasets pequeños (~1.5k). **Ninguno ofrece una comparación empírica de 6 clasificadores modernos (incl. LightGBM/CatBoost) con métricas de clasificación por clase y prueba de significancia sobre ~226k partidos multi-liga.**"
2. **Consenso a citar**: (a) el empate es la clase más difícil (recall <35%) — papers 1,3,5,10; (b) techo pre-partido ~52–57% en 3 clases — papers 3,4,7; (c) gradient boosting/RF son los ganadores recurrentes — papers 1,3,4,10.
3. **Debate a posicionar**: validación temporal (Bunker, papers 2/4/8) vs k-fold CV (Choi, Atta Mills, papers 5/10). Decidir y justificar nuestra postura (recomendado: split temporal por el `MatchDate`, alineado con la mayoría de la literatura de escala).
4. **≥15 referencias**: estos 10 + 5 adicionales (buscar: pi-ratings Constantinou, Dixon-Coles, Elo Hvattum, SMOTE Chawla, SHAP Lundberg).
