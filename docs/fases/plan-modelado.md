# Plan de Modelado — Fase 3 (Predicción de resultado H/D/A)

> Documento de diseño. Aterriza los modelos, su fundamentación matemática (anclada
> a lo visto en el curso), el protocolo de validación y la selección de
> hiperparámetros, en continuidad directa con lo decidido en el preprocesamiento
> (Fase 2). No es un notebook: es el mapa que el notebook `3.0-modeling-evaluation`
> va a ejecutar.

Semilla global `random_state=42` en todo. `fit` solo en train. Optimización por
`f1_macro`. Cómputo en GPU (RTX 3060) para los boosting.

---

## 1. El hilo: qué del EDA y el preprocesamiento condiciona el modelado

Cada decisión de modelado se justifica en una propiedad medida de los datos, no en
preferencia. Esta es la tabla de trazabilidad que el jurado va a querer ver:

| Hallazgo (Fase 1/2) | Consecuencia en el modelado |
|---|---|
| Desbalance **IR=1.68** (H 44.6% / A 28.9% / D 26.5%), moderado | `class_weight='balanced'` de base; **no** optimizar accuracy sino `f1_macro`. |
| El **empate D no es mayoritario en ninguna región** del espacio (tasa máx ~30%); recall con techo estructural <35% | Reportar F1 **por clase**; SMOTE solo-en-train como experimento para D; no castigar al modelo por accuracy. |
| `elo_diff` y `form5_diff` separan H>D>A monótonamente; D queda en el medio | Frontera de decisión **no lineal alrededor del 0**: favorece árboles/ensembles sobre modelos lineales puros. |
| **Multicolinealidad**: los 5 diffs son combinaciones lineales exactas de sus componentes (PCA halló **5 autovalores nulos**; VIF→∞) | Modelos lineales/SVM: regularización L2 (Tikhonov) **obligatoria** o usar la variante **PCA** (k=13). Árboles: inmunes, usan la matriz cruda. |
| Elo faltante **estructural por liga** (ligas no europeas 100% sin Elo); flag `elo_missing` | Los árboles pueden **particionar por `elo_missing`** y usar Form/rolling donde no hay Elo. Señal explícita, no ruido. |
| **Deriva temporal + COVID** (home-win cae de ~47% a ~42%) | Validación **temporal** (holdout por fecha) + `TimeSeriesSplit` en la CV; jamás K-Fold aleatorio. |
| Features **mayoritariamente numéricas y de baja dimensión** (62, de las cuales 38 son one-hot de liga) | No hay maldición de la dimensionalidad grave; PCA es más higiene/ilustración que necesidad para los árboles. |
| Sin cuotas (decisión de diseño) | El modelo predice **sin la muleta del bookmaker**; el techo baja a ~50-55% acc y eso es lo honesto. |

**Ancla empírica ya medida** (LogReg `balanced` sin tunear, train→val): accuracy
0.448, **f1_macro 0.420**, recall del empate 0.22. Supera al Dummy en f1_macro
(0.20→0.42) pero apenas en accuracy (0.439→0.448). Ese contraste es la prueba
viva de por qué la métrica correcta es f1_macro, no accuracy.

---

## 2. Protocolo de validación

### 2.1 Partición temporal (ya construida en Fase 2)

- **train** ≤ 2021 (191 099 filas) · **val** 2022–2023 (24 669) · **test** 2024–2025 (14 786).
- Frontera limpia verificada: ninguna fila de train es posterior a una de val/test.

Justificación teórica. La desigualdad de Hoeffding y la cota VC exigen que la
hipótesis se fije **antes** de tocar el conjunto de evaluación:

$$P\big[\,|E_{in}(g)-E_{out}(g)|>\epsilon\,\big]\le 4\,m_{\mathcal H}(2N)\,e^{-\frac{1}{8}\epsilon^2 N}.$$

Un K-Fold aleatorio sobre datos con orden temporal rompe ese supuesto: mete
partidos del futuro en el entrenamiento y el $E_{out}$ estimado deja de ser
insesgado. El holdout por fecha respeta el calendario real.

### 2.2 CV interna para hiperparámetros: `TimeSeriesSplit`

La selección de hiperparámetros se hace **dentro de train** con `TimeSeriesSplit`
(K=5 pliegues expansivos): el pliegue $i$ entrena con $[t_0, t_i]$ y valida con
$[t_i, t_{i+1}]$, siempre hacia adelante. El score de un modelo es el promedio de
f1_macro sobre los pliegues:

$$\text{CV}(\lambda)=\frac{1}{K}\sum_{i=1}^{K} \text{f1}_{\text{macro}}\big(g^{-}_{\lambda,i}\big).$$

Esto NO convierte el problema en serie de tiempo: los modelos siguen siendo
clasificadores estándar (árboles, SVM, boosting). `TimeSeriesSplit` es solo el
esquema de remuestreo honesto para elegir $\lambda$ sin fuga temporal. El `val`
2022–2023 queda para la selección final entre familias y `test` intacto para el
número que se reporta.

### 2.3 Higiene anti-fuga

El `preprocessor.joblib` (StandardScaler + OneHot + medianas de imputación) se
ajustó **solo en train**. En modelado se aplica `transform` ciego a val/test. Toda
búsqueda de hiperparámetros ocurre dentro de train.

---

## 3. Manejo del desbalance

Dos estrategias, comparadas de forma controlada:

1. **`class_weight='balanced'`** (estrategia primaria). Reescala la pérdida por la
   frecuencia inversa de clase, $w_c = N/(K\,N_c)$, sin alterar la distribución de
   datos. Es lo adecuado para un desbalance moderado (IR=1.68) y evita inventar
   ejemplos sintéticos.
2. **SMOTE solo-en-train** (experimento para el empate). Interpola vecinos de la
   clase minoritaria D: $x_{\text{new}} = x_i + \delta\,(x_{zi}-x_i)$, $\delta\sim U(0,1)$,
   $x_{zi}$ un vecino kNN de $x_i$. Se aplica **exclusivamente al train** después
   del split (nunca a val/test; sería fuga). Choi et al. (2023) reportan que el
   balanceo mejora la detección del empate; lo verificamos, no lo asumimos.

Comparación: mismo modelo con (a) sin balanceo, (b) class_weight, (c) SMOTE →
tabla de f1_macro y recall de D. Requiere instalar `imbalanced-learn`.

---

## 4. Métrica de optimización y evaluación

**Optimización**: `f1_macro` (promedia el F1 de H, D y A con igual peso, así el
empate cuenta lo mismo que las clases fáciles).

$$\text{F1}_{\text{macro}}=\frac{1}{3}\sum_{c\in\{H,D,A\}} \frac{2\,P_c R_c}{P_c+R_c}.$$

**Reporte completo (rúbrica 3.9)**, en test: Precision/Recall/F1 (macro, weighted
y **por clase**), **Cohen's Kappa** $\kappa=\frac{p_o-p_e}{1-p_e}$, **ROC-AUC**
One-vs-Rest macro, matriz de confusión, y log-loss. La matriz de confusión es
central: nos interesa **dónde** se confunde el empate (esperamos que se lo coman
H y A, coherente con el EDA).

**Significancia estadística**: prueba de **Wilcoxon** de rangos con signo sobre los
f1_macro por pliegue (o por bloque temporal) entre el mejor modelo y cada baseline,
$\alpha=0.05$. Responde "¿la mejora es real o azar?".

---

## 5. Modelos y su fundamentación matemática

Se distingue lo **visto en clase** (base obligatoria, defendible ante pregunta
cruzada) de lo **añadido** (eleva el trabajo, se justifica por los datos).

### 5.0 Baselines (piso a superar)

- **DummyClassifier(most_frequent)**: predice siempre H. Accuracy 0.439 en val,
  f1_macro 0.20. Es el piso trivial.
- **Naive Bayes gaussiano** (sílabo, Unidad III — clasificación Bayesiana). Asume
  independencia condicional: $\hat y=\arg\max_c P(c)\prod_j P(x_j\mid c)$ con
  $P(x_j\mid c)=\mathcal N(\mu_{jc},\sigma_{jc}^2)$. Barato y alineado al curso.
  Su supuesto de independencia lo penalizan nuestros features correlacionados
  (Elo↔Form↔odds), así que sirve de **contraste**: si NB pierde feo, es evidencia
  de que la interacción entre variables importa.

### 5.1 Vistos en clase

#### 5.1.1 Árbol de decisión — Clase-6 (DT-IG)

Particiona el espacio en regiones ortogonales $R_1,\dots,R_M$; predice la clase
mayoritaria de cada región. El split se elige maximizando la **ganancia de
información** (= información mutua):

$$IG(Y,X)=H(Y)-H(Y\mid X),\qquad H(Y)=-\sum_{c} p(c)\log_2 p(c).$$

C4.5 corrige el sesgo hacia atributos de alta cardinalidad con el **Gain Ratio**
$GR=IG/\text{SplitInfo}$; CART usa **Gini** $1-\sum_c p_c^2$ (aproximación barata de
la entropía). Encaja con nuestros datos porque la frontera del empate es no lineal
alrededor de `elo_diff≈0` y el árbol la corta sin supuestos de linealidad. Su
debilidad (alta varianza, inestabilidad) es justo lo que los ensembles corrigen.

#### 5.1.2 SVM — Clase-7

Maximiza el margen geométrico $1/\lVert w\rVert$ resolviendo el problema primal

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2 \quad \text{s.a.}\quad y_n(w^\top x_n+b)\ge 1,$$

cuyo dual (multiplicadores $\alpha_n\ge 0$, KKT) es

$$\max_{\alpha}\ \sum_n \alpha_n-\tfrac12\sum_{n,m}y_n y_m\,\alpha_n\alpha_m\,x_n^\top x_m,\quad \textstyle\sum_n \alpha_n y_n=0.$$

Los **vectores de soporte** son los $x_n$ con $\alpha_n>0$. El producto interno
$x_n^\top x_m$ es donde entra el **kernel** (extensión vía Bishop Cap. 7, en el
sílabo): $K(x_n,x_m)=\exp(-\gamma\lVert x_n-x_m\rVert^2)$ (RBF) permite fronteras
no lineales, y la constante **C** del *soft-margin* controla el compromiso
margen/error. Multiclase H/D/A por One-vs-One. Requiere estandarización (hecha) y
sufre la colinealidad → aquí usaremos la **variante PCA** o L2. Costo: SVM-RBF
escala mal con 191k filas; se acota con submuestreo estratificado o `LinearSVC`.

#### 5.1.3 Random Forest — Clase-9

Bagging + selección aleatoria de atributos. Entrena $B$ árboles sin poda sobre
réplicas bootstrap y promedia por votación. La varianza del promedio de árboles con
correlación $\rho$ es

$$\rho\,\sigma^2+\frac{1-\rho}{B}\,\sigma^2,$$

y restringir cada split a $m<p$ atributos **descorrelaciona** los árboles (baja
$\rho$), que es la clave de RF. Trae **error OOB** gratis (el ~36.8% no muestreado,
$(1-1/N)^N\to e^{-1}$) y **importancia por permutación**. Es de los ganadores
recurrentes en la literatura de este problema; inmune a la colinealidad y al Elo
faltante (particiona por `elo_missing`).

#### 5.1.4 Boosting — Clase-9

AdaBoost combina aprendices débiles secuencialmente:

$$H(x)=\operatorname{sign}\Big(\sum_m \alpha_m h_m(x)\Big),\qquad \alpha_m=\tfrac12\ln\frac{1-\epsilon_m}{\epsilon_m},$$

subiendo el peso de los ejemplos mal clasificados en cada ronda. Reduce **sesgo**
(a diferencia de bagging, que reduce varianza). Es el puente conceptual hacia el
gradient boosting de 5.2.

### 5.2 Valor añadido (no visto en clase, justificado por los datos)

#### 5.2.1 Regresión Logística multinomial

Baseline lineal fuerte y calibrable. Softmax + entropía cruzada:

$$P(y=c\mid x)=\frac{e^{w_c^\top x}}{\sum_{k} e^{w_k^\top x}},\qquad \mathcal L=-\sum_i \log P(y_i\mid x_i)+\lambda\lVert W\rVert^2.$$

El término $\lambda\lVert W\rVert^2$ es **exactamente** la regularización de Tikhonov
de Clase-2 ($\theta=(X^\top X+\lambda I)^{-1}X^\top y$): vuelve invertible la matriz
singular por los diffs colineales. Por eso podemos incluir componentes y diffs a la
vez. Ya medido: f1_macro 0.42.

#### 5.2.2 Gradient Boosting: XGBoost / LightGBM / CatBoost

Generalizan AdaBoost a descenso de gradiente funcional (Hastie Cap. 10). En cada
ronda ajustan un árbol al gradiente de la pérdida; XGBoost usa además el segundo
orden:

$$\mathcal L^{(t)}\approx\sum_i\Big[g_i f_t(x_i)+\tfrac12 h_i f_t(x_i)^2\Big]+\Omega(f_t),\quad \Omega(f)=\gamma T+\tfrac12\lambda\lVert w\rVert^2,$$

con $g_i,h_i$ el gradiente y hessiano de la entropía cruzada softmax. Son el estado
del arte en tabular y los punteros de la literatura del problema (CatBoost/GBT +
ratings). **LightGBM** (crecimiento *leaf-wise* + histogramas) es rápido; **CatBoost**
maneja categóricas nativas (útil para `Division`) y reduce el *target leakage* con
*ordered boosting*. Corren en **GPU** (`device='cuda'`, verificado). Riesgo:
sobreajuste si se dejan crecer sin `early_stopping` sobre val.

#### 5.2.3 Ensamble final: Voting / Stacking

Es la contribución del trabajo (ningún paper compara estos 6+ modelos a esta
escala). **Soft-Voting** promedia probabilidades; **Stacking** entrena un
meta-modelo (LogReg) sobre las predicciones OOF de los base. El Teorema del Jurado
de Condorcet lo respalda: con modelos mejor que el azar y **diversos** (un lineal,
un RF, un boosting), el consenso mejora. Aquí la diversidad es real porque las
familias fallan distinto.

#### 5.2.4 Opcionales

- **MLP tabular** (sílabo lo lista, no se profundizó): 1–2 capas, ReLU, dropout.
  Costo/beneficio dudoso vs boosting en tabular; se deja como exploración.
- **t-SNE / UMAP**: solo **visualización** de separabilidad de las 3 clases en 2D
  (una figura para el paper), no como features.

---

## 6. Selección de hiperparámetros (sin números mágicos)

Metodología idéntica en espíritu a la del PC-4 (barrer un rango, elegir por un
criterio con gráfico, reportar los números reales): cada hiperparámetro clave se
barre con **curva de validación** sobre `TimeSeriesSplit`, graficando f1_macro de
train vs CV. El punto elegido es donde la CV deja de subir (control del gap
$E_{in}-E_{out}$, o sea sesgo-varianza de Abu-Mostafa Cap. 4). Los espacios
grandes se buscan con `RandomizedSearchCV`; los chicos con `GridSearchCV`.

| Modelo | Hiperparámetros y rango | Búsqueda | Criterio |
|---|---|---|---|
| LogReg | `C ∈ {0.01…10}` (log), `penalty=l2`, `class_weight` | Grid | curva de validación en `C` (regularización) |
| Árbol | `max_depth ∈ {3…20}`, `min_samples_leaf ∈ {1…200}`, `ccp_alpha` | Grid | codo en `max_depth` (pre-pruning) |
| SVM-RBF | `C ∈ {0.1…100}`, `gamma ∈ {scale, 0.001…1}` | Randomized | f1_macro CV; submuestreo por costo |
| Random Forest | `n_estimators ∈ {200…800}`, `max_features ∈ {sqrt,log2,0.3}`, `max_depth`, `min_samples_leaf` | Randomized | estabilización del **error OOB** + f1_macro CV |
| XGBoost/LGBM | `learning_rate ∈ {0.01…0.3}`, `max_depth/num_leaves`, `n_estimators` (con `early_stopping`), `subsample`, `colsample_bytree`, `reg_lambda` | Randomized (Optuna opcional) | f1_macro CV + early stopping en val |
| CatBoost | `depth ∈ {4…10}`, `learning_rate`, `l2_leaf_reg`, `iterations` | Randomized | f1_macro CV |
| Voting/Stacking | pesos / meta-modelo | Grid corto | f1_macro en val |

Reproducibilidad: `random_state=42` en cada estimador, en la búsqueda y en los
folds. Cada modelo reporta su tabla `best_params + f1_macro CV + tiempo`, como en
el PC-4.

---

## 7. Reducción de dimensionalidad: qué matriz usa cada modelo

- **Árboles / RF / boosting** → matriz **cruda** `X_*.parquet` (62 features con
  nombres). Inmunes a colinealidad y a escala; conservar nombres es lo que habilita
  **SHAP** e importancia nativa en Fase 4.
- **LogReg / SVM / (kNN)** → pueden usar `X_*.parquet` con L2, o la variante
  **`X_*_pca.parquet`** (13 PC + flags + one-hot). La variante PCA elimina de raíz
  la colinealidad (los 5 autovalores nulos) y ataca la "maldición de la
  dimensionalidad" que la rúbrica menciona para modelos de distancia.

Se comparan ambas rutas para al menos un modelo lineal, para tener evidencia de si
PCA ayuda o no en este dataset (probablemente marginal, dado que D=62 es bajo).

---

## 8. Presupuesto de cómputo

- GPU **RTX 3060 Laptop (6 GB)**; XGBoost `device='cuda'` verificado. LightGBM/CatBoost
  con soporte GPU análogo.
- Dataset **completo (191k train)** para árboles/boosting/LogReg (rápidos).
- **SVM-RBF**: coste cuadrático → submuestra estratificada de train (p. ej. 30–40k)
  o `LinearSVC`; se documenta el submuestreo (no se oculta).
- Fallback del notebook de Sergio: si un `fit` supera ~8 min, baja a submuestra 30%.
- Instalar en la máquina de entrenamiento: `lightgbm catboost imbalanced-learn`
  (y `optuna` si se usa la búsqueda bayesiana).

---

## 9. Entregables de Fase 3 y enganche con Fase 4

- `models/*.pkl` (mejor de cada familia) + `results/grid_search_results.csv`.
- Predicciones OOF/val/test para el Wilcoxon y el stacking.
- La matriz cruda con nombres (`X_*.parquet`) es lo que Fase 4 (Luis) necesita para
  **SHAP** y permutation importance; por eso se guardó en parquet y no en `.npy`.

**Orden de ejecución sugerido**: baselines → modelos de clase (árbol, SVM, RF,
boosting) → LogReg → boosting avanzado (XGB/LGBM/CatBoost) → ensamble → evaluación
completa + Wilcoxon + SHAP.
