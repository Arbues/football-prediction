# `models/` — Modelos entrenados (serializados)

Modelos ya entrenados y el preprocesador ajustado, listos para inferencia sin reentrenar.
Todos se generan al ejecutar `src/modeling/` (semilla fija `random_state = 42`).

## Contenido

| Archivo | Modelo |
|---|---|
| `00_dummy.pkl`, `00_naive_bayes.pkl` | Líneas base (clase mayoritaria y Naive Bayes gaussiano). |
| `01_logreg.pkl` | Regresión logística multinomial (L2). |
| `02_tree.pkl` | Árbol de decisión. |
| `03_random_forest.pkl` | Random Forest. |
| `04_adaboost.pkl` | AdaBoost. |
| `05_svm_linear.pkl`, `05_svm_rbf.pkl` | SVM lineal y de kernel radial. |
| `06_xgboost.pkl`, `07_lightgbm.pkl`, `08_catboost.pkl` | Motores de *gradient boosting*. |
| `10_voting.pkl`, `10_stacking.pkl` | Ensambles (**Stacking** es el modelo propuesto). |
| `preprocessor.joblib` | `ColumnTransformer` ajustado **solo con train** (imputación + escalado + one-hot). |

## Cargar un modelo

```python
import joblib
modelo = joblib.load("models/10_stacking.pkl")
pre    = joblib.load("models/preprocessor.joblib")
# aplicar 'pre' a los datos crudos antes de 'modelo.predict(...)'
```

> Algunos archivos son grandes (Random Forest ≈ 85 MB; Voting/Stacking ≈ 86–88 MB). Si el
> `.gitignore` los excluye por tamaño, se regeneran corriendo `src/modeling/` en orden.
