"""AdaBoost (Clase-9: boosting con aprendices débiles, alpha_m = ½ ln((1-e)/e)).

Combina árboles poco profundos secuencialmente subiendo el peso de los ejemplos
mal clasificados. Reduce SESGO (a diferencia del bagging, que reduce varianza). Es
el puente conceptual hacia el gradient boosting. Multiclase por SAMME. Se usa base
depth-3 (no stumps depth-1): con la frontera no lineal del empate, el stump es
demasiado débil y SAMME colapsa (alpha≈0, no predice D). El base lleva
class_weight='balanced' para no ignorar al empate.
"""
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import _common as C


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    C.header("AdaBoost (SAMME, árboles depth-3 balanceados)")

    stump = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                   random_state=C.SEED)
    n_range = [50, 100, 200, 300]
    base = AdaBoostClassifier(estimator=stump, learning_rate=1.0, random_state=C.SEED)
    _, cv_mean, _, _ = C.validation_sweep(
        base, "n_estimators", n_range, d["X_train"], d["y_train"],
        "04_adaboost", "AdaBoost — n_estimators (rondas de boosting)")
    print("  curva n_estimators (CV f1_macro):",
          {int(n): round(float(s), 4) for n, s in zip(n_range, cv_mean)})

    grid = {"n_estimators": [100, 200, 300],
            "learning_rate": [0.1, 0.5, 1.0]}
    search = GridSearchCV(
        AdaBoostClassifier(estimator=stump, random_state=C.SEED),
        grid, scoring=C.SCORING, cv=C.get_cv(), n_jobs=-1, refit=True)
    with C.timer("gridsearch+fit") as t:
        search.fit(d["X_train"], d["y_train"])
    clf = search.best_estimator_
    print(f"  best: {search.best_params_}  CV f1_macro={search.best_score_:.4f}")

    preamble = ["MODELO: AdaBoostClassifier (SAMME) con árboles depth-3 balanceados",
                f"best_params (Grid, TimeSeriesSplit): {search.best_params_}",
                f"CV f1_macro: {search.best_score_:.4f}", "",
                "Barrido n_estimators (learning_rate=1.0) -> CV f1_macro:"]
    for n, sc in zip(n_range, cv_mean):
        preamble.append(f"  n={n:>3} -> {sc:.4f}")

    C.finalize_val("04_adaboost", "AdaBoost", "ensemble-boosting", clf, d, preamble,
                   str(search.best_params_), search.best_score_, t["seconds"],
                   "AdaBoost")


if __name__ == "__main__":
    main()
