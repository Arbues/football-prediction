"""Árbol de decisión (Clase-6: entropía / ganancia de información / Gini).

Particiona el espacio en regiones ortogonales; corta la frontera no lineal del
empate alrededor de elo_diff≈0 sin supuestos de linealidad. Se muestra el codo en
max_depth (pre-pruning) con curva de validación y se afinan criterio /
min_samples_leaf por GridSearch sobre TimeSeriesSplit. Inmune a la colinealidad.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import _common as C


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    C.header("Árbol de decisión (pre-pruning por profundidad)")

    depth_range = [3, 5, 7, 9, 11, 14, 17, 20]
    base = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=50,
                                  class_weight="balanced", random_state=C.SEED)
    _, cv_mean, _, _ = C.validation_sweep(
        base, "max_depth", depth_range, d["X_train"], d["y_train"],
        "02_tree", "Árbol — codo en max_depth (entropía)")
    print("  curva max_depth (CV f1_macro):",
          {int(dpt): round(float(s), 4) for dpt, s in zip(depth_range, cv_mean)})

    grid = {"criterion": ["gini", "entropy"],
            "max_depth": [6, 8, 10, 12, 16],
            "min_samples_leaf": [20, 50, 100, 200]}
    search = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=C.SEED),
        grid, scoring=C.SCORING, cv=C.get_cv(), n_jobs=-1, refit=True)
    with C.timer("gridsearch+fit") as t:
        search.fit(d["X_train"], d["y_train"])
    clf = search.best_estimator_
    print(f"  best: {search.best_params_}  CV f1_macro={search.best_score_:.4f}")

    preamble = ["MODELO: DecisionTreeClassifier (Gain Ratio / Gini)",
                f"best_params (Grid, TimeSeriesSplit): {search.best_params_}",
                f"CV f1_macro: {search.best_score_:.4f}",
                "class_weight='balanced'", "",
                "Codo max_depth (entropía, min_samples_leaf=50) -> CV f1_macro:"]
    for dpt, sc in zip(depth_range, cv_mean):
        preamble.append(f"  depth={dpt:>2} -> {sc:.4f}")

    C.finalize_val("02_tree", "DecisionTree", "arbol", clf, d, preamble,
                   str(search.best_params_), search.best_score_, t["seconds"],
                   "Árbol de decisión")


if __name__ == "__main__":
    main()
