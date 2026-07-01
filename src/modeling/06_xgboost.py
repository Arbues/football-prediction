"""XGBoost (gradient boosting de 2.º orden, Hastie Cap.10; GPU device='cuda').

Generaliza AdaBoost a descenso de gradiente funcional: en cada ronda ajusta un
árbol al gradiente/hessiano de la entropía cruzada softmax, con regularización
Omega(f)=gamma*T + ½*lambda*||w||^2. Se busca el espacio (anclado al kernel #3)
por RandomizedSearch sobre TimeSeriesSplit con n_estimators fijo; el modelo final
usa early_stopping sobre val para fijar el número de rondas. Desbalance vía
sample_weight balanceado. Estado del arte en tabular.
"""
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import _common as C

FINAL_NAME = "06_xgboost"


def build(**kw):
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", device="cuda", random_state=C.SEED,
        n_jobs=-1, **kw)


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    C.header("XGBoost (gradient boosting, GPU device='cuda')")

    space = {"learning_rate": uniform(0.02, 0.28), "max_depth": randint(3, 9),
             "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
             "reg_lambda": uniform(0.0, 5.0), "min_child_weight": randint(1, 8)}
    search = RandomizedSearchCV(
        build(n_estimators=300), space, n_iter=25, scoring=C.SCORING,
        cv=C.get_cv(), n_jobs=1, random_state=C.SEED, refit=False, verbose=0)
    with C.timer("randomsearch(GPU)") as t:
        search.fit(d["X_train"], d["y_train"])
    bp = search.best_params_
    print(f"  best: { {k: round(v,4) if isinstance(v,float) else v for k,v in bp.items()} }")
    print(f"  CV f1_macro={search.best_score_:.4f}")

    # modelo final: n_estimators grande + early stopping sobre val (2022-2023)
    sw = compute_sample_weight("balanced", d["y_train"])
    clf = build(n_estimators=1500, early_stopping_rounds=50, **bp)
    with C.timer("fit final(GPU, early stopping)") as t2:
        clf.fit(d["X_train"], d["y_train"], sample_weight=sw,
                eval_set=[(d["X_val"], d["y_val"])], verbose=False)
    best_it = clf.best_iteration
    print(f"  best_iteration (early stopping) = {best_it}")

    preamble = ["MODELO: XGBoost (multi:softprob, tree_method=hist, device=cuda)",
                f"best_params (Randomized, TimeSeriesSplit): "
                f"{ {k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()} }",
                f"CV f1_macro: {search.best_score_:.4f}",
                f"n_estimators final por early_stopping (val): {best_it}",
                "desbalance: sample_weight='balanced'"]
    bp_str = f"{ {k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()} }, n_est={best_it}"
    C.finalize_val(FINAL_NAME, "XGBoost", "boosting-gpu", clf, d, preamble,
                   bp_str, search.best_score_, t["seconds"] + t2["seconds"],
                   "XGBoost")


if __name__ == "__main__":
    main()
