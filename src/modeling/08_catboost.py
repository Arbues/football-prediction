"""CatBoost (gradient boosting con ordered boosting; task_type='GPU').

Árboles simétricos (oblivious) + ordered boosting, que reduce el sesgo de
predicción (target leakage) frente al boosting clásico. auto_class_weights por el
desbalance; early_stopping sobre val. Corre en GPU. Se busca depth/lr/l2_leaf_reg
por RandomizedSearch sobre TimeSeriesSplit.
"""
import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

import _common as C

FINAL_NAME = "08_catboost"


def gpu_available():
    try:
        CatBoostClassifier(task_type="GPU", iterations=2, verbose=0).fit(
            np.random.rand(50, 4), np.random.randint(0, 3, 50))
        return True
    except Exception as e:
        print(f"  [CatBoost] GPU no disponible ({str(e)[:60]}...) -> CPU")
        return False


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    task = "GPU" if gpu_available() else "CPU"
    C.header(f"CatBoost (ordered boosting, task_type={task})")

    common = dict(loss_function="MultiClass", auto_class_weights="Balanced",
                  task_type=task, random_seed=C.SEED, verbose=0,
                  allow_writing_files=False)
    space = {"depth": randint(4, 11), "learning_rate": uniform(0.02, 0.28),
             "l2_leaf_reg": uniform(1.0, 9.0)}
    search = RandomizedSearchCV(
        CatBoostClassifier(iterations=400, **common),
        space, n_iter=15, scoring=C.SCORING, cv=C.get_cv(), n_jobs=1,
        random_state=C.SEED, refit=False)
    with C.timer(f"randomsearch({task})") as t:
        search.fit(d["X_train"], d["y_train"])
    bp = search.best_params_
    print(f"  best: { {k: round(v,4) if isinstance(v,float) else v for k,v in bp.items()} }")
    print(f"  CV f1_macro={search.best_score_:.4f}")

    clf = CatBoostClassifier(iterations=2000, early_stopping_rounds=50, **common, **bp)
    with C.timer(f"fit final({task}, early stopping)") as t2:
        clf.fit(d["X_train"], d["y_train"], eval_set=(d["X_val"], d["y_val"]))
    best_it = clf.get_best_iteration()
    print(f"  best_iteration (early stopping) = {best_it}")

    preamble = [f"MODELO: CatBoost (MultiClass, ordered boosting, task_type={task})",
                f"best_params (Randomized, TimeSeriesSplit): "
                f"{ {k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()} }",
                f"CV f1_macro: {search.best_score_:.4f}",
                f"iterations final por early_stopping (val): {best_it}",
                "desbalance: auto_class_weights='Balanced'"]
    bp_str = f"{ {k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()} }, iters={best_it}"
    C.finalize_val(FINAL_NAME, "CatBoost", "boosting-gpu", clf, d, preamble,
                   bp_str, search.best_score_, t["seconds"] + t2["seconds"],
                   "CatBoost")


if __name__ == "__main__":
    main()
