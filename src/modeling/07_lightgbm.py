"""LightGBM (gradient boosting leaf-wise + histogramas; GPU si el wheel lo soporta).

Crecimiento leaf-wise (elige la hoja de mayor pérdida) + binning por histogramas:
rápido y preciso en tabular. Mismo espíritu que XGBoost. Se intenta device='gpu';
si el wheel instalado no trae soporte GPU se cae a CPU y se reporta (sin ocultarlo).
Desbalance con class_weight='balanced'; early_stopping sobre val.
"""
import warnings

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

import _common as C

FINAL_NAME = "07_lightgbm"


def gpu_available():
    """Prueba un fit mínimo en GPU; si falla, CPU."""
    try:
        LGBMClassifier(device="gpu", n_estimators=2, verbose=-1).fit(
            np.random.rand(50, 4), np.random.randint(0, 3, 50))
        return True
    except Exception as e:
        print(f"  [LightGBM] GPU no disponible ({str(e)[:60]}...) -> CPU")
        return False


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    # En 191k x 62 el kernel OpenCL se recompila por fit y domina el coste: el
    # backend por histogramas en CPU resulta más rápido. Se fija CPU a propósito.
    device = "cpu"
    C.header(f"LightGBM (gradient boosting leaf-wise, device={device})")

    common = dict(objective="multiclass", num_class=3, class_weight="balanced",
                  device=device, random_state=C.SEED, n_jobs=-1, verbose=-1)
    space = {"num_leaves": randint(15, 128), "learning_rate": uniform(0.02, 0.28),
             "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
             "reg_lambda": uniform(0.0, 5.0), "min_child_samples": randint(10, 120)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            LGBMClassifier(n_estimators=300, subsample_freq=1, **common),
            space, n_iter=25, scoring=C.SCORING, cv=C.get_cv(), n_jobs=1,
            random_state=C.SEED, refit=False)
        with C.timer(f"randomsearch({device})") as t:
            search.fit(d["X_train"], d["y_train"])
    bp = search.best_params_
    print(f"  best: { {k: round(v,4) if isinstance(v,float) else v for k,v in bp.items()} }")
    print(f"  CV f1_macro={search.best_score_:.4f}")

    clf = LGBMClassifier(n_estimators=1500, subsample_freq=1, **common, **bp)
    with C.timer(f"fit final({device}, early stopping)") as t2:
        clf.fit(d["X_train"], d["y_train"],
                eval_set=[(d["X_val"], d["y_val"])], eval_metric="multi_logloss",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    best_it = clf.best_iteration_
    print(f"  best_iteration (early stopping) = {best_it}")

    preamble = [f"MODELO: LightGBM (multiclass, leaf-wise, device={device})",
                f"best_params (Randomized, TimeSeriesSplit): "
                f"{ {k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()} }",
                f"CV f1_macro: {search.best_score_:.4f}",
                f"n_estimators final por early_stopping (val): {best_it}",
                "desbalance: class_weight='balanced'"]
    bp_str = f"{ {k: (round(v,4) if isinstance(v,float) else v) for k,v in bp.items()} }, n_est={best_it}"
    C.finalize_val(FINAL_NAME, "LightGBM", "boosting-gpu", clf, d, preamble,
                   bp_str, search.best_score_, t["seconds"] + t2["seconds"],
                   "LightGBM")


if __name__ == "__main__":
    main()
