"""Random Forest (Clase-9: bagging + descorrelación por m<p atributos).

Promedia B árboles sin poda sobre réplicas bootstrap; restringir cada split a
m<p atributos baja la correlación rho entre árboles y con ella la varianza del
promedio. Trae error OOB gratis e importancia por permutación. Inmune a
colinealidad y al Elo faltante (particiona por elo_missing).
"""
import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV

import _common as C


def oob_curve(best_params, d):
    """Error OOB vs n_estimators para mostrar la estabilización (figura)."""
    import matplotlib.pyplot as plt

    ns = [100, 200, 300, 400, 500]
    oob_err = []
    params = {k: v for k, v in best_params.items() if k != "n_estimators"}
    for n in ns:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True, bootstrap=True,
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=C.SEED, **params)
        rf.fit(d["X_train"], d["y_train"])
        oob_err.append(1.0 - rf.oob_score_)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ns, oob_err, "o-", color="tab:green")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("error OOB (1 - oob_score)")
    ax.set_title("Random Forest — estabilización del error OOB")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "oob_03_rf.png", dpi=130)
    plt.close(fig)
    return dict(zip(ns, oob_err))


def perm_importance(clf, d, k=6000, top=15):
    """Importancia por permutación sobre submuestra de val (coste acotado)."""
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(C.SEED)
    idx = rng.choice(len(d["X_val"]), size=min(k, len(d["X_val"])), replace=False)
    Xs, ys = d["X_val"].iloc[idx], d["y_val"][idx]
    r = permutation_importance(clf, Xs, ys, n_repeats=5, random_state=C.SEED,
                               scoring=C.SCORING, n_jobs=-1)
    order = np.argsort(r.importances_mean)[::-1][:top]
    names = [d["X_val"].columns[i].split("__")[-1] for i in order]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.barh(range(len(order))[::-1], r.importances_mean[order], color="tab:green",
            xerr=r.importances_std[order])
    ax.set_yticks(range(len(order))[::-1], names)
    ax.set_xlabel("caída de f1_macro al permutar")
    ax.set_title("Random Forest — importancia por permutación (val)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "permimp_03_rf.png", dpi=130)
    plt.close(fig)
    return list(zip(names, r.importances_mean[order].round(4)))


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    C.header("Random Forest (bagging + descorrelación)")

    space = {"n_estimators": randint(250, 550),
             "max_features": ["sqrt", "log2", 0.3],
             "max_depth": [None, 12, 20],
             "min_samples_leaf": [1, 5, 20]}
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample", bootstrap=True,
                               n_jobs=-1, random_state=C.SEED),
        space, n_iter=10, scoring=C.SCORING, cv=C.get_cv(), n_jobs=1,
        random_state=C.SEED, refit=True)
    with C.timer("randomsearch+fit") as t:
        search.fit(d["X_train"], d["y_train"])
    clf = search.best_estimator_
    print(f"  best: {search.best_params_}  CV f1_macro={search.best_score_:.4f}")

    oob = oob_curve(search.best_params_, d)
    print(f"  OOB error: {({k: round(v, 4) for k, v in oob.items()})}")
    imp = perm_importance(clf, d)
    print(f"  top importancia: {imp[:6]}")

    preamble = ["MODELO: RandomForestClassifier (bagging, max_features<p)",
                f"best_params (Randomized, TimeSeriesSplit): {search.best_params_}",
                f"CV f1_macro: {search.best_score_:.4f}",
                "class_weight='balanced_subsample'", "",
                "Error OOB (1-oob_score) vs n_estimators:"]
    for n, e in oob.items():
        preamble.append(f"  n={n:>4} -> OOB_err={e:.4f}")
    preamble += ["", "Importancia por permutación (top, sobre val):"]
    for name, val in imp:
        preamble.append(f"  {name:<22} {val:.4f}")

    C.finalize_val("03_random_forest", "RandomForest", "ensemble-bagging", clf, d,
                   preamble, str(search.best_params_), search.best_score_,
                   t["seconds"], "Random Forest")


if __name__ == "__main__":
    main()
