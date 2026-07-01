"""SVM (Clase-7: margen máximo, dual/KKT, kernel RBF vía Bishop).

Maximiza el margen geométrico; los vectores de soporte son los x_n con alpha_n>0.
El kernel RBF K(x,x')=exp(-gamma||x-x'||^2) habilita fronteras no lineales (la que
pide el empate). Coste O(n^2): el RBF se ajusta sobre una submuestra estratificada
que respeta el orden temporal (~35k); se documenta el submuestreo. LinearSVC sobre
el train completo queda como contraste lineal. Requiere estandarización (ya hecha).
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC, LinearSVC

import _common as C

SUBSAMPLE = 35000


def stratified_temporal_subsample(X, y, n, seed=C.SEED):
    """Submuestra estratificada que conserva el orden cronológico (índice ascendente)."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X, y))
    idx = np.sort(idx)  # los datos vienen ordenados por fecha -> preserva el calendario
    return X.iloc[idx], y[idx]


def run_rbf(d):
    C.header(f"SVM-RBF (submuestra estratificada temporal n={SUBSAMPLE})")
    Xs, ys = stratified_temporal_subsample(d["X_train"], d["y_train"], SUBSAMPLE)
    print(f"  submuestra: {Xs.shape}  clases: {np.bincount(ys)}")

    space = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]}
    search = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=False,
            random_state=C.SEED),
        space, n_iter=8, scoring=C.SCORING, cv=C.get_cv(3), n_jobs=-1,
        random_state=C.SEED, refit=False)
    with C.timer("randomsearch") as t:
        search.fit(Xs, ys)
    bp = search.best_params_
    print(f"  best: {bp}  CV f1_macro={search.best_score_:.4f}")

    clf = SVC(kernel="rbf", class_weight="balanced", probability=True,
              random_state=C.SEED, **bp)
    with C.timer("refit(prob=True)") as t2:
        clf.fit(Xs, ys)

    preamble = ["MODELO: SVC kernel RBF (submuestra estratificada temporal)",
                f"submuestra de entrenamiento: {SUBSAMPLE} filas (de 191099)",
                f"best_params (Randomized, TimeSeriesSplit K=3): {bp}",
                f"CV f1_macro: {search.best_score_:.4f}",
                "class_weight='balanced'; probability=True para ROC/log-loss"]
    C.finalize_val("05_svm_rbf", "SVM-RBF", "svm", clf, d, preamble, str(bp),
                   search.best_score_, t["seconds"] + t2["seconds"], "SVM-RBF")


def run_linear(d):
    C.header("LinearSVC (contraste lineal, train completo)")
    C_range = [0.01, 0.1, 1.0, 10.0]
    base = LinearSVC(class_weight="balanced", dual="auto", max_iter=5000,
                     random_state=C.SEED)
    best_C, cv_mean, _, _ = C.validation_sweep(
        base, "C", C_range, d["X_train"], d["y_train"],
        "05_svm_linear", "LinearSVC — barrido de C", logx=True)
    best_i = int(np.argmax(cv_mean))
    print(f"  C* = {best_C}  (CV f1_macro={cv_mean[best_i]:.4f})")

    # CalibratedClassifierCV envuelve LinearSVC para obtener predict_proba (ROC/log-loss)
    clf = CalibratedClassifierCV(
        LinearSVC(C=best_C, class_weight="balanced", dual="auto", max_iter=5000,
                  random_state=C.SEED), method="sigmoid", cv=3)
    with C.timer("fit(calibrado)") as t:
        clf.fit(d["X_train"], d["y_train"])

    preamble = ["MODELO: LinearSVC (contraste lineal) + calibración sigmoide",
                f"C elegido por CV TimeSeriesSplit: {best_C}",
                f"CV f1_macro en el óptimo: {cv_mean[best_i]:.4f}",
                "train completo (191099); class_weight='balanced'", "",
                "Barrido (C -> CV f1_macro):"]
    for c, sc in zip(C_range, cv_mean):
        preamble.append(f"  {c:>6} -> {sc:.4f}")
    C.finalize_val("05_svm_linear", "SVM-Linear", "svm", clf, d, preamble,
                   f"C={best_C}", cv_mean[best_i], t["seconds"], "LinearSVC")


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    run_rbf(d)
    run_linear(d)


if __name__ == "__main__":
    main()
