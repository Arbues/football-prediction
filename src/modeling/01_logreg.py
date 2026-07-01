"""Regresión Logística multinomial (softmax + entropía cruzada, L2 = Tikhonov).

Baseline lineal fuerte. El término L2 vuelve invertible la matriz singular por
los 5 diffs colineales (autovalores nulos del PCA). Se barre C (inverso de la
regularización) con curva de validación sobre TimeSeriesSplit y se elige donde la
CV deja de mejorar. class_weight='balanced' por el desbalance moderado.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import _common as C


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    C.header("Regresión Logística multinomial (L2, class_weight=balanced)")

    C_range = np.logspace(-2, 1, 7)  # 0.01 ... 10
    base = LogisticRegression(solver="lbfgs", max_iter=2000,
                              class_weight="balanced", random_state=C.SEED)
    best_C, cv_mean, _, _ = C.validation_sweep(
        base, "C", C_range, d["X_train"], d["y_train"],
        "01_logreg", "LogReg — barrido de C (regularización L2)", logx=True)
    best_i = int(np.argmax(cv_mean))
    print(f"  C* = {best_C:.4g}  (CV f1_macro={cv_mean[best_i]:.4f})")

    clf = LogisticRegression(solver="lbfgs", max_iter=2000,
                             class_weight="balanced", C=best_C, random_state=C.SEED)
    with C.timer("fit") as t:
        clf.fit(d["X_train"], d["y_train"])

    preamble = ["MODELO: LogisticRegression multinomial (softmax), penalty=L2",
                f"C elegido por CV TimeSeriesSplit: {best_C:.4g}",
                f"CV f1_macro en el óptimo: {cv_mean[best_i]:.4f}",
                "class_weight='balanced', solver='lbfgs', max_iter=2000", "",
                "Barrido (C -> CV f1_macro):"]
    for c, sc in zip(C_range, cv_mean):
        preamble.append(f"  {c:>8.4g} -> {sc:.4f}")

    C.finalize_val("01_logreg", "LogReg", "lineal", clf, d, preamble,
                   f"C={best_C:.4g}, penalty=l2, class_weight=balanced",
                   cv_mean[best_i], t["seconds"], "LogReg multinomial")


if __name__ == "__main__":
    main()
