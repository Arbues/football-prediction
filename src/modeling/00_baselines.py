"""Baselines: piso trivial (DummyClassifier) + Naive Bayes gaussiano (sílabo).

El Dummy fija el piso de accuracy (predecir siempre H). NB es el clasificador
Bayesiano del curso; su supuesto de independencia condicional lo penalizan
nuestras features correlacionadas (Elo/Form), así que sirve de contraste: si NB
rinde pobre en f1_macro, es evidencia de que la interacción entre variables importa.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB

import _common as C


def run_dummy(d):
    C.header("Baseline — DummyClassifier(most_frequent)")
    clf = DummyClassifier(strategy="most_frequent", random_state=C.SEED)
    with C.timer("fit") as t:
        clf.fit(d["X_train"], d["y_train"])
    proba = clf.predict_proba(d["X_val"])
    pred = clf.predict(d["X_val"])
    m = C.compute_metrics(d["y_val"], pred, proba)
    txt = ["MODELO: DummyClassifier(strategy='most_frequent')",
           "Predice siempre la clase mayoritaria (H). Piso trivial.", "",
           "== Validación (val 2022-2023) ==",
           C.report_text(d["y_val"], pred, proba)]
    C.save_report("00_dummy", "\n".join(txt))
    C.plot_confusion(d["y_val"], pred, "00_dummy", "Dummy (most_frequent)")
    C.save_model(clf, "00_dummy")
    C.append_grid_row({"model": "Dummy", "family": "baseline",
                       "best_params": "strategy=most_frequent", "cv_f1_macro": np.nan,
                       "val_f1_macro": m["f1_macro"], "val_accuracy": m["accuracy"],
                       "val_recall_D": m["per_class"]["D"]["recall"],
                       "fit_seconds": round(t["seconds"], 2)})


def run_naive_bayes(d):
    C.header("Baseline — GaussianNB (barrido var_smoothing)")
    smooth = np.logspace(-9, -3, 7)
    train_sc, cv_sc = validation_curve(
        GaussianNB(), d["X_train"], d["y_train"],
        param_name="var_smoothing", param_range=smooth,
        cv=C.get_cv(), scoring=C.SCORING, n_jobs=-1,
    )
    cv_mean = cv_sc.mean(axis=1)
    best_i = int(np.argmax(cv_mean))
    best_vs = float(smooth[best_i])
    C.plot_validation_curve(smooth, train_sc, cv_sc, "var_smoothing",
                            "00_nb", "Naive Bayes — barrido var_smoothing", logx=True)
    print(f"  var_smoothing* = {best_vs:.2e} (CV f1_macro={cv_mean[best_i]:.4f})")

    clf = GaussianNB(var_smoothing=best_vs)
    with C.timer("fit") as t:
        clf.fit(d["X_train"], d["y_train"])
    proba = clf.predict_proba(d["X_val"])
    pred = clf.predict(d["X_val"])
    m = C.compute_metrics(d["y_val"], pred, proba)
    txt = ["MODELO: GaussianNB (naive Bayes gaussiano)",
           f"var_smoothing elegido por CV TimeSeriesSplit: {best_vs:.2e}",
           f"CV f1_macro en el óptimo: {cv_mean[best_i]:.4f}", "",
           "Barrido (var_smoothing -> CV f1_macro):"]
    for vs, sc in zip(smooth, cv_mean):
        txt.append(f"  {vs:.2e} -> {sc:.4f}")
    txt += ["", "== Validación (val 2022-2023) ==",
            C.report_text(d["y_val"], pred, proba)]
    C.save_report("00_naive_bayes", "\n".join(txt))
    C.plot_confusion(d["y_val"], pred, "00_nb", "GaussianNB")
    C.plot_roc_ovr(d["y_val"], proba, "00_nb", "GaussianNB — ROC OvR")
    C.save_model(clf, "00_naive_bayes")
    C.append_grid_row({"model": "GaussianNB", "family": "baseline",
                       "best_params": f"var_smoothing={best_vs:.2e}",
                       "cv_f1_macro": round(cv_mean[best_i], 4),
                       "val_f1_macro": m["f1_macro"], "val_accuracy": m["accuracy"],
                       "val_recall_D": m["per_class"]["D"]["recall"],
                       "fit_seconds": round(t["seconds"], 2)})


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    print(f"train={d['X_train'].shape}  val={d['X_val'].shape}  test={d['X_test'].shape}")
    run_dummy(d)
    run_naive_bayes(d)


if __name__ == "__main__":
    main()
