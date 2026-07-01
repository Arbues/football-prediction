"""Evaluación final en TEST (2024-2025, intacto) + significancia estadística.

Carga todos los modelos guardados, los evalúa en test (rúbrica 3.9 completa),
declara "el propuesto" = el mejor ensamble por f1_macro en val, y contrasta su
superioridad con la prueba de Wilcoxon de rangos con signo (alpha=0.05) sobre el
f1_macro por bloque temporal (mes) frente a cada baseline. Vuelca tabla
comparativa, matriz de confusión y ROC del propuesto, y barras comparativas.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

import _common as C

# nombre_archivo -> (etiqueta, familia)
REGISTRY = [
    ("00_dummy", "Dummy", "baseline"),
    ("00_naive_bayes", "GaussianNB", "baseline"),
    ("01_logreg", "LogReg", "lineal"),
    ("02_tree", "DecisionTree", "arbol"),
    ("03_random_forest", "RandomForest", "ensemble-bagging"),
    ("04_adaboost", "AdaBoost", "ensemble-boosting"),
    ("05_svm_rbf", "SVM-RBF", "svm"),
    ("05_svm_linear", "SVM-Linear", "svm"),
    ("06_xgboost", "XGBoost", "boosting-gpu"),
    ("07_lightgbm", "LightGBM", "boosting-gpu"),
    ("08_catboost", "CatBoost", "boosting-gpu"),
    ("10_voting", "Voting", "ensemble-final"),
    ("10_stacking", "Stacking", "ensemble-final"),
]


def load_available():
    import joblib

    out = []
    for fname, label, family in REGISTRY:
        p = C.MODELS / f"{fname}.pkl"
        if p.exists():
            out.append((fname, label, family, joblib.load(p)))
        else:
            print(f"  (omitido, no existe: {fname})")
    return out


def safe_pred(clf, X):
    pred = np.asarray(clf.predict(X)).ravel().astype(int)
    proba = clf.predict_proba(X)
    return pred, proba


def per_block_f1(y_true, y_pred, blocks, min_n=50):
    """f1_macro por bloque temporal (mes); descarta bloques con pocos partidos."""
    scores, keys = [], []
    for b in np.unique(blocks):
        m = blocks == b
        if m.sum() >= min_n:
            scores.append(f1_score(y_true[m], y_pred[m], average="macro",
                                   labels=C.CLASSES_ENC, zero_division=0))
            keys.append(str(b))
    return np.array(scores), keys


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    yte = d["y_test"]
    models = load_available()

    # --- métricas en test para todos ---
    grid = pd.read_csv(C.GRID_CSV) if C.GRID_CSV.exists() else pd.DataFrame()
    val_f1 = dict(zip(grid.get("model", []), grid.get("val_f1_macro", [])))
    preds = {}
    rows = []
    C.header("Evaluación en TEST (2024-2025)")
    for fname, label, family, clf in models:
        pred, proba = safe_pred(clf, d["X_test"])
        preds[label] = pred
        m = C.compute_metrics(yte, pred, proba)
        rows.append({
            "model": label, "family": family,
            "val_f1_macro": round(float(val_f1.get(label, np.nan)), 4),
            "test_f1_macro": round(m["f1_macro"], 4),
            "test_acc": round(m["accuracy"], 4),
            "test_f1_H": round(m["per_class"]["H"]["f1"], 4),
            "test_f1_D": round(m["per_class"]["D"]["f1"], 4),
            "test_f1_A": round(m["per_class"]["A"]["f1"], 4),
            "test_recall_D": round(m["per_class"]["D"]["recall"], 4),
            "test_kappa": round(m["kappa"], 4),
            "test_roc_auc": round(m.get("roc_auc_ovr_macro", np.nan), 4),
            "test_logloss": round(m.get("log_loss", np.nan), 4),
        })
        print(f"  {label:14} test f1_macro={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}"
              f"  recall_D={m['per_class']['D']['recall']:.4f}")
    tbl = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    tbl.to_csv(C.RESULTS / "11_test_metrics.csv", index=False)

    # --- declarar el propuesto: mejor ensamble por val f1_macro ---
    ens = [r for r in rows if r["family"] == "ensemble-final"]
    pool = ens if ens else rows
    proposed = max(pool, key=lambda r: (r["val_f1_macro"]
                                        if not np.isnan(r["val_f1_macro"])
                                        else r["test_f1_macro"]))
    proposed_label = proposed["model"]
    print(f"\n  PROPUESTO (mejor ensamble por val): {proposed_label}")

    # --- Wilcoxon por bloque temporal (mes) vs cada baseline ---
    ref = C.load_index_ref()
    test_ref = ref[ref["split"] == "test"].reset_index(drop=True)
    assert len(test_ref) == len(yte), f"desalineado: {len(test_ref)} vs {len(yte)}"
    blocks = test_ref["MatchDate"].dt.to_period("M").astype(str).to_numpy()

    prop_scores, keys = per_block_f1(yte, preds[proposed_label], blocks)
    wtxt = ["PRUEBA DE WILCOXON (rangos con signo, alpha=0.05)",
            f"Propuesto: {proposed_label}",
            f"Unidad de pareo: f1_macro por bloque mensual en test (n={len(prop_scores)} bloques)",
            f"f1_macro medio del propuesto por bloque: {prop_scores.mean():.4f}", ""]
    for base_label in ("Dummy", "GaussianNB"):
        if base_label not in preds:
            continue
        base_scores, _ = per_block_f1(yte, preds[base_label], blocks)
        diff = prop_scores - base_scores
        try:
            stat, p = wilcoxon(prop_scores, base_scores, alternative="greater")
        except ValueError as e:
            stat, p = np.nan, np.nan
            wtxt.append(f"  (wilcoxon {base_label}: {e})")
        signif = "SÍ (significativo)" if (p is not None and p < 0.05) else "no"
        wtxt.append(f"vs {base_label:12}: media_propuesto={prop_scores.mean():.4f} "
                    f"media_base={base_scores.mean():.4f}  Δ={diff.mean():+.4f}  "
                    f"W={stat:.1f}  p={p:.2e}  -> supera: {signif}")
    C.save_report("11_wilcoxon", "\n".join(wtxt))

    # --- reporte comparativo en texto ---
    lines = ["COMPARATIVA EN TEST (ordenado por f1_macro):", "",
             tbl.to_string(index=False), "",
             f"Propuesto declarado: {proposed_label}",
             f"Mejor f1_macro en test: {tbl.iloc[0]['model']} ({tbl.iloc[0]['test_f1_macro']})"]
    C.save_report("11_test_comparison", "\n".join(lines))

    # --- figuras del propuesto en test ---
    prop_clf = next(clf for f, l, fam, clf in models if l == proposed_label)
    pred_p, proba_p = safe_pred(prop_clf, d["X_test"])
    C.plot_confusion(yte, pred_p, "11_proposed_test",
                     f"{proposed_label} — test (confusión)")
    C.plot_roc_ovr(yte, proba_p, "11_proposed_test",
                   f"{proposed_label} — test ROC OvR")

    # barras comparativas f1_macro (val vs test)
    import matplotlib.pyplot as plt

    t2 = tbl.iloc[::-1]
    y = np.arange(len(t2))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(y + 0.2, t2["test_f1_macro"], 0.4, label="test", color="tab:blue")
    ax.barh(y - 0.2, t2["val_f1_macro"], 0.4, label="val", color="tab:orange")
    ax.set_yticks(y, t2["model"])
    ax.axvline(0.42, color="grey", ls="--", lw=0.8, label="ancla LogReg (0.42)")
    ax.set_xlabel("f1_macro")
    ax.set_title("Comparativa de modelos (val vs test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "11_model_comparison.png", dpi=130)
    plt.close(fig)
    print("  -> figures/modeling/11_model_comparison.png")
    print("\nTABLA FINAL:\n", tbl.to_string(index=False))


if __name__ == "__main__":
    main()
