"""Experimento controlado de desbalance sobre el campeón (XGBoost).

Compara, con la MISMA configuración de modelo, tres estrategias frente al empate:
  (a) sin balanceo, (b) sample_weight='balanced' (reescala la pérdida por
  frecuencia inversa), (c) SMOTE solo-en-train (interpola vecinos kNN de D).
Se mide f1_macro y, sobre todo, recall del empate D en val. SMOTE se aplica
EXCLUSIVAMENTE al train tras el split (aplicarlo a val/test sería fuga).
Nota: SMOTE interpola también sobre columnas one-hot -> valores fraccionarios;
es la limitación conocida del método, se reporta.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import _common as C

# Configuración representativa y FIJA (no se re-busca; el objeto de estudio es el balanceo)
XGB_CFG = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
               tree_method="hist", device="cuda", n_estimators=400, max_depth=6,
               learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
               reg_lambda=1.0, random_state=C.SEED, n_jobs=-1)


def evaluate(clf, d, tag):
    proba = clf.predict_proba(d["X_val"])
    pred = clf.predict(d["X_val"])
    m = C.compute_metrics(d["y_val"], pred, proba)
    print(f"  [{tag:12}] f1_macro={m['f1_macro']:.4f}  recall_D={m['per_class']['D']['recall']:.4f}"
          f"  acc={m['accuracy']:.4f}")
    return m


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    C.header("Desbalance: sin balanceo vs class_weight vs SMOTE (XGBoost fijo)")
    rows = []

    # (a) sin balanceo
    clf = XGBClassifier(**XGB_CFG).fit(d["X_train"], d["y_train"])
    rows.append(("sin_balanceo", evaluate(clf, d, "sin_balanceo")))

    # (b) sample_weight balanceado
    sw = compute_sample_weight("balanced", d["y_train"])
    clf = XGBClassifier(**XGB_CFG).fit(d["X_train"], d["y_train"], sample_weight=sw)
    rows.append(("class_weight", evaluate(clf, d, "class_weight")))

    # (c) SMOTE solo en train
    sm = SMOTE(random_state=C.SEED, k_neighbors=5)
    Xtr_s, ytr_s = sm.fit_resample(d["X_train"], d["y_train"])
    print(f"  SMOTE: train {d['X_train'].shape[0]} -> {Xtr_s.shape[0]} "
          f"(clases: {np.bincount(ytr_s)})")
    clf = XGBClassifier(**XGB_CFG).fit(Xtr_s, ytr_s)
    rows.append(("SMOTE", evaluate(clf, d, "SMOTE")))

    # reporte + figura comparativa
    txt = ["EXPERIMENTO DE DESBALANCE — XGBoost con configuración fija:",
           f"  {XGB_CFG}", "",
           "Estrategia      f1_macro   recall_D   recall_H   recall_A   accuracy"]
    for tag, m in rows:
        pc = m["per_class"]
        txt.append(f"  {tag:12}  {m['f1_macro']:.4f}    {pc['D']['recall']:.4f}"
                   f"     {pc['H']['recall']:.4f}     {pc['A']['recall']:.4f}"
                   f"     {m['accuracy']:.4f}")
    txt += ["", "Lectura: el balanceo se juega en recall_D (empate) vs accuracy global.",
            "SMOTE interpola sobre one-hot (limitación conocida, reportada)."]
    C.save_report("09_imbalance", "\n".join(txt))

    import matplotlib.pyplot as plt
    tags = [r[0] for r in rows]
    f1s = [r[1]["f1_macro"] for r in rows]
    rec_d = [r[1]["per_class"]["D"]["recall"] for r in rows]
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar(x - 0.2, f1s, 0.4, label="f1_macro", color="tab:blue")
    ax.bar(x + 0.2, rec_d, 0.4, label="recall_D", color="tab:orange")
    ax.set_xticks(x, tags)
    ax.set_ylabel("score (val)")
    ax.set_title("Desbalance: efecto en f1_macro y recall del empate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "09_imbalance.png", dpi=130)
    plt.close(fig)
    print(f"  -> figures/modeling/09_imbalance.png")


if __name__ == "__main__":
    main()
