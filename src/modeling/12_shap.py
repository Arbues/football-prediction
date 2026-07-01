"""Interpretabilidad del campeón por SHAP (Fase 4, enganche con la evaluación).

Usa TreeExplainer (exacto y rápido para árboles) sobre XGBoost. Los valores de
Shapley reparten la predicción entre features respetando eficiencia y simetría
(teoría de juegos cooperativos). Se muestra la importancia global (media |SHAP|
agregada sobre H/D/A) y el beeswarm de la clase difícil (empate D), para ver qué
variables la empujan. Complementa la importancia por permutación del Random Forest.
"""
import joblib
import numpy as np

import _common as C


def main():
    C.set_seeds()
    import shap

    d = C.load_xy("raw")
    model = joblib.load(C.MODELS / "06_xgboost.pkl")
    names = [c.split("__")[-1] for c in d["X_test"].columns]

    rng = np.random.RandomState(C.SEED)
    idx = rng.choice(len(d["X_test"]), size=min(2500, len(d["X_test"])), replace=False)
    Xs = d["X_test"].iloc[idx]
    C.header(f"SHAP TreeExplainer sobre XGBoost (muestra n={len(Xs)} de test)")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    # normaliza a lista [array(n,feat) por clase]
    if isinstance(sv, list):
        sv_list = sv
    elif sv.ndim == 3:  # (n, feat, clases)
        sv_list = [sv[:, :, k] for k in range(sv.shape[2])]
    else:
        sv_list = [sv]

    # importancia global: media |SHAP| agregada sobre clases
    glob = np.mean([np.abs(s).mean(axis=0) for s in sv_list], axis=0)
    order = np.argsort(glob)[::-1]

    import matplotlib.pyplot as plt

    top = order[:15]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.barh(range(len(top))[::-1], glob[top], color="tab:purple")
    ax.set_yticks(range(len(top))[::-1], [names[i] for i in top])
    ax.set_xlabel("media |valor SHAP| (agregado H/D/A)")
    ax.set_title("XGBoost — importancia global por SHAP")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "shap_importance.png", dpi=130)
    plt.close(fig)

    # beeswarm de la clase D (empate) si está disponible
    k_D = int(C._LE.transform(["D"])[0])
    if len(sv_list) > k_D:
        plt.figure()
        shap.summary_plot(sv_list[k_D], Xs, feature_names=names, show=False,
                          max_display=15)
        plt.title("SHAP — clase D (empate)")
        plt.tight_layout()
        plt.savefig(C.FIGURES / "shap_summary_D.png", dpi=130, bbox_inches="tight")
        plt.close()

    txt = ["INTERPRETABILIDAD SHAP — campeón XGBoost (muestra de test)",
           f"TreeExplainer; n={len(Xs)}", "",
           "Importancia global (media |SHAP| agregada sobre H/D/A), top 15:"]
    for i in order[:15]:
        txt.append(f"  {names[i]:<22} {glob[i]:.4f}")
    txt += ["", "Figuras: figures/modeling/shap_importance.png, shap_summary_D.png"]
    C.save_report("12_shap", "\n".join(txt))
    print(f"  top: {[names[i] for i in order[:6]]}")


if __name__ == "__main__":
    main()
