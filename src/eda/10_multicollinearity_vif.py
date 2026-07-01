"""EDA 10 - Multicolinealidad (VIF).

El Factor de Inflacion de la Varianza mide cuanto se infla la varianza de un
coeficiente por la correlacion de esa feature con las demas. Se calcula como el
elemento diagonal de la inversa de la matriz de correlacion: VIF_i = (R^-1)_ii.
Regla: VIF>10 grave, 5<VIF<=10 moderada. Complementa la matriz de correlacion (05).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ensure_figures_dir, load_matches, new_report

# Conjunto que veria un modelo lineal (SIN los diffs derivados, que son
# combinacion lineal exacta y darian VIF -> infinito).
MODEL_FEATURES = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "OddHome", "OddDraw", "OddAway",
]


def vif_from_corr(corr: pd.DataFrame) -> pd.Series:
    """VIF_i = i-esimo elemento diagonal de la inversa de la matriz de correlacion."""
    inv = np.linalg.inv(corr.values)
    return pd.Series(np.diag(inv), index=corr.columns)


def main() -> None:
    log = new_report("10_multicollinearity_vif.txt")
    figdir = ensure_figures_dir()
    m = load_matches()

    X = m[MODEL_FEATURES].dropna()
    corr = X.corr(method="pearson")
    vif = vif_from_corr(corr).sort_values(ascending=False)

    log("=" * 70)
    log("EDA 10 - MULTICOLINEALIDAD (VIF)")
    log("=" * 70)
    log(f"Features del modelo lineal ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
    log(f"Filas completas usadas: {len(X):,}")
    log("")
    log("VIF por feature (desc):")
    log(f"{'feature':<12} {'VIF':>8}  interpretacion")
    for f, v in vif.items():
        if v > 10:
            tag = "GRAVE (>10)"
        elif v > 5:
            tag = "moderada (5-10)"
        else:
            tag = "baja (<5)"
        log(f"{f:<12} {v:>8.2f}  {tag}")

    log("")
    log(f"VIF medio = {vif.mean():.2f}   VIF max = {vif.max():.2f} ({vif.idxmax()})")

    log("")
    log("-" * 70)
    log("REDUNDANCIA POR CONSTRUCCION (features derivadas)")
    log("-" * 70)
    log("elo_diff = HomeElo - AwayElo  y  form_diff = FormHome - FormAway son")
    log("combinaciones lineales EXACTAS: incluirlas junto a sus componentes vuelve")
    log("singular a R (VIF -> infinito). Regla para el modelado lineal: usar los")
    log("DIFFS *o* los componentes, nunca ambos. Los arboles (RF/XGB/LGBM) son")
    log("inmunes a la multicolinealidad, alli el problema no aplica.")

    log("")
    log("-" * 70)
    log("LECTURA")
    log("-" * 70)
    grave = vif[vif > 10].index.tolist()
    mod = vif[(vif > 5) & (vif <= 10)].index.tolist()
    log(f"Features con VIF grave (>10): {grave if grave else 'ninguna'}")
    log(f"Features con VIF moderado (5-10): {mod if mod else 'ninguna'}")
    log("Form3* y Form5* del mismo equipo estan correlacionadas (r~0.82); si un")
    log("modelo lineal sufre, conservar solo Form5 (ventana mas larga, mas informacion).")

    # --- Figura ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#d62728" if v > 10 else "#ff7f0e" if v > 5 else "#4c72b0" for v in vif.values]
    ax.barh(vif.index[::-1], vif.values[::-1], color=colors[::-1], edgecolor="black")
    ax.axvline(5, color="orange", linestyle="--", lw=1, label="moderada (5)")
    ax.axvline(10, color="red", linestyle="--", lw=1, label="grave (10)")
    ax.set_xlabel("VIF")
    ax.set_title("Factor de Inflacion de la Varianza (features del modelo lineal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "vif_barplot.png", dpi=150)
    plt.close(fig)
    log(f"\n[figura] vif_barplot.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
