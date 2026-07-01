"""EDA 05 - Correlacion lineal (Pearson) y monotona (Spearman).

Matriz de correlacion entre variables pre-partido seguras, derivadas y cuotas.
Dos heatmaps (Pearson y Spearman), deteccion de pares redundantes |r|>0.9 y
correlacion punto-biserial de cada feature con el resultado local (H vs no-H).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _common import ensure_figures_dir, load_matches, new_report

FEATURES = [
    "HomeElo", "AwayElo", "elo_diff",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "form3_diff", "form5_diff",
    "OddHome", "OddDraw", "OddAway",
]


def heatmap(corr: pd.DataFrame, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    log = new_report("05_correlation.txt")
    figdir = ensure_figures_dir()
    m = load_matches()
    m["elo_diff"] = m["HomeElo"] - m["AwayElo"]
    m["form3_diff"] = m["Form3Home"] - m["Form3Away"]
    m["form5_diff"] = m["Form5Home"] - m["Form5Away"]

    X = m[FEATURES]
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")

    log("=" * 70)
    log("EDA 05 - CORRELACION")
    log("=" * 70)
    log(f"Variables: {FEATURES}")
    log(f"Filas usadas (pairwise, con nulos por par): n max = {len(X):,}")
    log("")
    log("MATRIZ PEARSON (lineal):")
    log(pearson.round(3).to_string())
    log("")
    log("MATRIZ SPEARMAN (monotona, robusta a outliers):")
    log(spearman.round(3).to_string())

    # --- Pares redundantes (|r| alto) ---
    log("")
    log("-" * 70)
    log("PARES CON |Pearson| > 0.9 (candidatos a redundancia -> descartar uno)")
    log("-" * 70)
    found = False
    cols = pearson.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson.iloc[i, j]
            if abs(r) > 0.9:
                found = True
                log(f"  {cols[i]:<11} ~ {cols[j]:<11}  r = {r:+.3f}")
    if not found:
        log("  (ninguno > 0.9)")

    log("")
    log("PARES CON 0.7 < |Pearson| <= 0.9 (correlacion fuerte, vigilar):")
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson.iloc[i, j]
            if 0.7 < abs(r) <= 0.9:
                log(f"  {cols[i]:<11} ~ {cols[j]:<11}  r = {r:+.3f}")

    # --- Correlacion con el objetivo (punto-biserial: local gana = 1) ---
    log("")
    log("-" * 70)
    log("CORRELACION FEATURE vs RESULTADO (local gana H=1, resto=0)")
    log("-" * 70)
    y_home = (m["FTResult"] == "H").astype(float)
    rows = []
    for f in FEATURES:
        s = m[f]
        mask = s.notna()
        r = np.corrcoef(s[mask], y_home[mask])[0, 1]
        rows.append((f, r))
    for f, r in sorted(rows, key=lambda t: -abs(t[1])):
        log(f"  {f:<11} r = {r:+.3f}")
    log("")
    log("Nota: OddHome correlaciona negativo con H (cuota baja => local favorito).")
    log("elo_diff y form*_diff son las senales pre-partido mas informativas.")

    heatmap(pearson, "Matriz de correlacion de Pearson (features pre-partido)",
            figdir / "correlation_matrix.png")
    heatmap(spearman, "Matriz de correlacion de Spearman (features pre-partido)",
            figdir / "correlation_spearman.png")
    log(f"\n[figuras] correlation_matrix.png, correlation_spearman.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
