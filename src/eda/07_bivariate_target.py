"""EDA 07 - Analisis bivariado feature vs resultado.

Muestra que las variables pre-partido (Elo, elo_diff, forma, cuotas) separan
las clases H/D/A: medias por clase y boxplots agrupados. Es la evidencia de
que existe senal predictiva ANTES del partido (justifica el modelado).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _common import TARGET_ORDER, ensure_figures_dir, load_matches, new_report

sns.set_theme(style="whitegrid")
PALETTE = {"H": "#2ca02c", "D": "#7f7f7f", "A": "#d62728"}


def main() -> None:
    log = new_report("07_bivariate_target.txt")
    figdir = ensure_figures_dir()
    m = load_matches()
    m["elo_diff"] = m["HomeElo"] - m["AwayElo"]
    m["form5_diff"] = m["Form5Home"] - m["Form5Away"]
    m = m[m["FTResult"].notna()].copy()

    feats = ["HomeElo", "AwayElo", "elo_diff", "Form5Home", "Form5Away",
             "form5_diff", "OddHome", "OddDraw", "OddAway"]

    log("=" * 70)
    log("EDA 07 - FEATURES vs RESULTADO (FTResult)")
    log("=" * 70)
    log("MEDIA de cada feature por clase (H=local, D=empate, A=visita):")
    grp = m.groupby("FTResult")[feats].mean().reindex(TARGET_ORDER).round(2)
    log(grp.to_string())

    log("")
    log("MEDIANA de cada feature por clase:")
    grpm = m.groupby("FTResult")[feats].median().reindex(TARGET_ORDER).round(2)
    log(grpm.to_string())

    # Separacion monotona esperada en elo_diff/form_diff: H > D > A.
    log("")
    log("-" * 70)
    log("SEPARACION DE CLASES (senal pre-partido)")
    log("-" * 70)
    for f in ["elo_diff", "form5_diff", "OddHome"]:
        h, d, a = grp.loc["H", f], grp.loc["D", f], grp.loc["A", f]
        log(f"{f:<11}: H={h:+.2f}  D={d:+.2f}  A={a:+.2f}")
    log("")
    log("Lectura: elo_diff y form5_diff decrecen monotonamente H > D > A: cuando el")
    log("local es mas fuerte, gana; cuando el visitante lo es, gana la visita. El empate")
    log("(D) queda en el MEDIO con diferencias ~0 -> por eso es la clase mas dificil de")
    log("separar. OddHome sube de H a A (mas cuota al local = menos probable que gane).")

    # --- Figura 1: Elo y elo_diff por resultado ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, f in zip(axes, ["HomeElo", "AwayElo", "elo_diff"]):
        sns.boxplot(data=m, x="FTResult", y=f, order=TARGET_ORDER, hue="FTResult",
                    palette=PALETTE, legend=False, ax=ax, showfliers=False)
        ax.set_title(f"{f} por resultado"); ax.set_xlabel("FTResult")
    fig.suptitle("Elo pre-partido segun el resultado final")
    fig.tight_layout(); fig.savefig(figdir / "elo_by_result.png", dpi=150); plt.close(fig)

    # --- Figura 2: forma por resultado ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, f in zip(axes, ["Form5Home", "Form5Away", "form5_diff"]):
        sns.boxplot(data=m, x="FTResult", y=f, order=TARGET_ORDER, hue="FTResult",
                    palette=PALETTE, legend=False, ax=ax, showfliers=False)
        ax.set_title(f"{f} por resultado"); ax.set_xlabel("FTResult")
    fig.suptitle("Forma reciente (puntos ult. 5) segun el resultado final")
    fig.tight_layout(); fig.savefig(figdir / "form_by_result.png", dpi=150); plt.close(fig)

    log(f"\n[figuras] elo_by_result.png, form_by_result.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
