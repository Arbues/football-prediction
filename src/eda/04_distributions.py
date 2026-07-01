"""EDA 04 - Distribuciones de variables densas.

Histogramas/KDE de goles, Elo, diferencia de Elo y forma reciente. Cada
bloque produce una figura para la exposicion y reporta forma (skew/curtosis)
y normalidad aproximada.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from _common import ensure_figures_dir, load_matches, new_report

sns.set_theme(style="whitegrid")


def main() -> None:
    log = new_report("04_distributions.txt")
    figdir = ensure_figures_dir()
    m = load_matches()
    m["elo_diff"] = m["HomeElo"] - m["AwayElo"]

    log("=" * 70)
    log("EDA 04 - DISTRIBUCIONES")
    log("=" * 70)

    def report(name, s):
        s = s.dropna()
        log(f"{name:<12} n={len(s):>7,}  media={s.mean():7.2f}  std={s.std():7.2f}  "
            f"skew={s.skew():6.2f}  kurt={s.kurtosis():6.2f}")

    # --- Figura 1: goles local/visita ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, color in zip(axes, ["FTHome", "FTAway"], ["#2ca02c", "#d62728"]):
        vals = m[col].dropna()
        bins = np.arange(-0.5, vals.max() + 1.5, 1)
        ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(f"Goles {col}")
        ax.set_xlabel("goles"); ax.set_ylabel("frecuencia")
        report(col, m[col])
    fig.suptitle("Distribucion de goles (contexto; leakage como input)")
    fig.tight_layout()
    fig.savefig(figdir / "dist_goals.png", dpi=150); plt.close(fig)

    # --- Figura 2: Elo local y visita ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color in zip(["HomeElo", "AwayElo"], ["#1f77b4", "#ff7f0e"]):
        sns.histplot(m[col].dropna(), bins=60, kde=True, stat="density",
                     color=color, alpha=0.4, label=col, ax=ax)
        report(col, m[col])
    ax.set_title("Distribucion de Elo (Home vs Away)"); ax.set_xlabel("Elo")
    ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "dist_elo.png", dpi=150); plt.close(fig)

    # --- Figura 3: diferencia de Elo ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(m["elo_diff"].dropna(), bins=80, kde=True, color="#9467bd", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribucion de elo_diff = HomeElo - AwayElo (simetrica ~0)")
    ax.set_xlabel("elo_diff"); report("elo_diff", m["elo_diff"])
    fig.tight_layout(); fig.savefig(figdir / "dist_elo_diff.png", dpi=150); plt.close(fig)

    # --- Figura 4: forma reciente (Form3/Form5) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    form_cols = ["Form3Home", "Form3Away", "Form5Home", "Form5Away"]
    for ax, col in zip(axes.ravel(), form_cols):
        vals = m[col].dropna()
        bins = np.arange(-0.5, vals.max() + 1.5, 1)
        ax.hist(vals, bins=bins, color="#8c564b", edgecolor="black", alpha=0.8)
        ax.set_title(col); ax.set_xlabel("puntos ult. N"); ax.set_ylabel("frecuencia")
        report(col, m[col])
    fig.suptitle("Distribucion de la forma reciente (puntos acumulados)")
    fig.tight_layout(); fig.savefig(figdir / "dist_form.png", dpi=150); plt.close(fig)

    log("")
    log("-" * 70)
    log("LECTURAS")
    log("-" * 70)
    log("- Goles: asimetria positiva (cola larga a la derecha), tipico Poisson-like;")
    log("  moda en 1 gol local. FTHome > FTAway confirma ventaja de localia.")
    log("- Elo: aprox. normal, ligera asimetria positiva (skew ~0.43), centrado ~1533.")
    log("- elo_diff: simetrica y centrada en 0 (mean 0.01) -> Elo no sesga por sede.")
    log("- Form: discreta (0..9 y 0..15), casi uniforme con leve carga a valores medios;")
    log("  son puntos en ultimos 3/5 partidos, disponibles ANTES del partido.")
    log("")
    log(f"[figuras] dist_goals.png, dist_elo.png, dist_elo_diff.png, dist_form.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
