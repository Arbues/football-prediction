"""EDA 06 - Deteccion de valores atipicos (univariados).

Regla del rango intercuartilico (IQR) para cada variable numerica relevante:
conteo de outliers, limites y deteccion de valores centinela / imposibles en
las columnas de cuotas. Figura con boxplots.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from _common import ensure_figures_dir, load_matches, new_report

sns.set_theme(style="whitegrid")

ELO_FORM = ["HomeElo", "AwayElo", "elo_diff", "Form5Home", "Form5Away"]
ODDS = ["OddHome", "OddDraw", "OddAway", "Over25", "Under25",
        "HandiSize", "HandiHome", "HandiAway", "MaxOver25", "MaxUnder25"]
GOALS = ["FTHome", "FTAway"]


def iqr_stats(s):
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = ((s < lo) | (s > hi)).sum()
    return q1, q3, iqr, lo, hi, out, len(s)


def main() -> None:
    log = new_report("06_outliers.txt")
    figdir = ensure_figures_dir()
    m = load_matches()
    m["elo_diff"] = m["HomeElo"] - m["AwayElo"]

    log("=" * 70)
    log("EDA 06 - OUTLIERS (regla IQR: fuera de [Q1-1.5IQR, Q3+1.5IQR])")
    log("=" * 70)

    def block(title, cols):
        log(f"\n[{title}]")
        log(f"{'columna':<12} {'Q1':>9} {'Q3':>9} {'lim_inf':>10} {'lim_sup':>10} "
            f"{'#outliers':>10} {'%out':>7}")
        for c in cols:
            q1, q3, iqr, lo, hi, out, n = iqr_stats(m[c])
            log(f"{c:<12} {q1:>9.2f} {q3:>9.2f} {lo:>10.2f} {hi:>10.2f} "
                f"{out:>10,} {100*out/n:>6.2f}%")

    block("Elo y forma", ELO_FORM)
    block("Cuotas de mercado", ODDS)
    block("Goles (contexto)", GOALS)

    # --- Valores centinela / imposibles en cuotas ---
    log("")
    log("-" * 70)
    log("VALORES SOSPECHOSOS (centinela / imposibles en cuotas)")
    log("-" * 70)
    log(f"HandiSize == -99.90 (centinela de faltante): "
        f"{int((m['HandiSize'] == -99.90).sum()):,} filas")
    log(f"OddHome == 0 (cuota imposible): {int((m['OddHome'] == 0).sum()):,} filas")
    log(f"MaxAway == 0 (cuota imposible): {int((m['MaxAway'] == 0).sum()):,} filas")
    for c in ["MaxOver25", "MaxUnder25", "HandiHome", "MaxAway"]:
        log(f"{c}: max = {m[c].max():.1f}  (colas extremas -> errores de captura)")
    log("Lectura: una cuota no puede ser 0 ni 172; son errores/centinelas a limpiar")
    log("o winsorizar en preprocesamiento. Elo/Form NO tienen valores imposibles;")
    log("sus 'outliers' IQR son extremos legitimos (equipos muy fuertes/debiles).")

    # --- Figuras: boxplots normalizados por z-score para verlos juntos ---
    def boxfig(cols, title, fname):
        data = m[cols].apply(lambda s: (s - s.mean()) / s.std())
        fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 5))
        sns.boxplot(data=data, ax=ax, showfliers=True, fliersize=1)
        ax.set_title(title + " (estandarizado z para comparar escalas)")
        ax.set_ylabel("z-score"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(figdir / fname, dpi=150); plt.close(fig)

    boxfig(ELO_FORM + GOALS, "Boxplots: Elo, forma y goles", "boxplots_outliers.png")
    boxfig(["OddHome", "OddDraw", "OddAway", "Over25", "Under25"],
           "Boxplots: cuotas de mercado", "boxplots_odds.png")
    log(f"\n[figuras] boxplots_outliers.png, boxplots_odds.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
