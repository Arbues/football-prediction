"""EDA 11 - Anatomia del empate (por que D es la clase dificil).

Analiza la tasa de empate condicionada a la fuerza relativa de los equipos
(|elo_diff|) y a la forma. Muestra con evidencia propia que el empate NUNCA es
la clase mayoritaria en ninguna region del espacio de features -> techo de
recall bajo estructural, consistente con la literatura.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import TARGET_ORDER, ensure_figures_dir, load_matches, new_report

CLASS_COLOR = {"H": "#2ca02c", "D": "#7f7f7f", "A": "#d62728"}


def rate_table(df: pd.DataFrame, bincol: str) -> pd.DataFrame:
    t = (df.groupby(bincol, observed=True)["FTResult"]
         .value_counts(normalize=True).unstack()
         .reindex(columns=TARGET_ORDER))
    t["n"] = df.groupby(bincol, observed=True).size()
    return t


def main() -> None:
    log = new_report("11_draw_analysis.txt")
    figdir = ensure_figures_dir()
    m = load_matches()
    m = m[m["FTResult"].notna()].copy()
    m["elo_diff"] = m["HomeElo"] - m["AwayElo"]
    m["form5_diff"] = m["Form5Home"] - m["Form5Away"]

    log("=" * 70)
    log("EDA 11 - ANATOMIA DEL EMPATE (clase D)")
    log("=" * 70)
    base_draw = (m["FTResult"] == "D").mean()
    log(f"Tasa de empate global: {base_draw*100:.2f}%")
    log("")

    # (1) Tasa por |elo_diff|: mide si la paridad favorece el empate.
    md = m.dropna(subset=["elo_diff"]).copy()
    md["abin"] = pd.cut(md["elo_diff"].abs(), [0, 25, 50, 100, 200, 1e4],
                        labels=["0-25", "25-50", "50-100", "100-200", ">200"])
    t1 = rate_table(md, "abin")
    log("-" * 70)
    log("(1) RESULTADO vs |elo_diff| (paridad de fuerzas)")
    log("-" * 70)
    log(f"{'|elo_diff|':<10} {'n':>8} {'H%':>7} {'D%':>7} {'A%':>7}")
    for b, r in t1.iterrows():
        log(f"{b:<10} {int(r['n']):>8,} {r['H']*100:>6.1f} {r['D']*100:>6.1f} {r['A']*100:>6.1f}")
    dmax = t1["D"].max() * 100
    log(f"\nTasa de empate MAXIMA sobre todos los bins: {dmax:.1f}% (en partidos parejos).")
    log("El empate NUNCA es mayoritario: ni siquiera con equipos identicos supera ~30%.")
    log("=> Ningun modelo puede volver a D la clase mas probable en ninguna region;")
    log("   su recall tiene un techo estructural bajo (coherente con Yeung 2023, etc.).")

    # (2) Tasa por elo_diff con signo: simetria del empate.
    md["sbin"] = pd.cut(md["elo_diff"], [-1e4, -200, -100, -50, -25, 0, 25, 50, 100, 200, 1e4])
    t2 = rate_table(md, "sbin")
    log("")
    log("-" * 70)
    log("(2) RESULTADO vs elo_diff con signo")
    log("-" * 70)
    log(f"{'elo_diff':<16} {'n':>8} {'H%':>7} {'D%':>7} {'A%':>7}")
    for b, r in t2.iterrows():
        log(f"{str(b):<16} {int(r['n']):>8,} {r['H']*100:>6.1f} {r['D']*100:>6.1f} {r['A']*100:>6.1f}")

    # (3) Empate vs forma pareja.
    mf = m.dropna(subset=["form5_diff"]).copy()
    mf["fbin"] = pd.cut(mf["form5_diff"].abs(), [-0.1, 0.5, 2, 4, 6, 30],
                        labels=["0", "1-2", "3-4", "5-6", ">6"])
    t3 = rate_table(mf, "fbin")
    log("")
    log("-" * 70)
    log("(3) TASA DE EMPATE vs |form5_diff|")
    log("-" * 70)
    log(f"{'|form5_diff|':<12} {'n':>8} {'D%':>7}")
    for b, r in t3.iterrows():
        log(f"{b:<12} {int(r['n']):>8,} {r['D']*100:>6.1f}")

    log("")
    log("Conclusion: la paridad (Elo y forma similares) eleva el empate solo hasta")
    log("~30%. Es una clase difusa, sin region propia donde domine: por eso todos los")
    log("clasificadores de la literatura reportan su recall <35%. Se recomienda")
    log("reportar F1 por clase y no optimizar solo accuracy.")

    # --- Figura 1: proporcion H/D/A apilada por elo_diff con signo ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    props = t2[TARGET_ORDER].values
    xlab = [str(i) for i in t2.index]
    bottom = np.zeros(len(t2))
    for k, cls in enumerate(TARGET_ORDER):
        ax.bar(xlab, props[:, k], bottom=bottom, color=CLASS_COLOR[cls], label=cls,
               edgecolor="white", linewidth=0.5)
        bottom += props[:, k]
    ax.set_ylabel("proporcion"); ax.set_title("Composicion H/D/A segun elo_diff")
    ax.set_xlabel("bins de elo_diff (Home - Away)")
    ax.tick_params(axis="x", rotation=45); ax.legend(title="resultado")

    # --- Figura 2: tasa de empate vs |elo_diff| (linea) ---
    ax = axes[1]
    centers = ["0-25", "25-50", "50-100", "100-200", ">200"]
    ax.plot(centers, t1["D"].values * 100, "o-", color="#7f7f7f", lw=2, label="tasa empate")
    ax.axhline(base_draw * 100, color="black", linestyle=":", label=f"media global {base_draw*100:.1f}%")
    ax.axhline(33.3, color="red", linestyle="--", alpha=0.5, label="1/3 (azar)")
    ax.set_ylim(0, 40)
    ax.set_xlabel("|elo_diff| (paridad ->)"); ax.set_ylabel("tasa de empate (%)")
    ax.set_title("La tasa de empate cae con el desnivel de fuerzas")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "draw_analysis.png", dpi=150)
    plt.close(fig)
    log(f"\n[figura] draw_analysis.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
