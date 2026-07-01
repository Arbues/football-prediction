"""EDA 01 - Analisis de valores faltantes.

Tabla de % de nulos por columna (ordenada), agrupada por la taxonomia
anti-leakage, y figura de barras horizontal para la exposicion.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import (
    ensure_figures_dir, load_matches, new_report,
)
from importlib import import_module

# Reutiliza la clasificacion definida en 00 para no duplicarla.
classify_column = import_module("00_overview").classify_column

GROUP_COLORS = {
    "TARGET": "#2ca02c",
    "PRE_SEGURA": "#1f77b4",
    "ODDS_MERCADO": "#17becf",
    "CONTEXTO": "#7f7f7f",
    "CIERRE_BOOKMAKER": "#ff7f0e",
    "LEAKAGE_POST": "#d62728",
    "SIN_CLASIFICAR": "#000000",
}


def main() -> None:
    log = new_report("01_missingness.txt")
    matches = load_matches()
    n = len(matches)

    miss = matches.isna().sum()
    pct = (100 * miss / n).round(2)
    grupo = pd.Series({c: classify_column(c) for c in matches.columns})
    tabla = (
        pd.DataFrame({"n_nulos": miss, "pct_nulos": pct, "grupo": grupo})
        .sort_values("pct_nulos", ascending=False)
    )

    log("=" * 70)
    log("EDA 01 - VALORES FALTANTES")
    log("=" * 70)
    log(f"N = {n:,} filas   celdas totales = {matches.size:,}")
    log(f"Celdas nulas totales: {int(miss.sum()):,} "
        f"({100*miss.sum()/matches.size:.2f}% del dataset)")
    log("")
    log("% NULOS POR COLUMNA (desc):")
    log(f"{'columna':<13} {'n_nulos':>10} {'pct':>8}  grupo")
    for col, row in tabla.iterrows():
        log(f"{col:<13} {int(row['n_nulos']):>10,} {row['pct_nulos']:>7.2f}%  {row['grupo']}")

    # --- Cortes de decision ---
    log("")
    log("-" * 70)
    log("UMBRALES DE DECISION")
    log("-" * 70)
    gt50 = tabla[tabla["pct_nulos"] > 50].index.tolist()
    b20_50 = tabla[(tabla["pct_nulos"] > 20) & (tabla["pct_nulos"] <= 50)].index.tolist()
    lt5 = tabla[tabla["pct_nulos"] < 5].index.tolist()
    log(f">50% nulos (candidatas a descarte): {len(gt50)} -> {gt50}")
    log(f"20-50% nulos (imputacion cuidadosa): {len(b20_50)} -> {b20_50}")
    log(f"<5% nulos (imputacion trivial):     {len(lt5)} -> {lt5}")

    # Nulos del target (critico): filas sin FTResult son inservibles.
    log("")
    log(f"[target] filas con FTResult nulo: {int(matches['FTResult'].isna().sum())} "
        f"(se descartan: no hay etiqueta)")

    # % nulos de Elo por anio -> confirma que solo existe desde ~2006.
    log("")
    log("-" * 70)
    log("NULOS DE HomeElo POR ANIO (confirma disponibilidad temporal)")
    log("-" * 70)
    by_year = matches.assign(anio=matches["MatchDate"].dt.year)
    elo_year = by_year.groupby("anio")["HomeElo"].apply(lambda s: 100 * s.isna().mean())
    for anio, p in elo_year.items():
        bar = "#" * int(p / 3)
        log(f"  {anio}: {p:5.1f}% nulo  {bar}")

    # --- Figura: barras horizontales de % nulos por columna ---
    ensure_figures_dir()
    plot_df = tabla[tabla["pct_nulos"] > 0].iloc[::-1]  # asc para barh
    colors = [GROUP_COLORS[g] for g in plot_df["grupo"]]
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(plot_df.index, plot_df["pct_nulos"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(50, color="black", linestyle="--", linewidth=1, label="umbral 50%")
    ax.set_xlabel("% de valores faltantes")
    ax.set_title("Valores faltantes por columna (coloreado por grupo anti-leakage)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles + [ax.lines[0]], list(GROUP_COLORS.keys()) + ["umbral 50%"],
              loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = ensure_figures_dir() / "missingness_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"\n[figura] {out}")

    log.save()


if __name__ == "__main__":
    main()
