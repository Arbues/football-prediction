"""EDA 08 - Decision de columnas a descartar (cierre del EDA).

Consolida los hallazgos de 00-07 en una tabla accionable: que columnas van al
modelo y cuales se descartan, con el motivo (data leakage post-partido, exceso
de nulos, redundancia). Es el puente hacia la Fase 2 (preprocesamiento).
"""

from __future__ import annotations

import pandas as pd

from _common import (
    CLOSING_COLS, LEAKAGE_POST, ODDS_COLS, SAFE_PRE_NUMERIC, TARGET,
    load_matches, new_report,
)


def main() -> None:
    log = new_report("08_drop_decisions.txt")
    m = load_matches()
    n = len(m)
    pct_null = (100 * m.isna().mean()).round(1)

    # Motivo de descarte por columna (prioridad: leakage > nulos > redundancia).
    decision: dict[str, tuple[str, str]] = {}

    for c in LEAKAGE_POST:
        decision[c] = ("DESCARTAR", "data leakage: estadistica POST-partido")

    for c in CLOSING_COLS:
        decision[c] = ("DESCARTAR", f"cierre bookmaker poco documentado, {pct_null[c]:.0f}% nulo")

    decision["MatchTime"] = ("DESCARTAR", f"{pct_null['MatchTime']:.0f}% nulo y sin valor predictivo")

    # Redundancia por construccion: elo_diff = HomeElo - AwayElo.
    decision["_nota_redundancia"] = (
        "VIGILAR",
        "elo_diff/form_diff son combinacion lineal de sus componentes: "
        "usar los diffs O los pares, no ambos en modelos lineales",
    )

    # Lo que se conserva.
    keep_pre = SAFE_PRE_NUMERIC + ["Division", "MatchDate", "HomeTeam", "AwayTeam"]
    keep_odds = ODDS_COLS

    log("=" * 70)
    log("EDA 08 - DECISION DE COLUMNAS (cierre de Fase 1)")
    log("=" * 70)
    log(f"Dataset: {n:,} filas x {m.shape[1]} columnas.")
    log("")

    log("-" * 70)
    log("(1) CONSERVAR - features pre-partido seguras (input del modelo)")
    log("-" * 70)
    for c in keep_pre:
        log(f"  KEEP   {c:<12} nulos {pct_null.get(c, 0):>5.1f}%")

    log("")
    log("-" * 70)
    log("(2) CONSERVAR CONDICIONAL - cuotas de mercado")
    log("-" * 70)
    log("  Modelar CON y SIN cuotas: son 'leak del sabio' (correlacion ~0.89 con")
    log("  elo_diff). La version SIN cuotas es la mas honesta; la version CON mide")
    log("  el techo. Limpiar antes: centinelas (OddHome==0, HandiSize==-99.9) y colas.")
    for c in keep_odds:
        log(f"  ODDS   {c:<12} nulos {pct_null.get(c, 0):>5.1f}%")

    log("")
    log("-" * 70)
    log("(3) TARGET")
    log("-" * 70)
    log(f"  {TARGET}: 3 filas nulas -> se descartan (sin etiqueta).")

    log("")
    log("-" * 70)
    log("(4) DESCARTAR - con motivo")
    log("-" * 70)
    drop_rows = [(c, v[1], pct_null.get(c, float('nan')))
                 for c, v in decision.items() if v[0] == "DESCARTAR"]
    log(f"{'columna':<13} {'%nulo':>7}  motivo")
    for c, motivo, pn in sorted(drop_rows, key=lambda t: -t[2]):
        log(f"{c:<13} {pn:>6.1f}%  {motivo}")
    log(f"\nTotal a descartar: {len(drop_rows)} columnas.")

    log("")
    log("-" * 70)
    log("(5) NULOS A IMPUTAR (columnas que se conservan)")
    log("-" * 70)
    log("  HomeElo/AwayElo: ~38.6% nulo, NO uniforme por anio (crece 2012-2024 por")
    log("    ligas menores anadidas). NO imputar con 0. Opciones: (a) filtrar a")
    log("    partidos con Elo, (b) imputar por mediana de liga/temporada, (c) usar el")
    log("    Elo del archivo EloRatings.csv con join temporal (date <= MatchDate).")
    log("  Form3/Form5: ~0.65% nulo (primeras jornadas) -> imputar 0 (sin historial).")
    log("  Odds: 1-36% nulo segun columna -> imputar mediana o descartar sub-bloques")
    log("    con >35% nulo (Over/Under, Handi) si se busca cobertura completa.")

    log("")
    log("-" * 70)
    log("RESUMEN NUMERICO")
    log("-" * 70)
    n_keep = len(keep_pre) + len(keep_odds) + 1
    n_drop = len(drop_rows)
    log(f"  Conservadas (incl. condicionales + target): {n_keep}")
    log(f"  Descartadas:                                {n_drop}")
    log(f"  Total:                                      {n_keep + n_drop} de {m.shape[1]}")

    log.save()


if __name__ == "__main__":
    main()
