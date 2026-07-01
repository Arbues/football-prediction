"""EDA 00 - Vision general del dataset.

Origen, tamano en disco y memoria, N/D, tipos, diccionario de variables y
clasificacion de cada columna segun la taxonomia anti-leakage. Sin figuras.
"""

from __future__ import annotations

import numpy as np

from _common import (
    CLOSING_COLS, CONTEXT_COLS, ELO_CSV, LEAKAGE_POST, MATCHES_CSV,
    ODDS_COLS, SAFE_PRE_NUMERIC, TARGET, load_elo, load_matches, new_report,
)


def classify_column(col: str) -> str:
    if col == TARGET:
        return "TARGET"
    if col in LEAKAGE_POST:
        return "LEAKAGE_POST"
    if col in SAFE_PRE_NUMERIC:
        return "PRE_SEGURA"
    if col in ODDS_COLS:
        return "ODDS_MERCADO"
    if col in CLOSING_COLS:
        return "CIERRE_BOOKMAKER"
    if col in CONTEXT_COLS:
        return "CONTEXTO"
    return "SIN_CLASIFICAR"


def main() -> None:
    log = new_report("00_overview.txt")

    matches = load_matches()
    elo = load_elo()

    log("=" * 70)
    log("EDA 00 - VISION GENERAL")
    log("=" * 70)
    log("Origen: Kaggle 'adamgbor/club-football-match-data-2000-2025' (publico)")
    log(f"Archivo Matches: {MATCHES_CSV.name}  ({MATCHES_CSV.stat().st_size/1e6:.2f} MB en disco)")
    log(f"Archivo Elo:     {ELO_CSV.name}  ({ELO_CSV.stat().st_size/1e6:.2f} MB en disco)")
    log("")

    n, d = matches.shape
    mem_mb = matches.memory_usage(deep=True).sum() / 1e6
    log(f"Matches:  N = {n:,} instancias   D = {d} caracteristicas")
    log(f"          memoria en RAM (deep): {mem_mb:.1f} MB")
    log(f"Elo:      {elo.shape[0]:,} filas x {elo.shape[1]} columnas")
    log("")
    log(f"Rango temporal MatchDate: {matches['MatchDate'].min().date()} -> {matches['MatchDate'].max().date()}")
    log(f"Divisiones (ligas) unicas: {matches['Division'].nunique()}")
    log(f"Equipos locales unicos:    {matches['HomeTeam'].nunique()}")
    log("")

    # --- Diccionario de variables + clasificacion anti-leakage ---
    log("-" * 70)
    log("DICCIONARIO DE VARIABLES (tipo, %no-nulo, #unicos, grupo)")
    log("-" * 70)
    header = f"{'#':>2}  {'columna':<13} {'dtype':<12} {'%no-nulo':>8} {'#unicos':>8}  grupo"
    log(header)
    grupos: dict[str, list[str]] = {}
    for i, col in enumerate(matches.columns):
        dtype = str(matches[col].dtype)
        pct_full = 100 * matches[col].notna().mean()
        nuniq = matches[col].nunique(dropna=True)
        grupo = classify_column(col)
        grupos.setdefault(grupo, []).append(col)
        log(f"{i:>2}  {col:<13} {dtype:<12} {pct_full:>7.1f}% {nuniq:>8,}  {grupo}")

    log("")
    log("-" * 70)
    log("RESUMEN POR GRUPO (taxonomia anti-leakage)")
    log("-" * 70)
    for grupo in ["TARGET", "PRE_SEGURA", "ODDS_MERCADO", "CONTEXTO",
                  "CIERRE_BOOKMAKER", "LEAKAGE_POST", "SIN_CLASIFICAR"]:
        cols = grupos.get(grupo, [])
        log(f"{grupo:<18} ({len(cols):>2}): {', '.join(cols) if cols else '-'}")

    # Chequeo de integridad: ninguna columna debe quedar SIN_CLASIFICAR.
    sin = grupos.get("SIN_CLASIFICAR", [])
    log("")
    log(f"[chequeo] columnas sin clasificar: {len(sin)} -> {sin if sin else 'ninguna (OK)'}")

    log("")
    log("primeras 3 filas (columnas de contexto + target):")
    log(matches[CONTEXT_COLS + SAFE_PRE_NUMERIC + [TARGET]].head(3).to_string())

    log.save()


if __name__ == "__main__":
    main()
