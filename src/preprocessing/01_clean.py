"""Preprocesamiento 01 - Limpieza base y contrato de columnas.

Aplica el contrato de entrada cerrado en el EDA (results/08_drop_decisions.txt):
descarta las 17 columnas de fuga post-partido (salvo el marcador FTHome/FTAway,
que se conserva TEMPORALMENTE como materia prima para reconstruir features
historicas de partidos anteriores), las cuotas de mercado (decision de diseno:
version sin odds), los cierres de bookmaker y MatchTime. Elimina las 3 filas sin
etiqueta y ordena cronologicamente por MatchDate para todo lo que sigue.

Salida: data/processed/_matches_clean.parquet + results/12_clean.txt
"""

from __future__ import annotations

import pandas as pd

from _common import (
    CLOSING_COLS, CONTEXT_COLS, DROP_MISC, GOALS_COLS, LEAKAGE_POST, ODDS_COLS,
    SAFE_PRE_NUMERIC, TARGET, CLEAN_PARQUET, ensure_dirs, load_matches, new_report,
)


def main() -> None:
    ensure_dirs()
    rep = new_report("12_clean.txt")
    rep("=" * 70)
    rep("PREPROC 01 - LIMPIEZA BASE Y CONTRATO DE COLUMNAS")
    rep("=" * 70)

    df = load_matches()
    n0, d0 = df.shape
    rep(f"Dataset crudo: {n0:,} filas x {d0} columnas")

    # (1) Eliminar filas sin etiqueta (no hay target que aprender).
    n_null_target = int(df[TARGET].isna().sum())
    df = df[df[TARGET].notna()].copy()
    rep(f"\n(1) Filas sin FTResult descartadas: {n_null_target} -> quedan {len(df):,}")

    # (2) Contrato de columnas: que se conserva y por que.
    dropped = LEAKAGE_POST + ODDS_COLS + CLOSING_COLS + DROP_MISC
    keep = CONTEXT_COLS + SAFE_PRE_NUMERIC + GOALS_COLS + [TARGET]
    df = df[keep].copy()

    rep("\n(2) CONTRATO DE COLUMNAS")
    rep(f"  Conservadas ({len(keep)}): {keep}")
    rep(f"  Descartadas ({len(dropped)}):")
    rep(f"    - fuga post-partido (15): {LEAKAGE_POST}")
    rep(f"    - cuotas de mercado (13): {ODDS_COLS}")
    rep(f"    - cierres bookmaker (6): {CLOSING_COLS}")
    rep(f"    - sin valor (MatchTime): {DROP_MISC}")
    rep("  NOTA: FTHome/FTAway se conservan SOLO como insumo de features")
    rep("        historicas (03); se eliminan antes de la matriz final.")

    # (3) Orden cronologico estable: base de todo el feature engineering temporal
    # y del split. Desempate por indice original para reproducibilidad total.
    df = df.sort_values(
        ["MatchDate", "Division", "HomeTeam"], kind="mergesort"
    ).reset_index(drop=True)
    df.insert(0, "match_id", df.index.astype("int64"))
    rep(f"\n(3) Ordenado por MatchDate (desempate Division, HomeTeam).")
    rep(f"    Rango temporal: {df['MatchDate'].min().date()} -> {df['MatchDate'].max().date()}")

    # Reporte de nulos en lo que se conserva.
    rep("\n(4) NULOS EN COLUMNAS CONSERVADAS")
    na = df.isna().sum()
    for c in df.columns:
        if na[c] > 0:
            rep(f"  {c:<12} {na[c]:>8,}  ({100*na[c]/len(df):.2f}%)")

    df.to_parquet(CLEAN_PARQUET, index=False)
    rep(f"\n[guardado] {CLEAN_PARQUET}  ({df.shape[0]:,} x {df.shape[1]})")
    rep.save()


if __name__ == "__main__":
    main()
