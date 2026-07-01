"""Preprocesamiento 02 - Recuperacion de Elo por join temporal (anti-leakage).

El Elo falta en ~38.6% de las filas de forma ESTRUCTURAL por liga (ligas
menores sin cobertura). Antes de imputar, se recupera el Elo REAL disponible en
EloRatings.csv con un merge_asof "hacia atras": para cada partido se toma el
ultimo rating del club con fecha <= MatchDate (mismo patron date<=match_date del
kernel hack-hack). Esto NO es fuga: usa solo informacion anterior al partido.

Solo se RELLENAN nulos; los valores de Elo ya presentes en Matches se respetan.
Lo que siga nulo tras el join se resolvera con imputacion por mediana de liga
(train-only) + flag en el paso 05.

Salida: data/processed/_matches_elo.parquet + results/13_elo_join.txt
"""

from __future__ import annotations

import pandas as pd

from _common import (
    CLUB_ALIASES, ELO_PARQUET, CLEAN_PARQUET, load_elo, new_report,
    normalize_club,
)


def _norm_side(names: pd.Series) -> pd.Series:
    """Aplica alias conocidos y luego normaliza (strip/espacios)."""
    return names.map(lambda x: CLUB_ALIASES.get(x, x)).map(normalize_club)


def _asof_elo(matches: pd.DataFrame, elo: pd.DataFrame, club_col: str) -> pd.Series:
    """Ultimo Elo del club con date <= MatchDate (merge_asof backward)."""
    left = (
        matches[["match_id", "MatchDate", club_col]]
        .rename(columns={club_col: "club_norm"})
        .sort_values("MatchDate", kind="mergesort")
    )
    right = elo.sort_values("date", kind="mergesort")
    merged = pd.merge_asof(
        left, right,
        left_on="MatchDate", right_on="date",
        by="club_norm", direction="backward",
    )
    return merged.set_index("match_id")["elo"].reindex(matches["match_id"]).to_numpy()


def main() -> None:
    rep = new_report("13_elo_join.txt")
    rep("=" * 70)
    rep("PREPROC 02 - RECUPERACION DE ELO POR JOIN TEMPORAL (date <= MatchDate)")
    rep("=" * 70)

    df = pd.read_parquet(CLEAN_PARQUET)
    elo = load_elo()
    elo["club_norm"] = _norm_side(elo["club"])
    elo = elo[["club_norm", "date", "elo"]].dropna(subset=["club_norm"])

    df["home_norm"] = _norm_side(df["HomeTeam"])
    df["away_norm"] = _norm_side(df["AwayTeam"])

    n = len(df)
    home_na0 = int(df["HomeElo"].isna().sum())
    away_na0 = int(df["AwayElo"].isna().sum())
    rep(f"Filas: {n:,}")
    rep(f"EloRatings: {len(elo):,} registros, {elo['club_norm'].nunique():,} clubes normalizados")
    rep(f"Nulos ANTES  -> HomeElo {home_na0:,} ({100*home_na0/n:.1f}%) | "
        f"AwayElo {away_na0:,} ({100*away_na0/n:.1f}%)")

    # Cobertura de nombres tras armonizacion.
    elo_clubs = set(elo["club_norm"])
    match_clubs = set(df["home_norm"]) | set(df["away_norm"])
    inter = match_clubs & elo_clubs
    rep(f"\nCobertura de nombres tras alias+normalizacion: "
        f"{len(inter):,}/{len(match_clubs):,} equipos ({100*len(inter)/len(match_clubs):.1f}%)")

    # Join temporal por lado.
    home_join = _asof_elo(df, elo, "home_norm")
    away_join = _asof_elo(df, elo, "away_norm")

    df["HomeElo"] = df["HomeElo"].fillna(pd.Series(home_join, index=df.index))
    df["AwayElo"] = df["AwayElo"].fillna(pd.Series(away_join, index=df.index))

    home_na1 = int(df["HomeElo"].isna().sum())
    away_na1 = int(df["AwayElo"].isna().sum())
    rep(f"\nNulos DESPUES -> HomeElo {home_na1:,} ({100*home_na1/n:.1f}%) | "
        f"AwayElo {away_na1:,} ({100*away_na1/n:.1f}%)")
    rep(f"Elo REAL recuperado por join -> Home {home_na0 - home_na1:,} | "
        f"Away {away_na0 - away_na1:,}")

    # Diagnostico de lo que queda nulo: equipo ausente de EloRatings?
    rem = df[df["HomeElo"].isna()]
    absent = (~rem["home_norm"].isin(elo_clubs)).sum()
    rep(f"\nDe los {home_na1:,} Home nulos restantes, {absent:,} "
        f"({100*absent/max(home_na1,1):.1f}%) son equipos AUSENTES de EloRatings")
    rep("  -> ese nulo es irrecuperable por join; se imputara por mediana de")
    rep("     liga (train-only) + flag elo_missing en el paso 05.")

    # Ligas con mas nulo residual (confirma que es estructural por liga).
    rep("\nNULO RESIDUAL DE HomeElo POR LIGA (top 12):")
    by_div = df.groupby("Division")["HomeElo"].apply(lambda s: 100 * s.isna().mean())
    for div, pct in by_div.sort_values(ascending=False).head(12).items():
        rep(f"  {div:<5} {pct:5.1f}% nulo")

    df = df.drop(columns=["home_norm", "away_norm"])
    df.to_parquet(ELO_PARQUET, index=False)
    rep(f"\n[guardado] {ELO_PARQUET}  ({df.shape[0]:,} x {df.shape[1]})")
    rep.save()


if __name__ == "__main__":
    main()
