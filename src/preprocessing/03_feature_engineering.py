"""Preprocesamiento 03 - Ingenieria de features (estrictamente con el pasado).

Todas las variables derivadas se calculan usando UNICAMENTE partidos anteriores
a cada fila (orden temporal por match_id). Ninguna usa informacion del propio
partido ni de partidos futuros: es la garantia anti-leakage a nivel de fila
(distinta de la disciplina train-only, que aplica a estadisticos y va en 05).

Features construidas:
  - home/away_gf5, home/away_ga5 : media de goles a favor / en contra en los
    ultimos 5 partidos del equipo (cualquier sede), previos al actual.
  - home/away_rest_days          : dias desde el ultimo partido del equipo.
  - home/away_win_streak         : victorias consecutivas ANTES del partido.
  - h2h_played                   : enfrentamientos directos previos.
  - h2h_home_winrate             : tasa con que el local actual vencio a este
                                   rival en el pasado (cualquier sede).
  - h2h_avg_goals                : media de goles totales en enfrentamientos previos.
  - is_top_league                : 1 si la liga es top-5 europea.

El marcador FTHome/FTAway es la materia prima de gf/ga/streak/h2h y se ELIMINA
al final (ya cumplio su rol; como input directo seria fuga).

Salida: data/processed/_matches_features.parquet + results/14_features.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    FEATURES_PARQUET, ELO_PARQUET, ROLL_WINDOW, TOP5_LEAGUES, new_report,
)


def _win_streak_into(won_bool: pd.Series) -> pd.Series:
    """Victorias consecutivas ESTRICTAMENTE antes de cada fila (por equipo)."""
    won = won_bool.astype(int).to_numpy()
    out = np.zeros(len(won), dtype=float)
    run = 0
    for i in range(len(won)):
        out[i] = run
        run = run + 1 if won[i] == 1 else 0
    return pd.Series(out, index=won_bool.index)


def _team_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling de goles, descanso y racha por equipo en formato largo.

    Cada partido genera dos filas (lado local y visita). Se ordena por equipo y
    orden temporal (match_id) y se aplica shift(1) para excluir el partido en
    curso de toda ventana movil.
    """
    home = pd.DataFrame({
        "match_id": df["match_id"], "team": df["HomeTeam"], "venue": "H",
        "date": df["MatchDate"], "gf": df["FTHome"], "ga": df["FTAway"],
        "won": df["FTResult"].eq("H"),
    })
    away = pd.DataFrame({
        "match_id": df["match_id"], "team": df["AwayTeam"], "venue": "A",
        "date": df["MatchDate"], "gf": df["FTAway"], "ga": df["FTHome"],
        "won": df["FTResult"].eq("A"),
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "match_id"], kind="mergesort")

    g = long.groupby("team", sort=False)
    w = ROLL_WINDOW
    long["gf5"] = g["gf"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    long["ga5"] = g["ga"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    long["rest_days"] = g["date"].transform(lambda s: s.diff().dt.days)
    long["win_streak"] = g["won"].transform(_win_streak_into)
    return long


def _h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head con solo enfrentamientos previos (recorrido cronologico)."""
    played = np.zeros(len(df))
    winrate = np.full(len(df), np.nan)
    avg_goals = np.full(len(df), np.nan)
    hist: dict[frozenset, list[tuple[str, int]]] = {}

    it = df[["HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult"]].itertuples(index=False)
    for i, (home, away, fh, fa, res) in enumerate(it):
        key = frozenset((home, away))
        prev = hist.get(key)
        if prev:
            played[i] = len(prev)
            avg_goals[i] = sum(tg for _, tg in prev) / len(prev)
            winrate[i] = sum(1 for wnr, _ in prev if wnr == home) / len(prev)
        winner = home if res == "H" else (away if res == "A" else None)
        hist.setdefault(key, []).append((winner, int(fh + fa)))

    return pd.DataFrame({
        "h2h_played": played,
        "h2h_home_winrate": winrate,
        "h2h_avg_goals": avg_goals,
    }, index=df.index)


def main() -> None:
    rep = new_report("14_features.txt")
    rep("=" * 70)
    rep("PREPROC 03 - INGENIERIA DE FEATURES (SOLO PASADO)")
    rep("=" * 70)

    df = pd.read_parquet(ELO_PARQUET)
    rep(f"Filas: {len(df):,}  | ventana rolling = {ROLL_WINDOW}")

    # (1) Rolling por equipo (formato largo -> pivot a home/away).
    long = _team_rolling(df)
    piv = long.pivot_table(
        index="match_id", columns="venue",
        values=["gf5", "ga5", "rest_days", "win_streak"],
    )
    piv.columns = [f"{'home' if v == 'H' else 'away'}_{stat}" for stat, v in piv.columns]
    df = df.merge(piv, on="match_id", how="left")
    rep("\n(1) Rolling goles/descanso/racha por equipo: OK")

    # (2) Head-to-head previo.
    h2h = _h2h(df)
    df = pd.concat([df.reset_index(drop=True), h2h.reset_index(drop=True)], axis=1)
    rep("(2) Head-to-head (enfrentamientos previos): OK")

    # (3) Flag de liga top-5.
    df["is_top_league"] = df["Division"].isin(TOP5_LEAGUES).astype(int)
    rep(f"(3) is_top_league: {int(df['is_top_league'].sum()):,} filas en top-5")

    # (4) Se elimina el marcador (ya cumplio su rol como insumo historico).
    df = df.drop(columns=["FTHome", "FTAway"])

    new_feats = [c for c in df.columns if any(
        k in c for k in ("gf5", "ga5", "rest_days", "win_streak", "h2h", "is_top")
    )]
    rep(f"\n(4) Features nuevas ({len(new_feats)}): {new_feats}")
    rep("\nRESUMEN DE NULOS EN FEATURES NUEVAS (se imputan en 05):")
    for c in new_feats:
        na = int(df[c].isna().sum())
        rep(f"  {c:<20} nulos {na:>7,} ({100*na/len(df):5.2f}%)  "
            f"media={df[c].mean():.3f}")

    df.to_parquet(FEATURES_PARQUET, index=False)
    rep(f"\n[guardado] {FEATURES_PARQUET}  ({df.shape[0]:,} x {df.shape[1]})")
    rep.save()


if __name__ == "__main__":
    main()
