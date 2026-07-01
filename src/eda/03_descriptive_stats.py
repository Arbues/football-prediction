"""EDA 03 - Estadistica descriptiva de variables numericas.

Media, mediana, std, percentiles, skew y curtosis para las variables
pre-partido seguras, las derivadas (elo_diff, form_diff), las cuotas y
los goles (estos ultimos como contexto, no como input). Sin figura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    ODDS_COLS, SAFE_PRE_NUMERIC, load_matches, new_report,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

GOALS = ["FTHome", "FTAway"]  # leakage como input; se describen como contexto


def describe_block(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    pct = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    desc = df[cols].describe(percentiles=pct).T
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    desc["pct_nulo"] = 100 * df[cols].isna().mean()
    return desc.round(3)


def main() -> None:
    log = new_report("03_descriptive_stats.txt")
    matches = load_matches()

    # Derivadas clave (las mismas que usara el modelado).
    matches["elo_diff"] = matches["HomeElo"] - matches["AwayElo"]
    matches["form3_diff"] = matches["Form3Home"] - matches["Form3Away"]
    matches["form5_diff"] = matches["Form5Home"] - matches["Form5Away"]
    derived = ["elo_diff", "form3_diff", "form5_diff"]

    log("=" * 70)
    log("EDA 03 - ESTADISTICA DESCRIPTIVA")
    log("=" * 70)

    log("\n[A] VARIABLES PRE-PARTIDO SEGURAS")
    log(describe_block(matches, SAFE_PRE_NUMERIC).to_string())

    log("\n[B] VARIABLES DERIVADAS (feature engineering previsto)")
    log(describe_block(matches, derived).to_string())

    log("\n[C] CUOTAS DE MERCADO (Odd*/Max*/Over-Under/Handi)")
    log(describe_block(matches, ODDS_COLS).to_string())

    log("\n[D] GOLES (CONTEXTO - son leakage como input al modelo)")
    log(describe_block(matches, GOALS).to_string())

    # Lecturas cuantitativas para la interpretacion.
    log("\n" + "-" * 70)
    log("LECTURAS")
    log("-" * 70)
    log(f"Rango de Elo (Home): [{matches['HomeElo'].min():.0f}, {matches['HomeElo'].max():.0f}]")
    log(f"elo_diff medio = {matches['elo_diff'].mean():.2f} "
        f"(mediana {matches['elo_diff'].median():.2f}); ~0 => Elo es SIMETRICO: no hay "
        f"sesgo de emparejamiento local/visita. La ventaja de localia es un efecto de")
    log("sede independiente del Elo (por eso H domina aunque elo_diff sea ~0 en promedio).")
    frac_home_higher = 100 * (matches["elo_diff"].dropna() > 0).mean()
    log(f"Entre partidos CON Elo (no nulo): Elo local > visita en {frac_home_higher:.1f}% "
        f"(~50%, confirma simetria).")
    log(f"Form (0..): Form5 rango [{matches['Form5Home'].min():.0f}, "
        f"{matches['Form5Home'].max():.0f}], Form3 rango "
        f"[{matches['Form3Home'].min():.0f}, {matches['Form3Home'].max():.0f}] "
        f"-> confirman ser puntos acumulados en ultimos N partidos (pre-partido).")
    log(f"OddHome medio {matches['OddHome'].mean():.2f} < OddDraw "
        f"{matches['OddDraw'].mean():.2f} < OddAway {matches['OddAway'].mean():.2f} "
        f"-> el mercado tambien favorece al local (coherente con el sesgo H).")
    log(f"Goles: FTHome medio {matches['FTHome'].mean():.2f} > FTAway "
        f"{matches['FTAway'].mean():.2f} (ventaja de localia en marcador).")

    log.save()


if __name__ == "__main__":
    main()
