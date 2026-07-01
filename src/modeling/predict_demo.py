"""Demo de inferencia: predice dieciseisavos de un Mundial con los modelos entrenados.

ADVERTENCIA (fuera de distribución): los modelos se entrenaron con fútbol de CLUBES
(ligas 2000-2025), no con selecciones. Aquí se aplica el pipeline real de inferencia
(preprocessor.joblib + modelo) sobre vectores construidos a mano: todas las variables
se dejan SIMÉTRICAS salvo el Elo, de modo que la predicción la gobiernan `elo_diff`
(la variable dominante según SHAP) y el sesgo de localía aprendido. El estadio de un
Mundial es neutral y en eliminatoria no hay empate real (penales), así que esto es un
demo del pipeline, no un pronóstico confiable. Los Elo son aproximados y el bracket es
ilustrativo (el real no está definido).
"""
import json
import warnings

import joblib
import pandas as pd

import _common as C

warnings.filterwarnings("ignore")

# Elo aproximado de selecciones (World Football Elo, orden de magnitud, mediados 2026)
ELO = {
    "Argentina": 2145, "Francia": 2100, "España": 2120, "Inglaterra": 2055,
    "Brasil": 2035, "Portugal": 2015, "Paises Bajos": 1995, "Belgica": 1955,
    "Alemania": 1965, "Italia": 1985, "Croacia": 1930, "Uruguay": 1920,
    "Colombia": 1905, "Marruecos": 1900, "Suiza": 1875, "Dinamarca": 1870,
    "Japon": 1850, "Senegal": 1850, "Ecuador": 1830, "Estados Unidos": 1800,
    "Mexico": 1800, "Corea": 1785, "Canada": 1785, "Australia": 1755, "Peru": 1750,
}

# Dieciseisavos ilustrativos (bracket real no definido)
MATCHES = [
    ("Argentina", "Australia"), ("España", "Japon"), ("Francia", "Senegal"),
    ("Brasil", "Mexico"), ("Inglaterra", "Ecuador"), ("Portugal", "Suiza"),
    ("Paises Bajos", "Estados Unidos"), ("Alemania", "Croacia"),
    ("Uruguay", "Colombia"), ("Peru", "Marruecos"),
]


def raw_row(home, away):
    """Fila cruda para el preprocessor; simétrica salvo el Elo."""
    he, ae = ELO[home], ELO[away]
    return {
        "match_id": 0, "Division": "WC", "MatchDate": pd.Timestamp("2026-07-05"),
        "HomeTeam": home, "AwayTeam": away, "FTResult": "H", "year": 2026, "split": "test",
        "HomeElo": he, "AwayElo": ae, "elo_diff": he - ae,
        "Form5Home": 7, "Form5Away": 7, "form5_diff": 0,
        "Form3Home": 4, "Form3Away": 4, "form3_diff": 0,
        "home_gf5": 1.4, "away_gf5": 1.4, "gf5_diff": 0,
        "home_ga5": 1.1, "away_ga5": 1.1, "ga5_diff": 0,
        "home_rest_days": 5, "away_rest_days": 5,
        "home_win_streak": 1, "away_win_streak": 1,
        "h2h_played": 0, "h2h_home_winrate": 0.5, "h2h_avg_goals": 2.5,
        "is_top_league": 0, "elo_missing": 0,
    }


def build_matrix(pre, feats):
    df = pd.DataFrame([raw_row(h, a) for h, a in MATCHES])[list(pre.feature_names_in_)]
    return pd.DataFrame(pre.transform(df), columns=feats)


def predict_table(model, name, X):
    """Devuelve texto con la tabla de predicciones (proba en orden [A,D,H])."""
    P = model.predict_proba(X)
    lines = [f"===== {name} =====",
             f"{'Partido':32} {'Local(H)':>9} {'Empate(D)':>10} {'Visita(A)':>10}   Predicción"]
    for (h, a), p in zip(MATCHES, P):
        pH, pD, pA = float(p[2]), float(p[1]), float(p[0])
        m = max(pH, pD, pA)
        pick = h if m == pH else (a if m == pA else "Empate")
        lines.append(f"{h + ' vs ' + a:32} {pH:8.1%} {pD:10.1%} {pA:10.1%}   -> {pick}")
    return "\n".join(lines)


def main():
    C.set_seeds()
    pre = joblib.load(C.MODELS / "preprocessor.joblib")
    feats = json.load(open(C.PROC / "feature_names.json"))
    X = build_matrix(pre, feats)

    out = [__doc__.strip(), ""]
    for fname, label in [("06_xgboost", "XGBoost (campeón)"),
                         ("10_stacking", "Stacking (propuesto)")]:
        path = C.MODELS / f"{fname}.pkl"
        if path.exists():
            out.append(predict_table(joblib.load(path), label, X))
            out.append("")
    text = "\n".join(out)
    print(text)
    C.save_report("predict_demo", text)


if __name__ == "__main__":
    main()
