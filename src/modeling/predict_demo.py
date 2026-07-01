"""Demo de inferencia: predice los dieciseisavos del Mundial 2026 con los modelos.

ADVERTENCIA (fuera de distribución): los modelos se entrenaron con fútbol de CLUBES
(ligas 2000-2025), no con selecciones. Aquí se aplica el pipeline real de inferencia
(preprocessor.joblib + modelo) sobre vectores construidos a mano: todas las variables
se dejan SIMÉTRICAS salvo el Elo, de modo que la predicción la gobiernan `elo_diff`
(la variable dominante según SHAP) y el sesgo de localía aprendido. El "local" es solo
el primer equipo del emparejamiento (el estadio del Mundial es neutral, así que ese
sesgo es una distorsión conocida). En eliminatoria no hay empate real (penales). Los Elo
son aproximados (World Football Elo, orden de magnitud). Emparejamientos reales de los
dieciseisavos 2026 (fuentes: Wikipedia y Yahoo Sports, jul-2026). Es un demo del
pipeline, no un pronóstico confiable.
"""
import json
import unicodedata
import warnings

import joblib
import pandas as pd

import _common as C

warnings.filterwarnings("ignore")


def _norm(s):
    """Normaliza para comparar nombres sin tildes (Canadá == Canada)."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip().lower()

# Elo aproximado de selecciones (World Football Elo, mediados 2026)
ELO = {
    "Argentina": 2130, "España": 2120, "Francia": 2085, "Inglaterra": 2050,
    "Brasil": 2030, "Portugal": 2005, "Paises Bajos": 1990, "Alemania": 1955,
    "Belgica": 1945, "Croacia": 1920, "Colombia": 1900, "Marruecos": 1895,
    "Noruega": 1860, "Suiza": 1855, "Japon": 1845, "Senegal": 1840,
    "Ecuador": 1825, "Suecia": 1810, "Mexico": 1800, "Estados Unidos": 1795,
    "Austria": 1790, "Costa de Marfil": 1790, "Canada": 1785, "Argelia": 1770,
    "Australia": 1745, "Ghana": 1730, "Bosnia": 1720, "Paraguay": 1720,
    "Egipto": 1710, "RD Congo": 1690, "Sudafrica": 1680, "Cabo Verde": 1620,
}

# Dieciseisavos de final Mundial 2026. (local, visitante, resultado_real | None si pendiente)
FIXTURES = [
    ("Canada", "Sudafrica", "Canadá 1-0  → Canadá"),
    ("Brasil", "Japon", "Brasil 2-1  → Brasil"),
    ("Paraguay", "Alemania", "1-1, 4-3 pen  → Paraguay"),
    ("Marruecos", "Paises Bajos", "1-1, 3-2 pen  → Marruecos"),
    ("Noruega", "Costa de Marfil", "Noruega 2-1  → Noruega"),
    ("Francia", "Suecia", "Francia 3-0  → Francia"),
    ("Mexico", "Ecuador", "México 2-0  → México"),
    ("Inglaterra", "RD Congo", None),
    ("Belgica", "Senegal", None),
    ("Estados Unidos", "Bosnia", None),
    ("España", "Austria", None),
    ("Portugal", "Croacia", None),
    ("Suiza", "Argelia", None),
    ("Australia", "Egipto", None),
    ("Argentina", "Cabo Verde", None),
    ("Colombia", "Ghana", None),
]
MATCHES = [(h, a) for h, a, _ in FIXTURES]


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
    """Tabla de predicciones (proba en orden [A,D,H]) + comparación con lo real."""
    P = model.predict_proba(X)
    lines = [f"===== {name} =====",
             f"{'Partido':30} {'H':>6} {'D':>6} {'A':>6}  {'Pick modelo':14} {'Resultado real':22} {'':3}"]
    hits = total = 0
    for (h, a, real), p in zip(FIXTURES, P):
        pH, pD, pA = float(p[2]), float(p[1]), float(p[0])
        m = max(pH, pD, pA)
        pick = h if m == pH else (a if m == pA else "Empate")
        mark = ""
        if real is not None:
            total += 1
            ganador = real.split("→")[-1].strip()
            ok = (_norm(pick) == _norm(ganador)) or (pick == "Empate" and "pen" in real)
            hits += int(ok)
            mark = "OK" if ok else "x"
        lines.append(f"{h + ' vs ' + a:30} {pH:5.0%} {pD:5.0%} {pA:5.0%}  {pick:14} "
                     f"{(real or 'pendiente'):22} {mark:3}")
    if total:
        lines.append(f"\nAciertos sobre los {total} ya jugados: {hits}/{total}")
    return "\n".join(lines)


def main():
    C.set_seeds()
    pre = joblib.load(C.MODELS / "preprocessor.joblib")
    feats = json.load(open(C.PROC / "feature_names.json"))
    X = build_matrix(pre, feats)

    out = [__doc__.strip(), ""]
    for fname, label in [("06_xgboost", "XGBoost (campeón)"),
                         ("10_stacking", "Stacking (propuesto)")]:
        if (C.MODELS / f"{fname}.pkl").exists():
            out.append(predict_table(joblib.load(C.MODELS / f"{fname}.pkl"), label, X))
            out.append("")
    text = "\n".join(out)
    print(text)
    C.save_report("predict_demo", text)


if __name__ == "__main__":
    main()
