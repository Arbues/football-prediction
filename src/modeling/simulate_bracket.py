"""Simula el cuadro completo del Mundial 2026 (16avos -> final + 3er puesto) con el
modelo propuesto (Stacking) y genera una web (docs/bracket.html) con el resultado.

Fuera de distribución (modelos entrenados con clubes). Cada cruce se predice en las DOS
orientaciones y se promedia para cancelar el sesgo de localía (sede neutral): decide el
Elo, la variable dominante. Los 7 partidos ya jugados usan el resultado REAL; el resto,
la predicción del modelo. En eliminatoria no hay empate: avanza el equipo con mayor
probabilidad de ganar (la masa de empate se reparte).
"""
import json
import warnings

import joblib
import pandas as pd

import _common as C

warnings.filterwarnings("ignore")

ELO = {
    "ARG": 2130, "ESP": 2120, "FRA": 2085, "ENG": 2050, "BRA": 2030, "POR": 2005,
    "NED": 1990, "GER": 1955, "BEL": 1945, "CRO": 1920, "COL": 1900, "MAR": 1895,
    "NOR": 1860, "SUI": 1855, "JPN": 1845, "SEN": 1840, "ECU": 1825, "SWE": 1810,
    "MEX": 1800, "USA": 1795, "AUT": 1790, "CIV": 1790, "CAN": 1785, "DZA": 1770,
    "AUS": 1745, "GHA": 1730, "BIH": 1720, "PAR": 1720, "EGY": 1710, "COD": 1690,
    "RSA": 1680, "CPV": 1620,
}
NAME = {
    "GER": "Alemania", "PAR": "Paraguay", "FRA": "Francia", "SWE": "Suecia",
    "RSA": "Sudáfrica", "CAN": "Canadá", "NED": "P. Bajos", "MAR": "Marruecos",
    "POR": "Portugal", "CRO": "Croacia", "ESP": "España", "AUT": "Austria",
    "USA": "EE. UU.", "BIH": "Bosnia", "BEL": "Bélgica", "SEN": "Senegal",
    "BRA": "Brasil", "JPN": "Japón", "CIV": "C. Marfil", "NOR": "Noruega",
    "MEX": "México", "ECU": "Ecuador", "ENG": "Inglaterra", "COD": "R.D. Congo",
    "ARG": "Argentina", "CPV": "Cabo Verde", "AUS": "Australia", "EGY": "Egipto",
    "SUI": "Suiza", "DZA": "Argelia", "COL": "Colombia", "GHA": "Ghana",
}
ISO2 = {
    "GER": "DE", "PAR": "PY", "FRA": "FR", "SWE": "SE", "RSA": "ZA", "CAN": "CA",
    "NED": "NL", "MAR": "MA", "POR": "PT", "CRO": "HR", "ESP": "ES", "AUT": "AT",
    "USA": "US", "BIH": "BA", "BEL": "BE", "SEN": "SN", "BRA": "BR", "JPN": "JP",
    "CIV": "CI", "NOR": "NO", "MEX": "MX", "ECU": "EC", "COD": "CD", "ARG": "AR",
    "CPV": "CV", "AUS": "AU", "EGY": "EG", "SUI": "CH", "DZA": "DZ", "COL": "CO",
    "GHA": "GH",
}


def flag(code):
    if code == "ENG":
        return "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F"
    cc = ISO2.get(code)
    if not cc:
        return "\U0001F3F3"
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in cc)


# Resultados REALES de los 16avos ya jugados (marcador mostrado como en el bracket oficial)
REAL = {
    ("GER", "PAR"): ("PAR", "1 (3)", "1 (4)"),
    ("FRA", "SWE"): ("FRA", "3", "0"),
    ("RSA", "CAN"): ("CAN", "0", "1"),
    ("NED", "MAR"): ("MAR", "1 (2)", "1 (3)"),
    ("BRA", "JPN"): ("BRA", "2", "1"),
    ("CIV", "NOR"): ("NOR", "1", "2"),
    ("MEX", "ECU"): ("MEX", "2", "0"),
}

# 16avos por lado (orden del bracket, arriba->abajo)
LEFT_R32 = [("GER", "PAR"), ("FRA", "SWE"), ("RSA", "CAN"), ("NED", "MAR"),
            ("POR", "CRO"), ("ESP", "AUT"), ("USA", "BIH"), ("BEL", "SEN")]
RIGHT_R32 = [("BRA", "JPN"), ("CIV", "NOR"), ("MEX", "ECU"), ("ENG", "COD"),
             ("ARG", "CPV"), ("AUS", "EGY"), ("SUI", "DZA"), ("COL", "GHA")]

_PRE = joblib.load(C.MODELS / "preprocessor.joblib")
_FEATS = json.load(open(C.PROC / "feature_names.json"))
_MODEL = joblib.load(C.MODELS / "10_stacking.pkl")


def _row(eh, ea):
    return {"match_id": 0, "Division": "WC", "MatchDate": pd.Timestamp("2026-07-05"),
            "HomeTeam": "H", "AwayTeam": "A", "FTResult": "H", "year": 2026, "split": "test",
            "HomeElo": eh, "AwayElo": ea, "elo_diff": eh - ea,
            "Form5Home": 7, "Form5Away": 7, "form5_diff": 0, "Form3Home": 4,
            "Form3Away": 4, "form3_diff": 0, "home_gf5": 1.4, "away_gf5": 1.4,
            "gf5_diff": 0, "home_ga5": 1.1, "away_ga5": 1.1, "ga5_diff": 0,
            "home_rest_days": 5, "away_rest_days": 5, "home_win_streak": 1,
            "away_win_streak": 1, "h2h_played": 0, "h2h_home_winrate": 0.5,
            "h2h_avg_goals": 2.5, "is_top_league": 0, "elo_missing": 0}


def play(a, b, key=None):
    """Devuelve (winner, sa, sb, real) para el cruce a vs b."""
    k = key or (a, b)
    if k in REAL:
        w, sa, sb = REAL[k]
        return w, sa, sb, True
    # predicción neutral: promedio de las dos orientaciones
    df = pd.DataFrame([_row(ELO[a], ELO[b]), _row(ELO[b], ELO[a])])[list(_PRE.feature_names_in_)]
    X = pd.DataFrame(_PRE.transform(df), columns=_FEATS)
    P = _MODEL.predict_proba(X)  # columnas [A, D, H]
    pa = 0.5 * (P[0][2] + P[1][0])   # a gana
    pb = 0.5 * (P[0][0] + P[1][2])   # b gana
    w = a if pa >= pb else b
    return w, f"{round(100 * pa)}%", f"{round(100 * pb)}%", False


def round_of(pairs):
    """Juega una lista de cruces (tuplas de codigos) -> lista de dicts resultado."""
    res = []
    for a, b in pairs:
        w, sa, sb, real = play(a, b)
        res.append({"a": a, "b": b, "sa": sa, "sb": sb, "w": w, "real": real})
    return res


def winners(res):
    return [r["w"] for r in res]


def pairup(codes):
    return [(codes[i], codes[i + 1]) for i in range(0, len(codes), 2)]


def main():
    C.set_seeds()
    # LADO IZQUIERDO
    l16 = round_of(LEFT_R32)
    l8 = round_of(pairup(winners(l16)))
    l4 = round_of(pairup(winners(l8)))
    lsemi = round_of(pairup(winners(l4)))
    # LADO DERECHO
    r16 = round_of(RIGHT_R32)
    r8 = round_of(pairup(winners(r16)))
    r4 = round_of(pairup(winners(r8)))
    rsemi = round_of(pairup(winners(r4)))
    # FINAL y 3er puesto
    fin = round_of([(lsemi[0]["w"], rsemi[0]["w"])])
    loser = lambda res: res[0]["b"] if res[0]["w"] == res[0]["a"] else res[0]["a"]
    third = round_of([(loser(lsemi), loser(rsemi))])
    champ = fin[0]["w"]

    # ---- reporte de texto ----
    def show(res, title):
        out = [title]
        for r in res:
            tag = "real" if r["real"] else "sim "
            out.append(f"  [{tag}] {NAME[r['a']]:11}({r['sa']:>5}) vs {NAME[r['b']]:11}"
                       f"({r['sb']:>5})  -> {NAME[r['w']]}")
        return "\n".join(out)

    txt = [__doc__.strip(), "",
           show(l16, "== 16avos (izq) =="), show(l8, "== 8avos (izq) =="),
           show(l4, "== 4tos (izq) =="), show(lsemi, "== Semifinal (izq) =="),
           show(r16, "== 16avos (der) =="), show(r8, "== 8avos (der) =="),
           show(r4, "== 4tos (der) =="), show(rsemi, "== Semifinal (der) =="),
           show(third, "== 3er PUESTO =="), show(fin, "== FINAL =="), "",
           f"*** CAMPEÓN DEL MUNDO (simulado): {NAME[champ]} ***"]
    report = "\n".join(txt)
    print(report)
    C.save_report("simulate_bracket", report)

    html = build_html(l16, l8, l4, lsemi, r16, r8, r4, rsemi, fin, third, champ)
    out_path = C.ROOT / "docs" / "bracket.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  -> {out_path.relative_to(C.ROOT)}")


# --------------------------------------------------------------------------- #
# Generación del HTML (réplica del bracket)
# --------------------------------------------------------------------------- #
def team_row(code, score, is_winner):
    cls = "team win" if is_winner else "team"
    return (f'<div class="{cls}"><span class="fl">{flag(code)}</span>'
            f'<span class="cd">{code}</span><span class="sc">{score}</span></div>')


def match_box(r, extra=""):
    wa = r["w"] == r["a"]
    return (f'<div class="match {extra}">{team_row(r["a"], r["sa"], wa)}'
            f'{team_row(r["b"], r["sb"], not wa)}</div>')


def col(matches, header, extra=""):
    boxes = "".join(match_box(m, extra) for m in matches)
    return f'<div class="colwrap"><div class="hd">{header}</div><div class="col">{boxes}</div></div>'


def build_html(l16, l8, l4, lsemi, r16, r8, r4, rsemi, fin, third, champ):
    center = (
        '<div class="colwrap center">'
        '<div class="hd gold">FINAL</div>'
        f'<div class="col">{match_box(fin[0], "final")}'
        '<div class="lbl orange">3.er puesto</div>'
        f'{match_box(third[0], "third")}'
        f'<div class="champ">🏆 Campeón<br><b>{NAME[champ]}</b> {flag(champ)}</div>'
        '</div></div>')
    body = (
        col(l16, "16avos") + col(l8, "8avos") + col(l4, "4tos") + col(lsemi, "Semis")
        + center
        + col(rsemi, "Semis") + col(r4, "4tos") + col(r8, "8avos") + col(r16, "16avos"))
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mundial 2026 — Simulación (modelo Stacking)</title>
<style>
 * {{ box-sizing: border-box; }}
 body {{ margin:0; background:#0a3528; color:#dff0e6;
        font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
 header {{ padding:16px 22px; display:flex; align-items:baseline; gap:14px;
           background:#0c3d2d; border-bottom:1px solid #155e3f; }}
 header h1 {{ font-size:20px; margin:0; }}
 header .sub {{ font-size:12px; color:#8fce9f; }}
 .bracket {{ display:flex; gap:10px; padding:20px; align-items:stretch;
             overflow-x:auto; min-height:88vh; }}
 .colwrap {{ display:flex; flex-direction:column; min-width:120px; }}
 .hd {{ text-align:center; color:#5fd08a; font-weight:600; font-size:13px;
        margin-bottom:10px; }}
 .hd.gold {{ color:#f2c94c; }}
 .col {{ display:flex; flex-direction:column; justify-content:space-around;
         flex:1; gap:8px; }}
 .center {{ justify-content:flex-start; min-width:150px; }}
 .center .col {{ justify-content:center; }}
 .match {{ background:#0f3a2c; border:1px solid #1c4c3a; border-radius:8px;
           overflow:hidden; }}
 .match.final {{ border:2px solid #f2c94c; box-shadow:0 0 14px rgba(242,201,76,.35); }}
 .match.third {{ border:2px solid #e08a3c; }}
 .team {{ display:flex; align-items:center; gap:7px; padding:7px 9px;
          font-size:13px; color:#a9c6b5; }}
 .team + .team {{ border-top:1px solid #1c4c3a; }}
 .team .fl {{ font-size:15px; }}
 .team .cd {{ flex:1; letter-spacing:.5px; }}
 .team .sc {{ color:#7fae90; font-variant-numeric:tabular-nums; font-size:12px; }}
 .team.win {{ background:#155e3f; color:#ffffff; font-weight:700; }}
 .team.win .cd {{ color:#fff; }}
 .team.win .sc {{ color:#bfe9cf; }}
 .lbl {{ text-align:center; font-size:12px; margin:12px 0 6px; }}
 .lbl.orange {{ color:#e08a3c; font-weight:600; }}
 .champ {{ margin-top:16px; text-align:center; background:#123f2f;
           border:1px solid #f2c94c; border-radius:10px; padding:14px;
           font-size:15px; line-height:1.5; }}
 .champ b {{ font-size:19px; color:#f2c94c; }}
</style></head>
<body>
<header><h1>Fase Eliminatoria — Mundial 2026</h1>
<span class="sub">Simulación con el modelo <b>Stacking</b> · 7 partidos con resultado real,
resto predicho (números = prob. de ganar) · demo fuera de distribución</span></header>
<div class="bracket">{body}</div>
</body></html>"""


if __name__ == "__main__":
    main()
