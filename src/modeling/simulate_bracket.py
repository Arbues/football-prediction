"""Simula el cuadro completo del Mundial 2026 (16avos -> final + 3er puesto) con el
modelo propuesto (Stacking) y genera una web (docs/bracket.html).

Features REALES por selección: Elo actual + rendimiento de la FASE DE GRUPOS 2026
(puntos -> forma, goles a favor/en contra -> promedio de gol y racha). NO es solo Elo:
el modelo usa las diferencias de forma y de gol, así que un equipo con Elo algo menor
pero mejor fase de grupos puede quedar por delante. Cada cruce se predice en las DOS
orientaciones y se promedia (sede neutral, sin sesgo de localía). Los 7 partidos ya
jugados usan el resultado REAL; el resto, la predicción. En eliminatoria avanza el de
mayor probabilidad de ganar. Sigue siendo demo fuera de distribución (entrenado con
clubes); h2h y descanso se dejan neutros por falta de datos comparables.
"""
import json
import warnings

import joblib
import pandas as pd

import _common as C

warnings.filterwarnings("ignore")

# Datos REALES: Elo (eloratings/búsqueda) + fase de grupos Mundial 2026 (NBC Sports):
# gf/ga/pts = goles a favor, en contra y puntos en los 3 partidos de grupo.
TEAMS = {
    "ARG": {"elo": 2144, "gf": 10, "ga": 3, "pts": 9}, "ESP": {"elo": 2144, "gf": 8, "ga": 3, "pts": 7},
    "FRA": {"elo": 2123, "gf": 10, "ga": 2, "pts": 9}, "ENG": {"elo": 2050, "gf": 8, "ga": 4, "pts": 7},
    "BRA": {"elo": 2009, "gf": 10, "ga": 4, "pts": 7}, "POR": {"elo": 2005, "gf": 6, "ga": 1, "pts": 4},
    "NED": {"elo": 1990, "gf": 9, "ga": 3, "pts": 7}, "GER": {"elo": 1955, "gf": 8, "ga": 2, "pts": 6},
    "BEL": {"elo": 1945, "gf": 5, "ga": 2, "pts": 5}, "CRO": {"elo": 1920, "gf": 6, "ga": 6, "pts": 6},
    "COL": {"elo": 1900, "gf": 6, "ga": 3, "pts": 6}, "MAR": {"elo": 1895, "gf": 8, "ga": 5, "pts": 7},
    "NOR": {"elo": 1860, "gf": 6, "ga": 5, "pts": 6}, "SUI": {"elo": 1855, "gf": 9, "ga": 5, "pts": 7},
    "JPN": {"elo": 1845, "gf": 6, "ga": 2, "pts": 5}, "SEN": {"elo": 1840, "gf": 4, "ga": 3, "pts": 4},
    "ECU": {"elo": 1825, "gf": 4, "ga": 4, "pts": 4}, "SWE": {"elo": 1810, "gf": 4, "ga": 4, "pts": 4},
    "MEX": {"elo": 1800, "gf": 9, "ga": 3, "pts": 9}, "USA": {"elo": 1795, "gf": 7, "ga": 3, "pts": 6},
    "AUT": {"elo": 1790, "gf": 4, "ga": 4, "pts": 4}, "CIV": {"elo": 1790, "gf": 7, "ga": 5, "pts": 6},
    "CAN": {"elo": 1785, "gf": 9, "ga": 4, "pts": 4}, "DZA": {"elo": 1770, "gf": 2, "ga": 4, "pts": 4},
    "AUS": {"elo": 1745, "gf": 4, "ga": 4, "pts": 4}, "GHA": {"elo": 1730, "gf": 4, "ga": 4, "pts": 4},
    "BIH": {"elo": 1720, "gf": 3, "ga": 4, "pts": 4}, "PAR": {"elo": 1720, "gf": 3, "ga": 5, "pts": 4},
    "EGY": {"elo": 1710, "gf": 4, "ga": 2, "pts": 5}, "COD": {"elo": 1690, "gf": 5, "ga": 4, "pts": 4},
    "RSA": {"elo": 1680, "gf": 5, "ga": 6, "pts": 4}, "CPV": {"elo": 1620, "gf": 3, "ga": 3, "pts": 3},
}
ELO = {k: v["elo"] for k, v in TEAMS.items()}
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


def _row(hc, ac):
    """Fila cruda con features REALES de fase de grupos para el cruce hc(local) vs ac."""
    h, a = TEAMS[hc], TEAMS[ac]
    hgf, hga, agf, aga = h["gf"] / 3, h["ga"] / 3, a["gf"] / 3, a["ga"] / 3   # goles/partido
    hf5, af5 = h["pts"] / 3 * 5, a["pts"] / 3 * 5    # puntos escalados a 5 partidos
    hf3, af3 = h["pts"], a["pts"]                     # puntos en 3 (≈ Form3)
    hws, aws = h["pts"] // 3, a["pts"] // 3           # aprox. de victorias/racha
    return {"match_id": 0, "Division": "WC", "MatchDate": pd.Timestamp("2026-07-05"),
            "HomeTeam": hc, "AwayTeam": ac, "FTResult": "H", "year": 2026, "split": "test",
            "HomeElo": h["elo"], "AwayElo": a["elo"], "elo_diff": h["elo"] - a["elo"],
            "Form5Home": hf5, "Form5Away": af5, "form5_diff": hf5 - af5,
            "Form3Home": hf3, "Form3Away": af3, "form3_diff": hf3 - af3,
            "home_gf5": hgf, "away_gf5": agf, "gf5_diff": hgf - agf,
            "home_ga5": hga, "away_ga5": aga, "ga5_diff": hga - aga,
            "home_rest_days": 5, "away_rest_days": 5,
            "home_win_streak": hws, "away_win_streak": aws,
            "h2h_played": 0, "h2h_home_winrate": 0.5, "h2h_avg_goals": 2.5,
            "is_top_league": 0, "elo_missing": 0}


def _neutral_probs(a, b):
    """Prob. (a_gana, b_gana, empate) promediando las dos orientaciones (sede neutral)."""
    df = pd.DataFrame([_row(a, b), _row(b, a)])[list(_PRE.feature_names_in_)]
    X = pd.DataFrame(_PRE.transform(df), columns=_FEATS)
    P = _MODEL.predict_proba(X)  # columnas [A, D, H]
    pa = 0.5 * (P[0][2] + P[1][0])
    pb = 0.5 * (P[0][0] + P[1][2])
    pd_ = 0.5 * (P[0][1] + P[1][1])
    return float(pa), float(pb), float(pd_)


def play(a, b):
    """Juega el cruce a vs b -> dict con probs, ganador y marcador."""
    pa, pb, pd_ = _neutral_probs(a, b)
    pick = a if pa >= pb else b
    if (a, b) in REAL:
        w, sa, sb, real = *REAL[(a, b)], True
    else:
        w, sa, sb, real = pick, f"{round(100 * pa)}%", f"{round(100 * pb)}%", False
    return {"a": a, "b": b, "ea": ELO[a], "eb": ELO[b], "pa": pa, "pb": pb, "pd": pd_,
            "w": w, "sa": sa, "sb": sb, "real": real, "pick": pick,
            "fa": TEAMS[a]["pts"], "fb": TEAMS[b]["pts"],
            "gfa": TEAMS[a]["gf"], "gfb": TEAMS[b]["gf"]}


def round_of(pairs):
    """Juega una lista de cruces (tuplas de codigos) -> lista de dicts resultado."""
    return [play(a, b) for a, b in pairs]


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


def explain(m):
    """Justifica el cruce con Elo + forma real (puntos y goles de la fase de grupos)."""
    a, b, w = m["a"], m["b"], m["w"]
    na, nb, nw = NAME[a], NAME[b], NAME[w]
    d = abs(m["ea"] - m["eb"])
    mayor = na if m["ea"] >= m["eb"] else nb
    fav = a if m["pa"] >= m["pb"] else b
    nfav, favp = NAME[fav], max(m["pa"], m["pb"])
    gap = abs(m["pa"] - m["pb"])
    nivel = ("un abismo de nivel" if d >= 200 else "ventaja clara" if d >= 100
             else "ligera ventaja" if d >= 40 else "Elo casi idéntico")
    base = (f"<b>Elo:</b> {na} {m['ea']} vs {nb} {m['eb']} — {nivel} ({d} pts) para {mayor}. "
            f"<b>Forma en grupos:</b> {na} {m['fa']} pts y {m['gfa']} goles; "
            f"{nb} {m['fb']} pts y {m['gfb']} goles. ")
    if m["real"]:
        pens = "(" in m["sa"] or "(" in m["sb"]
        if m["pick"] == w:
            return (base + f"El modelo (Elo+forma) favorecía a {nfav} ({favp:.0%}) y así fue "
                    f"({m['sa']}–{m['sb']}). Acierto.")
        t = (base + f"El modelo daba favorito a {nfav} ({favp:.0%}), pero avanzó {nw} "
             f"({m['sa']}–{m['sb']}).")
        if pens:
            t += (f" Fue empate en los 90' —{m['pd']:.0%} de empate para el modelo— y se definió "
                  f"por penales, azar puro.")
        return t
    elo_fav = a if m["ea"] >= m["eb"] else b
    matiz = ""
    if elo_fav != fav:
        matiz = (f" Aquí la <b>forma pesa más que el Elo</b>: {nfav} llegó mejor de la fase de "
                 f"grupos y el modelo lo pone por delante pese a un Elo algo menor.")
    if gap >= 0.40:
        lect = f"victoria muy probable de {nfav} ({favp:.0%})"
    elif gap >= 0.18:
        lect = f"{nfav} favorito ({favp:.0%})"
    else:
        lect = f"casi un volado: {na} {m['pa']:.0%} vs {nb} {m['pb']:.0%}"
    return (base + f"<b>Predicción:</b> {lect}.{matiz} Avanza {nw}.")


def acc_item(m):
    badge = "real" if m["real"] else "sim"
    return (f'<details><summary>'
            f'<span class="mt">{flag(m["a"])} {m["a"]} vs {m["b"]} {flag(m["b"])}</span>'
            f'<span class="mr">→ {NAME[m["w"]]}</span>'
            f'<span class="bg {badge}">{badge}</span></summary>'
            f'<div class="exp">{explain(m)}</div></details>')


def acc_group(title, matches):
    return f'<div class="grp"><h3>{title}</h3>{"".join(acc_item(m) for m in matches)}</div>'


def build_accordion(l16, l8, l4, lsemi, r16, r8, r4, rsemi, fin, third):
    return (
        '<section class="explain"><h2>Explicación de cada resultado</h2>'
        '<p class="note">Cada acordeón detalla por qué salió ese resultado: el Elo de cada '
        'selección, la probabilidad del modelo y la lectura. Los marcados <b>real</b> usan el '
        'marcador oficial; los <b>sim</b>, la predicción del modelo.</p>'
        + acc_group("16avos · Izquierda", l16) + acc_group("16avos · Derecha", r16)
        + acc_group("8avos · Izquierda", l8) + acc_group("8avos · Derecha", r8)
        + acc_group("4tos · Izquierda", l4) + acc_group("4tos · Derecha", r4)
        + acc_group("Semifinales", lsemi + rsemi)
        + acc_group("3.er puesto", third) + acc_group("FINAL", fin)
        + '</section>')


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
    acc = build_accordion(l16, l8, l4, lsemi, r16, r8, r4, rsemi, fin, third)
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
 .explain {{ padding:6px 22px 46px; max-width:1080px; margin:0 auto; }}
 .explain h2 {{ color:#5fd08a; font-size:19px; }}
 .explain .note {{ color:#8fce9f; font-size:13px; line-height:1.6; }}
 .grp {{ margin-top:20px; }}
 .grp h3 {{ color:#f2c94c; font-size:15px; border-bottom:1px solid #1c4c3a;
            padding-bottom:6px; margin-bottom:8px; }}
 details {{ background:#0f3a2c; border:1px solid #1c4c3a; border-radius:8px; margin:7px 0; }}
 summary {{ cursor:pointer; padding:10px 12px; display:flex; align-items:center; gap:10px;
            list-style:none; font-size:14px; }}
 summary::-webkit-details-marker {{ display:none; }}
 summary::before {{ content:"\\25B8"; color:#5fd08a; }}
 details[open] summary::before {{ content:"\\25BE"; }}
 details[open] {{ border-color:#2e6b4f; }}
 .mt {{ font-weight:600; letter-spacing:.3px; }}
 .mr {{ color:#bfe9cf; }}
 .bg {{ margin-left:auto; font-size:11px; padding:2px 8px; border-radius:10px; }}
 .bg.real {{ background:#155e3f; color:#fff; }}
 .bg.sim {{ background:#243b32; color:#8fce9f; }}
 .exp {{ padding:0 14px 12px 32px; color:#cfe6d8; font-size:13px; line-height:1.65; }}
 .exp b {{ color:#eaf6ee; }}
</style></head>
<body>
<header><h1>Fase Eliminatoria — Mundial 2026</h1>
<span class="sub">Modelo <b>Stacking</b> con features reales (Elo + forma y goles de la fase de
grupos) · 7 partidos con resultado oficial, resto predicho (números = prob. de ganar) ·
demo fuera de distribución</span></header>
<div class="bracket">{body}</div>
{acc}
</body></html>"""


if __name__ == "__main__":
    main()
