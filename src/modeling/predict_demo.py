"""Predice los 16avos del Mundial 2026 con features REALES y compara con lo ya jugado.

Reutiliza la maquinaria de `simulate_bracket` (mismos datos: Elo + forma y goles de la
fase de grupos 2026, predicción neutral promediando ambas orientaciones). Para los 7
partidos ya disputados compara el pick del modelo (Stacking) con el resultado oficial.
Demo fuera de distribución: los modelos se entrenaron con fútbol de clubes.
"""
import warnings

import _common as C
from simulate_bracket import LEFT_R32, NAME, RIGHT_R32, play

warnings.filterwarnings("ignore")

R32 = LEFT_R32 + RIGHT_R32


def main():
    C.set_seeds()
    res = [play(a, b) for a, b in R32]
    out = ["16AVOS MUNDIAL 2026 — modelo Stacking",
           "Features reales: Elo + puntos/goles de la fase de grupos. Prob. neutral "
           "(promedio de ambas orientaciones).", "",
           f"{'Partido':26} {'gana izq':>9} {'empate':>7} {'gana der':>9}  "
           f"{'Pick':11} {'Resultado real':18}"]
    hits = tot = 0
    for m in res:
        na, nb = NAME[m["a"]], NAME[m["b"]]
        real = f"{m['sa']}-{m['sb']} -> {NAME[m['w']]}" if m["real"] else "pendiente"
        mark = ""
        if m["real"]:
            tot += 1
            ok = m["pick"] == m["w"]
            hits += int(ok)
            mark = "OK" if ok else "x"
        out.append(f"{na + ' vs ' + nb:26} {m['pa']:>8.0%} {m['pd']:>7.0%} {m['pb']:>9.0%}  "
                   f"{NAME[m['pick']]:11} {real:18} {mark}")
    out.append(f"\nAciertos en los {tot} ya jugados: {hits}/{tot}")
    txt = "\n".join(out)
    print(txt)
    C.save_report("predict_demo", txt)


if __name__ == "__main__":
    main()
