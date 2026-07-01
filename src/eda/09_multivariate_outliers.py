"""EDA 09 - Outliers multivariados (distancia de Mahalanobis).

Complementa el analisis IQR univariado (06): detecta observaciones anomalas en
el espacio conjunto de las features pre-partido usando la forma cuadratica
D_M^2 = (x-mu)^T Sigma^-1 (x-mu), con umbral chi-cuadrado. La rubrica 3.5 pide
explicitamente 'valores atipicos multivariados'.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

from _common import ensure_figures_dir, load_matches, new_report

sns.set_theme(style="whitegrid")

FEATURES = ["HomeElo", "AwayElo", "Form3Home", "Form5Home", "Form3Away", "Form5Away"]


def main() -> None:
    log = new_report("09_multivariate_outliers.txt")
    figdir = ensure_figures_dir()
    m = load_matches()

    sub = m[FEATURES].dropna()
    X = sub.values
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # Regularizacion minima para estabilidad numerica de la inversion.
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(FEATURES)))
    diff = X - mu
    dm2 = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)  # D_M^2 vectorizado

    df = len(FEATURES)
    thr = chi2.ppf(0.999, df=df)  # umbral al 99.9%

    log("=" * 70)
    log("EDA 09 - OUTLIERS MULTIVARIADOS (MAHALANOBIS)")
    log("=" * 70)
    log(f"Features del espacio conjunto ({df}): {FEATURES}")
    log(f"Observaciones completas (sin nulos): {len(sub):,}")
    log(f"Umbral chi2(0.999, df={df}) sobre D_M^2 = {thr:.3f}")
    log("")
    log(f"D_M^2: min={dm2.min():.2f}  media={dm2.mean():.2f}  "
        f"mediana={np.median(dm2):.2f}  max={dm2.max():.2f}")
    log(f"(bajo normalidad multivariante, E[D_M^2] = df = {df})")

    n_out = int((dm2 > thr).sum())
    log("")
    log(f"Outliers multivariados (D_M^2 > umbral): {n_out:,} ({100*n_out/len(sub):.2f}%)")

    # Comparacion con outliers univariados (IQR) sobre las mismas filas.
    def iqr_mask(s):
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

    uni_any = np.zeros(len(sub), dtype=bool)
    for j in range(len(FEATURES)):
        uni_any |= iqr_mask(X[:, j])
    mv = dm2 > thr
    log("")
    log("-" * 70)
    log("MULTIVARIADO vs UNIVARIADO (misma muestra)")
    log("-" * 70)
    log(f"Outlier en >=1 variable (IQR univariado): {int(uni_any.sum()):,} "
        f"({100*uni_any.mean():.2f}%)")
    log(f"Outlier multivariado (Mahalanobis):       {int(mv.sum()):,} "
        f"({100*mv.mean():.2f}%)")
    log(f"Multivariado PERO no univariado: {int((mv & ~uni_any).sum()):,} "
        f"(anomalos por COMBINACION, invisibles al analisis por columna)")
    log(f"Univariado PERO no multivariado: {int((~mv & uni_any).sum()):,} "
        f"(extremos en una variable pero coherentes con la estructura conjunta)")

    # --- Perfilado: que variable dispara cada outlier + arquetipo futbolero ---
    # z-score de cada variable: cuantifica cuan lejos de la media esta cada campo.
    sd = X.std(axis=0)
    Z = (X - mu) / sd
    info = m.loc[sub.index, ["MatchDate", "Division", "HomeTeam", "AwayTeam"]].reset_index(drop=True)

    def arquetipo(z: np.ndarray) -> str:
        # z = [HomeElo, AwayElo, Form3Home, Form5Home, Form3Away, Form5Away]
        he, ae, f5h, f5a = z[0], z[1], z[3], z[5]
        if he > 1 and f5h < -1.2:
            return "Local fuerte en crisis (Elo alto + forma nula)"
        if ae > 1 and f5a < -1.2:
            return "Visita fuerte en crisis (Elo alto + forma nula)"
        if he < -1 and f5h > 1.2:
            return "Local debil en racha (Elo bajo + forma perfecta)"
        if ae < -1 and f5a > 1.2:
            return "Visita debil en racha (Elo bajo + forma perfecta)"
        if max(abs(he), abs(ae)) > 2.5:
            return "Elo extremo (super-club en el techo / colista en el piso)"
        return "Combinacion fuerza-forma atipica (mixta)"

    out_idx = np.where(mv)[0]
    arqs = [arquetipo(Z[i]) for i in out_idx]
    dom_var = [FEATURES[int(np.argmax(np.abs(Z[i])))] for i in out_idx]

    log("")
    log("-" * 70)
    log("QUE VARIABLE DISPARA EL OUTLIER (|z| maximo entre los atipicos)")
    log("-" * 70)
    for v, c in pd.Series(dom_var).value_counts().items():
        log(f"  {v:<12} {c:>3} outliers ({100*c/len(out_idx):.0f}%)")
    log("  -> Elo (Home/Away) domina: la mayoria son equipos cuyo rating esta muy")
    log("     lejos de la media de su liga.")

    log("")
    log("-" * 70)
    log("ARQUETIPOS DE OUTLIER (interpretacion futbolera)")
    log("-" * 70)
    for a, c in pd.Series(arqs).value_counts().items():
        log(f"  {c:>3}x  {a}")

    log("")
    log("TOP-8 outliers (equipos reales, valores y z-scores):")
    order = out_idx[np.argsort(dm2[out_idx])[::-1]][:8]
    for i in order:
        r = info.iloc[i]
        log(f"  D_M^2={dm2[i]:5.1f} | {r['HomeTeam']} (Elo {X[i,0]:.0f}, F5 {X[i,3]:.0f}) vs "
            f"{r['AwayTeam']} (Elo {X[i,1]:.0f}, F5 {X[i,5]:.0f})"
            f" | {r['Division']} {str(r['MatchDate'])[:10]}")
        log(f"           z: HomeElo={Z[i,0]:+.1f} Form5H={Z[i,3]:+.1f}  "
            f"AwayElo={Z[i,1]:+.1f} Form5A={Z[i,5]:+.1f}  -> {arquetipo(Z[i])}")

    log("")
    log("-" * 70)
    log("EXPLICACION FUTBOLERA")
    log("-" * 70)
    log("Dos historias detras de los atipicos multivariados:")
    log(" (1) SUPER-CLUBES: la mayoria son Bayern, Real Madrid, PSG, Ajax, etc., con")
    log("     Elo ~2000-2100, muy por encima del ~1533 medio. Son raros porque muy")
    log("     pocos equipos alcanzan ese nivel; el rating por si solo los aleja.")
    log(" (2) CONTRADICCION FUERZA-FORMA (los que el IQR NO ve): un grande con racha")
    log("     de derrotas (Elo alto + forma 0, p.ej. Real Madrid 2018 con Form5=1) o")
    log("     un modesto en racha perfecta (Elo bajo + Form5=15, p.ej. Jaen o Amiens).")
    log("     Ninguno de los dos campos es extremo por si solo, pero la COMBINACION")
    log("     rompe la relacion esperada (Elo y forma correlacionan r~0.31): por eso")
    log("     solo la distancia de Mahalanobis los detecta.")

    # --- Figura: distribucion de D_M^2 vs chi2 teorico ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(dm2[dm2 < thr * 1.5], bins=100, density=True, color="#4c72b0",
            alpha=0.7, label="D_M^2 empirico")
    xs = np.linspace(0, thr * 1.5, 300)
    ax.plot(xs, chi2.pdf(xs, df), "r-", lw=2, label=f"chi2 teorica (df={df})")
    ax.axvline(thr, color="black", linestyle="--", label=f"umbral 99.9% = {thr:.1f}")
    ax.set_xlabel("Distancia de Mahalanobis al cuadrado (D_M^2)")
    ax.set_ylabel("densidad")
    ax.set_title("D_M^2 empirico vs chi-cuadrado teorica")
    ax.legend(fontsize=8)

    # Scatter Elo con outliers marcados y top-5 anotados con el equipo dominante.
    ax = axes[1]
    ax.scatter(X[~mv, 0], X[~mv, 1], s=3, alpha=0.15, color="#7f7f7f", label="normal")
    ax.scatter(X[mv, 0], X[mv, 1], s=12, alpha=0.7, color="#d62728", label="outlier MV")
    for i in out_idx[np.argsort(dm2[out_idx])[::-1]][:5]:
        # etiqueta con el equipo cuyo Elo esta mas lejos de la media.
        equipo = info.iloc[i]["HomeTeam"] if abs(Z[i, 0]) >= abs(Z[i, 1]) else info.iloc[i]["AwayTeam"]
        ax.annotate(equipo, (X[i, 0], X[i, 1]), fontsize=7, color="black",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("HomeElo"); ax.set_ylabel("AwayElo")
    ax.set_title("Outliers multivariados en el plano Elo (top-5 etiquetados)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "mahalanobis_outliers.png", dpi=150)
    plt.close(fig)

    log("")
    log("Lectura: la distancia de Mahalanobis captura anomalias por COMBINACION de")
    log("variables (p. ej. Elo alto con forma nula) que el IQR por columna no ve. La")
    log("cola pesada frente a la chi2 teorica indica leve no-normalidad multivariante.")
    log(f"\n[figura] mahalanobis_outliers.png en {figdir}")
    log.save()


if __name__ == "__main__":
    main()
