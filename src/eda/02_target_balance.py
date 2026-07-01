"""EDA 02 - Balance de clases del objetivo FTResult.

Conteo y porcentaje por clase (H/D/A), ratio de desbalance (IR) y baseline
del clasificador mayoritario. Figura de barras para la exposicion.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    TARGET, TARGET_ORDER, ensure_figures_dir, load_matches, new_report,
)

CLASS_LABEL = {"H": "Local (H)", "D": "Empate (D)", "A": "Visita (A)"}
CLASS_COLOR = {"H": "#2ca02c", "D": "#7f7f7f", "A": "#d62728"}


def main() -> None:
    log = new_report("02_target_balance.txt")
    matches = load_matches()

    y = matches[TARGET].dropna()
    counts = y.value_counts().reindex(TARGET_ORDER)
    pct = (100 * counts / counts.sum()).round(2)

    log("=" * 70)
    log("EDA 02 - BALANCE DE CLASES (FTResult)")
    log("=" * 70)
    log(f"Instancias etiquetadas: {int(counts.sum()):,} "
        f"(se excluyen {int(matches[TARGET].isna().sum())} nulos)")
    log("")
    log(f"{'clase':<12} {'conteo':>10} {'porcentaje':>12}")
    for c in TARGET_ORDER:
        log(f"{CLASS_LABEL[c]:<12} {int(counts[c]):>10,} {pct[c]:>11.2f}%")

    ir = counts.max() / counts.min()
    log("")
    log(f"Ratio de desbalance IR = clase_mayor / clase_menor = "
        f"{counts.max():,} / {counts.min():,} = {ir:.3f}")
    log(f"Clase mayoritaria: {counts.idxmax()} ({CLASS_LABEL[counts.idxmax()]})")
    log(f"Clase minoritaria: {counts.idxmin()} ({CLASS_LABEL[counts.idxmin()]})")
    log("")
    log("Lectura: desbalance MODERADO (IR < 2). No exige sobre-muestreo agresivo,")
    log("pero el empate (D) es minoritario y sera la clase mas dificil de predecir.")
    log("Se recomienda class_weight='balanced' y reportar F1 por clase (no solo accuracy).")

    # --- Baselines triviales (referencia para el modelado) ---
    log("")
    log("-" * 70)
    log("BASELINES TRIVIALES (piso a superar)")
    log("-" * 70)
    p = counts / counts.sum()
    acc_major = p.max()
    acc_random = (p ** 2).sum()  # predecir segun la distribucion a priori
    log(f"Accuracy prediciendo siempre la mayoritaria (H): {acc_major:.4f}")
    log(f"Accuracy de un clasificador aleatorio estratificado: {acc_random:.4f}")
    log(f"El modelo debe superar >= {acc_major:.4f} de accuracy para aportar valor.")

    # --- Figura ---
    ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [CLASS_LABEL[c] for c in TARGET_ORDER],
        [counts[c] for c in TARGET_ORDER],
        color=[CLASS_COLOR[c] for c in TARGET_ORDER],
        edgecolor="black",
    )
    for c, b in zip(TARGET_ORDER, bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{int(counts[c]):,}\n{pct[c]:.1f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Numero de partidos")
    ax.set_title(f"Balance de clases de FTResult (IR = {ir:.2f})")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ensure_figures_dir() / "class_balance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"\n[figura] {out}")

    log.save()


if __name__ == "__main__":
    main()
