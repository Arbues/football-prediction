"""Preprocesamiento 06 - PCA sobre el bloque continuo (reduccion + rubrica 3.6).

Aplica PCA al bloque de 22 features CONTINUAS ya estandarizadas (media 0, var 1),
que es donde vive la multicolinealidad detectada en el EDA: los diffs son
combinaciones lineales exactas de sus componentes (elo_diff = HomeElo - AwayElo,
etc.), asi que se esperan varias direcciones de varianza casi nula. PCA las
absorbe y descorrelaciona (Σ_Y diagonal; cf. Clase-3).

Se elige el numero de componentes por dos criterios exigidos por la rubrica:
  - Retencion del 95% de la varianza acumulada.
  - Criterio del codo (maxima distancia a la cuerda de la curva de varianza).

PCA se ajusta SOLO en train. Los flags binarios y el one-hot de Division NO se
reducen (no son continuos): se concatenan tras las componentes. Se guarda una
variante PCA del dataset para los modelos de distancia/lineales; los arboles
siguen usando la matriz cruda (X_*.parquet) por interpretabilidad/SHAP.

Salida: figures/pca_scree.png, pca_cumvar.png; data/processed/X_{split}_pca.parquet;
        results/17_pca.txt
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from _common import FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, new_report


def _elbow_index(cumvar: np.ndarray) -> int:
    """Codo por maxima distancia a la cuerda (metodo Kneedle simplificado)."""
    n = len(cumvar)
    x = np.arange(n)
    x0, y0, x1, y1 = 0, cumvar[0], n - 1, cumvar[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * cumvar + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0)
    return int(np.argmax(num / den))


def main() -> None:
    rep = new_report("17_pca.txt")
    rep("=" * 70)
    rep("PREPROC 06 - PCA SOBRE EL BLOQUE CONTINUO")
    rep("=" * 70)

    feat_names = json.loads((PROCESSED_DIR / "feature_names.json").read_text())
    num_cols = [c for c in feat_names if c.startswith("num__")]
    other_cols = [c for c in feat_names if not c.startswith("num__")]
    rep(f"Continuas para PCA: {len(num_cols)} | pasan sin reducir: {len(other_cols)}")

    Xtr = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    pca = PCA(random_state=RANDOM_STATE).fit(Xtr[num_cols].to_numpy())
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    eig = pca.explained_variance_

    k95 = int(np.argmax(cum >= 0.95) + 1)
    k_elbow = _elbow_index(cum) + 1
    rep(f"\nComponentes: {len(evr)}")
    rep(f"k para 95% varianza: {k95}")
    rep(f"k por criterio del codo: {k_elbow}")
    rep("\nVARIANZA POR COMPONENTE:")
    rep(f"  {'PC':>3} {'autovalor':>10} {'%var':>7} {'%acum':>7}")
    for i in range(len(evr)):
        rep(f"  {i+1:>3} {eig[i]:>10.3f} {100*evr[i]:>6.2f}% {100*cum[i]:>6.2f}%")

    n_zero = int((eig < 1e-8).sum())
    rep(f"\nDirecciones de varianza ~0 (colinealidad de los diffs): {n_zero}")
    rep(f"-> confirma que elo_diff/form5_diff/gf5_diff son combinaciones lineales")
    rep(f"   exactas; PCA las colapsa. Se adopta k = {k95} (criterio 95%).")

    k = k95
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Scree
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(eig) + 1), eig, "o-", color="#2c3e50")
    plt.axvline(k, ls="--", color="#e74c3c", label=f"k={k} (95%)")
    plt.xlabel("Componente principal"); plt.ylabel("Autovalor (varianza)")
    plt.title("Scree plot - PCA sobre features continuas"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "pca_scree.png", dpi=150); plt.close()
    # Varianza acumulada
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(cum) + 1), 100 * cum, "o-", color="#2c3e50")
    plt.axhline(95, ls="--", color="#e74c3c", label="95% varianza")
    plt.axvline(k, ls=":", color="#e74c3c")
    plt.xlabel("Nº de componentes"); plt.ylabel("Varianza acumulada (%)")
    plt.title("Varianza explicada acumulada"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "pca_cumvar.png", dpi=150); plt.close()

    # Guardar variante PCA (train-only fit) para cada split.
    pca_k = PCA(n_components=k, random_state=RANDOM_STATE).fit(Xtr[num_cols].to_numpy())
    pc_names = [f"PC{i+1}" for i in range(k)]
    rep("\nSHAPES VARIANTE PCA:")
    for name in ("train", "val", "test"):
        X = pd.read_parquet(PROCESSED_DIR / f"X_{name}.parquet")
        Z = pca_k.transform(X[num_cols].to_numpy())
        out = pd.concat(
            [pd.DataFrame(Z, columns=pc_names), X[other_cols].reset_index(drop=True)],
            axis=1,
        )
        out.to_parquet(PROCESSED_DIR / f"X_{name}_pca.parquet", index=False)
        rep(f"  {name:<6} {out.shape}  ({k} PC + {len(other_cols)} flags/one-hot)")

    rep(f"\n[figuras] pca_scree.png, pca_cumvar.png")
    rep(f"[guardado] X_{{train,val,test}}_pca.parquet")
    rep.save()


if __name__ == "__main__":
    main()
