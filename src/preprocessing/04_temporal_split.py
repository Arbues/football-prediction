"""Preprocesamiento 04 - Split temporal por fecha (holdout final).

La rubrica (Seccion 12) prohibe el K-Fold aleatorio en series temporales: entrenar
con el futuro para predecir el pasado invalida el poder predictivo. Se usa un
holdout por fecha:

  train : temporadas <= 2021   (ajuste + CV interna con TimeSeriesSplit)
  val   : 2022-2023            (seleccion de modelo / early stopping)
  test  : 2024-2025            (evaluacion final, intacto)

El corte respeta el orden real del calendario: ninguna fila de train es
posterior a una de val/test. La CV para elegir hiperparametros (TimeSeriesSplit)
se aplica DENTRO de train en la fase de modelado, no aqui.

Salida: data/processed/_matches_split.parquet + results/15_split.txt
"""

from __future__ import annotations

import pandas as pd

from _common import (
    FEATURES_PARQUET, SPLIT_PARQUET, TARGET, TARGET_ORDER,
    TEST_YEARS, TRAIN_END_YEAR, VAL_YEARS, new_report,
)


def _assign_split(year: int) -> str:
    if year <= TRAIN_END_YEAR:
        return "train"
    if VAL_YEARS[0] <= year <= VAL_YEARS[1]:
        return "val"
    return "test"


def main() -> None:
    rep = new_report("15_split.txt")
    rep("=" * 70)
    rep("PREPROC 04 - SPLIT TEMPORAL POR FECHA")
    rep("=" * 70)

    df = pd.read_parquet(FEATURES_PARQUET)
    df["year"] = df["MatchDate"].dt.year
    df["split"] = df["year"].map(_assign_split)

    rep(f"Corte: train <= {TRAIN_END_YEAR} | val {VAL_YEARS[0]}-{VAL_YEARS[1]} | "
        f"test {TEST_YEARS[0]}-{TEST_YEARS[1]}")

    rep("\nTAMANO Y RANGO POR SPLIT:")
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        rep(f"  {name:<6} {len(sub):>8,} filas ({100*len(sub)/len(df):4.1f}%)  "
            f"[{sub['MatchDate'].min().date()} -> {sub['MatchDate'].max().date()}]")

    # Verificacion de no solapamiento temporal (frontera limpia).
    tmax = df.loc[df.split == "train", "MatchDate"].max()
    vmin = df.loc[df.split == "val", "MatchDate"].min()
    vmax = df.loc[df.split == "val", "MatchDate"].max()
    testmin = df.loc[df.split == "test", "MatchDate"].min()
    ok = tmax < vmin and vmax < testmin
    rep(f"\nFrontera temporal limpia (train<val<test): {ok}")

    # Balance de clases por split: evidencia de deriva temporal (COVID, etc.).
    rep("\nBALANCE DE CLASES POR SPLIT (%):")
    rep(f"  {'split':<6} " + " ".join(f"{c:>7}" for c in TARGET_ORDER))
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        pct = sub[TARGET].value_counts(normalize=True) * 100
        rep(f"  {name:<6} " + " ".join(f"{pct.get(c, 0):7.2f}" for c in TARGET_ORDER))
    rep("  -> la caida de H% en val/test (efecto COVID 2020-21 y post) es")
    rep("     justamente lo que el split temporal preserva y el aleatorio borraria.")

    df.to_parquet(SPLIT_PARQUET, index=False)
    rep(f"\n[guardado] {SPLIT_PARQUET}  ({df.shape[0]:,} x {df.shape[1]})")
    rep.save()


if __name__ == "__main__":
    main()
