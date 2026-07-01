"""Preprocesamiento 05 - Imputacion train-only, diffs, escalado y codificacion.

Cierra la "Regla de Oro" anti-leakage: TODO estadistico de transformacion
(medianas de imputacion, media/desv del escalado, categorias del one-hot) se
calcula UNICAMENTE sobre train y se aplica ciegamente a val/test.

Pasos:
  (1) Imputacion justificada por dominio (medianas de train):
      - Elo: mediana por LIGA (fallback global) + flag elo_missing. El nulo es
        estructural por liga (ligas no europeas sin cobertura), por eso la
        mediana de la propia liga es la referencia natural, no 0 (evita la
        "trampa de la media" que colapsa la varianza; cf. Clase-2).
      - Form: 0 (primeras jornadas sin historial -> cero puntos acumulados).
      - gf5/ga5/rest_days: mediana de train (primer partido del equipo).
      - h2h_home_winrate: 0.5 (previa neutral sin enfrentamientos); h2h_avg_goals:
        mediana de train; h2h_played ya es 0.
  (2) Diferencias local-visita (senal mas informativa del EDA): elo_diff,
      form5_diff, form3_diff, gf5_diff, ga5_diff.
  (3) ColumnTransformer: StandardScaler (continuas) + passthrough (flags binarios)
      + OneHotEncoder (Division). fit SOLO en train.

Salida: data/processed/{X_train,X_val,X_test,y_train,y_val,y_test}.parquet,
        index_ref.parquet, feature_names.json; models/preprocessor.joblib;
        results/16_transform.txt
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _common import (
    PROCESSED_DIR, PROJECT_ROOT, SPLIT_PARQUET, TARGET, new_report,
)

FORM_COLS = ["Form3Home", "Form5Home", "Form3Away", "Form5Away"]
GF_GA_COLS = ["home_gf5", "away_gf5", "home_ga5", "away_ga5"]
REST_COLS = ["home_rest_days", "away_rest_days"]
STREAK_COLS = ["home_win_streak", "away_win_streak"]

CONTINUOUS = (
    ["HomeElo", "AwayElo", "elo_diff"]
    + ["Form5Home", "Form5Away", "form5_diff", "Form3Home", "Form3Away", "form3_diff"]
    + ["home_gf5", "away_gf5", "gf5_diff", "home_ga5", "away_ga5", "ga5_diff"]
    + REST_COLS + STREAK_COLS
    + ["h2h_played", "h2h_home_winrate", "h2h_avg_goals"]
)
FLAGS = ["is_top_league", "elo_missing"]
CATEGORICAL = ["Division"]


def _impute(df: pd.DataFrame, train_mask: pd.Series, rep) -> pd.DataFrame:
    df = df.copy()

    # flag antes de tocar el Elo
    df["elo_missing"] = (df["HomeElo"].isna() | df["AwayElo"].isna()).astype(int)

    elo_long = pd.concat([
        df.loc[train_mask, ["Division", "HomeElo"]].rename(columns={"HomeElo": "e"}),
        df.loc[train_mask, ["Division", "AwayElo"]].rename(columns={"AwayElo": "e"}),
    ])
    league_med = elo_long.groupby("Division")["e"].median()
    global_med = float(elo_long["e"].median())
    for col in ("HomeElo", "AwayElo"):
        df[col] = df[col].fillna(df["Division"].map(league_med)).fillna(global_med)
    rep(f"  Elo: mediana global train = {global_med:.1f}; "
        f"ligas con mediana propia = {league_med.notna().sum()}; flag elo_missing.")

    for col in FORM_COLS:
        df[col] = df[col].fillna(0.0)
    rep("  Form3/Form5: imputadas a 0 (sin historial en primeras jornadas).")

    for col in GF_GA_COLS + REST_COLS:
        med = float(df.loc[train_mask, col].median())
        df[col] = df[col].fillna(med)
    rep("  gf5/ga5/rest_days: imputadas a la mediana de train.")

    df["h2h_home_winrate"] = df["h2h_home_winrate"].fillna(0.5)
    h2h_g_med = float(df.loc[train_mask, "h2h_avg_goals"].median())
    df["h2h_avg_goals"] = df["h2h_avg_goals"].fillna(h2h_g_med)
    rep(f"  h2h: winrate->0.5 (previa neutral); avg_goals->mediana train {h2h_g_med:.2f}.")

    return df


def _add_diffs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elo_diff"] = df["HomeElo"] - df["AwayElo"]
    df["form5_diff"] = df["Form5Home"] - df["Form5Away"]
    df["form3_diff"] = df["Form3Home"] - df["Form3Away"]
    df["gf5_diff"] = df["home_gf5"] - df["away_gf5"]
    df["ga5_diff"] = df["home_ga5"] - df["away_ga5"]
    return df


def main() -> None:
    rep = new_report("16_transform.txt")
    rep("=" * 70)
    rep("PREPROC 05 - IMPUTACION TRAIN-ONLY, DIFFS, ESCALADO Y ONE-HOT")
    rep("=" * 70)

    df = pd.read_parquet(SPLIT_PARQUET)
    train_mask = df["split"] == "train"

    rep("\n(1) IMPUTACION (medianas calculadas SOLO en train):")
    df = _impute(df, train_mask, rep)
    df = _add_diffs(df)

    assert df[CONTINUOUS].isna().sum().sum() == 0, "quedan nulos en continuas"
    rep("\n(2) Diffs local-visita anadidos: elo_diff, form5_diff, form3_diff, gf5_diff, ga5_diff")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS),
            ("flags", "passthrough", FLAGS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )

    Xtr_raw = df[train_mask]
    pre.fit(Xtr_raw)  # fit SOLO en train (Regla de Oro)
    feat_names = list(pre.get_feature_names_out())
    rep(f"\n(3) ColumnTransformer fit en train: {len(feat_names)} features de salida")
    rep(f"    continuas={len(CONTINUOUS)} (StandardScaler) | flags={len(FLAGS)} | "
        f"one-hot Division={len(feat_names)-len(CONTINUOUS)-len(FLAGS)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    rep("\nSHAPES POR SPLIT:")
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name]
        X = pd.DataFrame(pre.transform(sub), columns=feat_names, index=sub.index)
        y = sub[TARGET].reset_index(drop=True)
        X = X.reset_index(drop=True)
        X.to_parquet(PROCESSED_DIR / f"X_{name}.parquet", index=False)
        y.to_frame(TARGET).to_parquet(PROCESSED_DIR / f"y_{name}.parquet", index=False)
        rep(f"  {name:<6} X={X.shape}  y={y.shape}  "
            f"(H/D/A = {(y=='H').mean():.3f}/{(y=='D').mean():.3f}/{(y=='A').mean():.3f})")

    # Referencia de trazabilidad e insumos para reproducibilidad.
    df[["match_id", "MatchDate", "Division", "split"]].to_parquet(
        PROCESSED_DIR / "index_ref.parquet", index=False)
    joblib.dump(pre, models_dir / "preprocessor.joblib")
    (PROCESSED_DIR / "feature_names.json").write_text(
        json.dumps(feat_names, indent=2), encoding="utf-8")

    rep("\n[guardado] X_/y_{train,val,test}.parquet, index_ref.parquet, "
        "feature_names.json, models/preprocessor.joblib")
    rep.save()


if __name__ == "__main__":
    main()
