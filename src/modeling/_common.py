"""Núcleo compartido de la Fase 3 (Modelado).

Centraliza semillas, carga de la matriz procesada, el esquema de validación
temporal, el cálculo de métricas (rúbrica 3.9) y el volcado de reportes/figuras.
Cada script `NN_*.py` importa de aquí para no repetir tubería.

Convención de etiquetas: se codifica FTResult con un LabelEncoder ajustado en
train. `le.classes_` queda alfabético -> A=0, D=1, H=2. Para el REPORTE se usa
siempre el orden futbolero H, D, A (fácil > difícil), fijado en CLASSES_ENC.
"""
from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin display: guarda PNG y no abre ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Fija las tres fuentes de aleatoriedad relevantes."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Rutas (ancladas al paquete, no al cwd)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]  # .../football-prediction
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "modeling"
FIGURES = ROOT / "figures" / "modeling"
MODELS = ROOT / "models"
for _d in (RESULTS, FIGURES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

GRID_CSV = RESULTS / "grid_search_results.csv"

# --------------------------------------------------------------------------- #
# Esquema de clases
# --------------------------------------------------------------------------- #
CLASSES_NAME = ["H", "D", "A"]          # orden de reporte (local, empate, visita)
_LE = LabelEncoder().fit(np.array(["A", "D", "H"]))  # A=0, D=1, H=2
CLASSES_ENC = list(_LE.transform(CLASSES_NAME))       # -> [2, 1, 0]
SORTED_ENC = [0, 1, 2]                                 # orden de columnas de proba


def encode(y: np.ndarray) -> np.ndarray:
    return _LE.transform(np.asarray(y).ravel())


def decode(y_enc: np.ndarray) -> np.ndarray:
    return _LE.inverse_transform(np.asarray(y_enc).ravel())


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
def load_xy(variant: str = "raw"):
    """Devuelve X_{train,val,test} (DataFrame) e y_{...} codificado (int).

    variant='raw'  -> matriz cruda con nombres (62 features).
    variant='pca'  -> variante PCA (13 PC + flags + one-hot).
    """
    suffix = "" if variant == "raw" else "_pca"
    out = {}
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = pd.read_parquet(PROC / f"X_{split}{suffix}.parquet")
        y_raw = pd.read_parquet(PROC / f"y_{split}.parquet").iloc[:, 0].to_numpy()
        out[f"y_{split}"] = encode(y_raw)
    return out


def load_index_ref() -> pd.DataFrame:
    """match_id, MatchDate, Division, split — para bloques temporales del Wilcoxon."""
    return pd.read_parquet(PROC / "index_ref.parquet")


def feature_names() -> list[str]:
    with open(PROC / "feature_names.json") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Validación
# --------------------------------------------------------------------------- #
def get_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """CV temporal expansiva DENTRO de train (no K-Fold aleatorio)."""
    return TimeSeriesSplit(n_splits=n_splits)


SCORING = "f1_macro"


# --------------------------------------------------------------------------- #
# Métricas (rúbrica 3.9)
# --------------------------------------------------------------------------- #
def compute_metrics(y_true_enc, y_pred_enc, y_proba=None) -> dict:
    """Precision/Recall/F1 (macro, weighted y por clase), Kappa, ROC-AUC OvR,
    log-loss y accuracy. y_proba con columnas en orden [A, D, H] (0,1,2)."""
    # ravel defensivo: CatBoost.predict() en MultiClass devuelve (n,1) y rompería
    # el broadcasting de la accuracy.
    y_true_enc = np.asarray(y_true_enc).ravel()
    y_pred_enc = np.asarray(y_pred_enc).ravel()
    acc = float((y_true_enc == y_pred_enc).mean())
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, average="macro", zero_division=0
    )
    p_wtd, r_wtd, f_wtd, _ = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, average="weighted", zero_division=0
    )
    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
        y_true_enc, y_pred_enc, labels=CLASSES_ENC, zero_division=0
    )
    kappa = float(cohen_kappa_score(y_true_enc, y_pred_enc))

    metrics = {
        "accuracy": acc,
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f_mac),
        "precision_weighted": float(p_wtd),
        "recall_weighted": float(r_wtd),
        "f1_weighted": float(f_wtd),
        "kappa": kappa,
        "per_class": {
            cls: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f_c[i]),
                "support": int(sup_c[i]),
            }
            for i, cls in enumerate(CLASSES_NAME)
        },
    }
    if y_proba is not None:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_enc, y_proba, multi_class="ovr",
                          average="macro", labels=SORTED_ENC)
        )
        metrics["log_loss"] = float(log_loss(y_true_enc, y_proba, labels=SORTED_ENC))
    return metrics


def report_text(y_true_enc, y_pred_enc, y_proba=None) -> str:
    """Bloque de texto con classification_report + métricas agregadas."""
    y_true_enc = np.asarray(y_true_enc).ravel()
    y_pred_enc = np.asarray(y_pred_enc).ravel()
    rep = classification_report(
        y_true_enc, y_pred_enc, labels=CLASSES_ENC,
        target_names=CLASSES_NAME, digits=4, zero_division=0,
    )
    m = compute_metrics(y_true_enc, y_pred_enc, y_proba)
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=CLASSES_ENC)
    lines = [rep, ""]
    lines.append(f"accuracy           : {m['accuracy']:.4f}")
    lines.append(f"f1_macro           : {m['f1_macro']:.4f}")
    lines.append(f"f1_weighted        : {m['f1_weighted']:.4f}")
    lines.append(f"cohen_kappa        : {m['kappa']:.4f}")
    if "roc_auc_ovr_macro" in m:
        lines.append(f"roc_auc_ovr_macro  : {m['roc_auc_ovr_macro']:.4f}")
        lines.append(f"log_loss           : {m['log_loss']:.4f}")
    lines.append("")
    lines.append("matriz de confusion (filas=verdad, cols=pred; orden H,D,A):")
    lines.append("        " + "  ".join(f"{c:>6}" for c in CLASSES_NAME))
    for i, c in enumerate(CLASSES_NAME):
        lines.append(f"  {c:>4}  " + "  ".join(f"{v:>6}" for v in cm[i]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Volcado de reportes / registro
# --------------------------------------------------------------------------- #
def save_report(name: str, text: str) -> Path:
    path = RESULTS / f"{name}.txt"
    path.write_text(text)
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def append_grid_row(row: dict) -> None:
    """Acumula una fila por modelo en grid_search_results.csv."""
    df_new = pd.DataFrame([row])
    if GRID_CSV.exists():
        df = pd.read_csv(GRID_CSV)
        df = df[df["model"] != row["model"]]  # idempotente: reemplaza si ya existía
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(GRID_CSV, index=False)


def save_model(obj, name: str) -> Path:
    import joblib

    path = MODELS / f"{name}.pkl"
    joblib.dump(obj, path)
    print(f"  -> {path.relative_to(ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def plot_confusion(y_true_enc, y_pred_enc, name: str, title: str) -> Path:
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=CLASSES_ENC)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), CLASSES_NAME)
    ax.set_yticks(range(3), CLASSES_NAME)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIGURES / f"cm_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_ovr(y_true_enc, y_proba, name: str, title: str) -> Path:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for cls in CLASSES_NAME:
        k = _LE.transform([cls])[0]
        y_bin = (y_true_enc == k).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, k])
        auc = roc_auc_score(y_bin, y_proba[:, k])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = FIGURES / f"roc_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_validation_curve(param_range, train_scores, cv_scores, param_name: str,
                          name: str, title: str, logx: bool = False) -> Path:
    """Curva train vs CV (f1_macro) para el barrido de un hiperparámetro."""
    tr_mean, tr_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    cv_mean, cv_std = np.mean(cv_scores, axis=1), np.std(cv_scores, axis=1)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.asarray(param_range, dtype=float)
    ax.plot(x, tr_mean, "o-", color="tab:blue", label="train")
    ax.fill_between(x, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="tab:blue")
    ax.plot(x, cv_mean, "o-", color="tab:orange", label="CV (TimeSeriesSplit)")
    ax.fill_between(x, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="tab:orange")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(param_name)
    ax.set_ylabel("f1_macro")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = FIGURES / f"valcurve_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Utilidades varias
# --------------------------------------------------------------------------- #
def finalize_val(name, model_label, family, clf, d, preamble, best_params_str,
                 cv_f1, fit_seconds, title, roc=True, save_pickle=True):
    """Evalúa en val 2022-2023, vuelca reporte + figuras, guarda modelo y
    registra la fila del grid. Devuelve el dict de métricas de val."""
    proba = clf.predict_proba(d["X_val"])
    pred = clf.predict(d["X_val"])
    m = compute_metrics(d["y_val"], pred, proba)
    txt = list(preamble) + ["", "== Validación (val 2022-2023) ==",
                            report_text(d["y_val"], pred, proba)]
    save_report(name, "\n".join(txt))
    plot_confusion(d["y_val"], pred, name, title)
    if roc:
        plot_roc_ovr(d["y_val"], proba, name, f"{title} — ROC OvR")
    if save_pickle:
        save_model(clf, name)
    append_grid_row({
        "model": model_label, "family": family, "best_params": best_params_str,
        "cv_f1_macro": (round(cv_f1, 4) if cv_f1 is not None else np.nan),
        "val_f1_macro": m["f1_macro"], "val_accuracy": m["accuracy"],
        "val_recall_D": m["per_class"]["D"]["recall"],
        "fit_seconds": round(fit_seconds, 2),
    })
    print(f"  val f1_macro={m['f1_macro']:.4f}  acc={m['accuracy']:.4f}  "
          f"recall_D={m['per_class']['D']['recall']:.4f}")
    return m


def validation_sweep(estimator, param_name, param_range, X, y, name, title,
                     logx: bool = False):
    """Barre un hiperparámetro con validation_curve sobre TimeSeriesSplit,
    grafica train vs CV y devuelve (valor*, cv_mean, train_scores, cv_scores)."""
    from sklearn.model_selection import validation_curve

    train_sc, cv_sc = validation_curve(
        estimator, X, y, param_name=param_name, param_range=list(param_range),
        cv=get_cv(), scoring=SCORING, n_jobs=-1,
    )
    plot_validation_curve(param_range, train_sc, cv_sc, param_name, name, title, logx)
    cv_mean = cv_sc.mean(axis=1)
    best_i = int(np.argmax(cv_mean))
    return param_range[best_i], cv_mean, train_sc, cv_sc


@contextmanager
def timer(label: str = "fit"):
    t0 = time.perf_counter()
    box = {}
    yield box
    box["seconds"] = time.perf_counter() - t0
    print(f"  [{label}] {box['seconds']:.1f}s")


def header(name: str) -> None:
    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)
