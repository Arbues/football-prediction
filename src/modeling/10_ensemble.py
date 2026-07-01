"""Ensambles: Soft-Voting y Stacking (la contribución del trabajo).

Bases diversas (Teorema del Jurado de Condorcet): un lineal (LogReg), un bagging
(Random Forest) y un boosting (XGBoost); fallan distinto, así que su consenso
mejora. Soft-Voting promedia probabilidades sobre los modelos ya ajustados
(congelados, sin refit -> sin fuga). Stacking entrena un meta-modelo (LogReg)
sobre las predicciones OUT-OF-FOLD de bases reconstruidas con TimeSeriesSplit.
"""
import joblib
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import _common as C

BASES = [("lr", "01_logreg"), ("rf", "03_random_forest"), ("xgb", "06_xgboost")]


def load_bases():
    return {short: joblib.load(C.MODELS / f"{name}.pkl") for short, name in BASES}


def fresh_for_stacking(loaded):
    """Reconstruye bases sin estado ajustado para OOF honesto dentro del Stacking.
    XGBoost: se fija n_estimators=best_iteration y se quita early_stopping (no hay
    eval_set dentro de la CV)."""
    from sklearn.base import clone

    xgb = loaded["xgb"]
    p = xgb.get_params()
    n_est = getattr(xgb, "best_iteration", None) or 400
    p.update(early_stopping_rounds=None, n_estimators=int(n_est))
    xgb_fresh = XGBClassifier(**p)
    return [("lr", clone(loaded["lr"])), ("rf", clone(loaded["rf"])),
            ("xgb", xgb_fresh)]


def run_voting(loaded, d):
    C.header("Soft-Voting (bases congeladas: LogReg + RF + XGBoost)")
    estimators = [(s, FrozenEstimator(loaded[s])) for s, _ in BASES]
    vc = VotingClassifier(estimators=estimators, voting="soft")
    with C.timer("fit(voting, sin refit)") as t:
        vc.fit(d["X_train"], d["y_train"])
    preamble = ["MODELO: Soft-Voting (promedio de probabilidades)",
                "bases congeladas (FrozenEstimator): LogReg, RandomForest, XGBoost",
                "sin refit de las bases -> sin fuga; el voting no aprende pesos"]
    C.finalize_val("10_voting", "Voting", "ensemble-final", vc, d, preamble,
                   "soft; bases=LogReg,RF,XGB", None, t["seconds"], "Soft-Voting")


def run_stacking(loaded, d):
    C.header("Stacking (meta-LogReg sobre OOF de LogReg + RF + XGBoost)")
    estimators = fresh_for_stacking(loaded)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                              random_state=C.SEED)
    # El OOF interno del stacker exige una PARTICIÓN (cross_val_predict no admite
    # TimeSeriesSplit). Se usa StratifiedKFold(5) DENTRO de train; la honestidad
    # temporal se mantiene en el nivel externo: val/test siguen intactos y futuros.
    inner_cv = StratifiedKFold(n_splits=5, shuffle=False)
    sc = StackingClassifier(estimators=estimators, final_estimator=meta,
                            stack_method="predict_proba", cv=inner_cv,
                            n_jobs=1, passthrough=False)
    with C.timer("fit(stacking, OOF TimeSeriesSplit)") as t:
        sc.fit(d["X_train"], d["y_train"])
    preamble = ["MODELO: Stacking (meta-modelo = LogReg sobre proba OOF de las bases)",
                "bases: LogReg, RandomForest, XGBoost (reconstruidas, OOF honesto)",
                "OOF interno StratifiedKFold(5); temporalidad garantizada en val/test",
                "meta con class_weight='balanced'"]
    C.finalize_val("10_stacking", "Stacking", "ensemble-final", sc, d, preamble,
                   "meta=LogReg; bases=LogReg,RF,XGB; inner_cv=SKF5", None,
                   t["seconds"], "Stacking")


def main():
    C.set_seeds()
    d = C.load_xy("raw")
    loaded = load_bases()
    run_voting(loaded, d)
    run_stacking(loaded, d)


if __name__ == "__main__":
    main()
