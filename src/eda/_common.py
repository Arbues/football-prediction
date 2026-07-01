"""Utilidades compartidas para los scripts de EDA (Fase 1).

Centraliza rutas, carga del dataset y la taxonomia de columnas (pre-partido
seguras vs. fuga de datos post-partido) para que cada script del EDA sea
autonomo pero no duplique estas definiciones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# --- Rutas del proyecto (resueltas desde este archivo, no desde el cwd) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

MATCHES_CSV = RAW_DIR / "Matches.csv"
ELO_CSV = RAW_DIR / "EloRatings.csv"

# --- Taxonomia de columnas (eje anti-leakage del EDA) ---

# Identificadores / contexto pre-partido.
CONTEXT_COLS = ["Division", "MatchDate", "MatchTime", "HomeTeam", "AwayTeam"]

# Numericas seguras conocidas ANTES del partido -> input valido para el modelo.
SAFE_PRE_NUMERIC = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
]

# Cuotas de mercado: pre-partido, pero "leak del sabio" (el bookmaker ya sabe).
# Se analizan aparte para decidir modelar con y sin ellas.
ODDS_COLS = [
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway",
]

# Columnas de cierre/derivadas de casa de apuestas (poco documentadas, ~51% nulas).
CLOSING_COLS = ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]

# Objetivo.
TARGET = "FTResult"

# LISTA NEGRA: estadisticas que solo existen DESPUES del partido -> data leakage.
LEAKAGE_POST = [
    "FTHome", "FTAway",           # marcador final (deriva el target)
    "HTHome", "HTAway", "HTResult",  # medio tiempo
    "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
    "HomeFouls", "AwayFouls",
    "HomeCorners", "AwayCorners",
    "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
]

TARGET_ORDER = ["H", "D", "A"]


def load_matches() -> pd.DataFrame:
    """Carga Matches.csv con tipos correctos y MatchDate como datetime."""
    df = pd.read_csv(
        MATCHES_CSV,
        low_memory=False,
        parse_dates=["MatchDate"],
    )
    return df


def load_elo() -> pd.DataFrame:
    """Carga EloRatings.csv con date como datetime."""
    return pd.read_csv(ELO_CSV, parse_dates=["date"])


class Tee:
    """Escribe simultaneamente en un archivo de texto y acumula lineas.

    Permite que cada script deje un .txt legible en results/ mientras
    tambien imprime en stdout para inspeccion inmediata.
    """

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self._lines: list[str] = []

    def __call__(self, *args) -> None:
        line = " ".join(str(a) for a in args)
        self._lines.append(line)
        print(line)

    def save(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
        print(f"\n[guardado] {self.out_path}")


def new_report(filename: str) -> Tee:
    """Crea un reporte de texto en results/<filename>."""
    return Tee(RESULTS_DIR / filename)


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
