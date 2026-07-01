"""Utilidades compartidas para el preprocesamiento (Fase 2).

Extiende las convenciones del EDA (rutas, Tee para reportes) con lo propio de
esta fase: directorio de datos procesados, cortes del split temporal, ligas
top-5, ventana de las medias moviles y normalizacion de nombres de club para el
join con EloRatings. Cada script del pipeline es autonomo pero no duplica estas
definiciones.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42

# --- Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

MATCHES_CSV = RAW_DIR / "Matches.csv"
ELO_CSV = RAW_DIR / "EloRatings.csv"

TARGET = "FTResult"
TARGET_ORDER = ["H", "D", "A"]

# --- Contexto e identificadores pre-partido ---
CONTEXT_COLS = ["Division", "MatchDate", "HomeTeam", "AwayTeam"]

# Numericas seguras que ya vienen en el dataset (conocidas ANTES del partido).
SAFE_PRE_NUMERIC = [
    "HomeElo", "AwayElo",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
]

# Marcador final: es leakage como INPUT del partido actual, pero materia prima
# legitima para reconstruir features historicas de partidos ANTERIORES
# (goles rolling, h2h). Se conserva solo hasta la ingenieria de features.
GOALS_COLS = ["FTHome", "FTAway"]

# LISTA NEGRA dura: estadisticas que solo existen DESPUES del partido.
LEAKAGE_POST = [
    "HTHome", "HTAway", "HTResult",
    "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
    "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners",
    "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
]

# Cuotas de mercado y cierres de bookmaker: se descartan por decision de diseno
# (version "honesta" sin la muleta del sabio). MatchTime: 57% nulo, sin valor.
ODDS_COLS = [
    "OddHome", "OddDraw", "OddAway",
    "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25",
    "HandiSize", "HandiHome", "HandiAway",
]
CLOSING_COLS = ["C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB"]
DROP_MISC = ["MatchTime"]

# --- Ligas top-5 europeas (codigos de la columna Division) ---
# E0 Premier League, SP1 La Liga, I1 Serie A, D1 Bundesliga, F1 Ligue 1.
TOP5_LEAGUES = {"E0", "SP1", "I1", "D1", "F1"}

# --- Split temporal por fecha (holdout final; ver 04) ---
# train: temporadas <= 2021  | val: 2022-2023  | test: 2024-2025.
TRAIN_END_YEAR = 2021
VAL_YEARS = (2022, 2023)
TEST_YEARS = (2024, 2025)

# --- Ventana de las medias moviles (coincide con Form5 y con la literatura) ---
ROLL_WINDOW = 5

# --- Artefactos intermedios del pipeline (data/processed) ---
CLEAN_PARQUET = PROCESSED_DIR / "_matches_clean.parquet"
ELO_PARQUET = PROCESSED_DIR / "_matches_elo.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "_matches_features.parquet"
SPLIT_PARQUET = PROCESSED_DIR / "_matches_split.parquet"

# Alias conocidos Matches -> EloRatings (los de mayor volumen; el resto de
# discrepancias caen a imputacion por mediana de liga + flag).
CLUB_ALIASES = {
    "Paris SG": "Paris Saint-Germain",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Sp Lisbon": "Sporting CP",
    "Nott'm Forest": "Nottingham Forest",
    "Inter": "Internazionale",
}


def normalize_club(name) -> str:
    """Normaliza un nombre de club para el join (strip, colapsa espacios).

    Corrige discrepancias triviales como 'Ajax ' vs 'Ajax'. Los alias reales
    (p. ej. 'Man United' vs 'Manchester United') se resuelven aparte con
    CLUB_ALIASES antes de normalizar.
    """
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name).strip()


def load_matches() -> pd.DataFrame:
    return pd.read_csv(MATCHES_CSV, low_memory=False, parse_dates=["MatchDate"])


def load_elo() -> pd.DataFrame:
    return pd.read_csv(ELO_CSV, parse_dates=["date"])


class Tee:
    """Escribe en un .txt de results/ mientras imprime en stdout."""

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
    return Tee(RESULTS_DIR / filename)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
