# Referencia — 5 kernels de Kaggle (feature engineering + modelado)

> Insumo directo para **Fase 2 (Preprocessing/FE — Arbués)** y **Fase 3 (Modeling — Sergio)**.
> Fuente: `pc5/EXPO/kagles anteriores/`. Mismo dataset que usamos: `club-football-match-data-2000-2025`.

---

## 🔴 Conclusión más importante

**Ningún kernel resuelve nuestro problema.** De los 5: 2 son EDA puro, 1 modela un problema DISTINTO (binario, ganador de eliminatoria UCL — no FTResult H/D/A), y 2 son basura (solo listados de directorio de Kaggle). **No hay benchmark de Acc/F1/logloss multiclase que podamos citar como referencia directa — lo construimos nosotros.** Pero sí hay patrones de código y anti-patrones muy útiles.

---

## Esquema real del dataset (confirmado en los kernels)

`Matches.csv` = **226,755 filas × 42 columnas**. *(Nota: PLANNING.md dice N=230557, D=48 — verificar en Fase 1 cuál es exacto; probable diferencia de versión del CSV.)*

**Target** `FTResult`: **H=101,225 (44.6%) / A=65,369 (28.8%) / D=60,158 (26.5%)**. Fuerte ventaja de local; empate = clase minoritaria y difícil.

### ✅ Features PRE-partido (seguras de usar)
`Division, MatchDate, MatchTime, HomeTeam, AwayTeam, HomeElo, AwayElo, Form3Home, Form5Home, Form3Away, Form5Away`, y odds de mercado (`OddHome/OddDraw/OddAway, MaxHome/MaxDraw/MaxAway, Over25/Under25/MaxOver25/MaxUnder25, HandiSize/HandiHome/HandiAway`).

### 🚫 LISTA NEGRA — features POST-partido (DATA LEAKAGE, prohibidas como input)
`FTHome, FTAway` (target derivado), `FTResult` (=target), `HTHome, HTAway, HTResult, HomeShots, AwayShots, HomeTarget, AwayTarget, HomeFouls, AwayFouls, HomeCorners, AwayCorners, HomeYellow, AwayYellow, HomeRed, AwayRed`.
> Todas son estadísticas que solo existen DESPUÉS de jugado el partido. Si entran al modelo = fuga garantizada. Los papers y kernels que reportan Acc>70% en 3 clases suelen estar contaminados por esto.

### ⚠️ Nulos importantes
- `HomeElo/AwayElo`: ~78k nulos (**35%** — solo hay Elo desde ~2006). NO imputar con 0.
- Stats de partido: ~50% nulos (irrelevante, van a la lista negra).
- Odds: ~1–12% nulos. `FTResult`: solo 3 nulos.

---

## Fichas por kernel

### 1. `club-football-match-data-2000-2025.txt` — EDA puro
- 14 celdas. `df.info()`, `df.isna().sum()`, heatmap de nulos, boxplots, histogramas KDE, barras Plotly. **Sin FE, sin modelos.**
- **Útil**: mejor referencia de esquema y conteo de nulos por columna. Base para el EDA de Arbués.

### 2. `football-match-data-analysis.txt` — EDA puro (Elo + Matches)
- 30 celdas, bien documentado. Agregaciones para gráficos, sin features derivadas reales.
- **Anti-patrones a NO copiar**:
  - `matches.dropna(thresh=len(matches)*0.5, axis=1)` → dropea ciego columnas con >50% nulos.
  - `fillna(0)` en numéricos (incluido Elo) → **grave**, mete outliers artificiales.
- **EDA útil (cifras)**: Elo medio por país (ESP 1643.6 > ENG 1596.4 > GER 1582.7 > ITA 1553.1 > FRA 1543.6); odds medias OddHome 2.43 < OddDraw 3.56 < OddAway 3.97 (el mercado favorece al local, coherente con el sesgo H); visitantes reciben más tarjetas. → justifica un feature explícito de **ventaja de localía**.

### 3. `hack-hack.txt` — ⭐ único con ML (problema distinto: binario UCL)
- **Target**: `label = 1 if winner==team_1 else 0` (ganador de eliminatoria UCL, dataset externo `ucl123`). El dataset de fútbol se usa SOLO como fuente de Elo.
- **6 features, todas pre-partido** (esto es lo valioso): `elo_team_1`, `elo_team_2`, `elo_diff`, `team_1_win_rate`, `team_2_win_rate`, `win_rate_diff`.
- **⭐ Patrón anti-leakage a COPIAR**: `get_latest_elo(team, date)` filtra `elo_df[(club==team) & (date <= match_date)].sort_values('date').iloc[-1]` → toma el Elo más reciente ANTES del partido. Este `date <= match_date` es exactamente cómo debemos reconstruir Elo/forma/h2h/rest.
- **Win-rate incremental**: `defaultdict` recorriendo partidos en orden cronológico (acumulado hasta ese momento). Buena base, pero ojo: calcularlo estrictamente con partidos ANTERIORES a cada fila.
- **Armonización de nombres**: `elo_df['club'].replace({"Man United":"Manchester United", "Man City":"Manchester City", "PSG":"Paris Saint-Germain"})` → copiar el diccionario para casar nombres entre `EloRatings.csv` y `Matches.csv`.
- **Modelo**: solo XGBClassifier + **Optuna** (25 trials). Espacio: `max_depth[3,6], lr[0.01,0.3], n_estimators[100,300], subsample[0.5,1.0], colsample_bytree[0.5,1.0]`. Buen punto de partida para nuestro XGB/LGBM.
- **🚫 Errores a NO imitar**: `train_test_split` aleatorio sobre datos temporales; fecha fija por temporada para buscar Elo (mezcla rondas); muestra minúscula (~195 filas → Acc val 0.79–0.87 no fiable); código muerto y un bug (suma Elo como si fueran goles).

### 4 y 5. `notebook4ab68711db.txt` y `_v2.txt` — 🗑️ basura
- Solo la celda plantilla de Kaggle (`os.walk('/kaggle/input')` imprimiendo rutas). No cargan el dataset, no modelan. **Descartar.**

---

## Recomendaciones accionables para Fase 2 y 3

### Feature engineering (todo estrictamente con partidos ANTERIORES a cada fila, ordenando por `MatchDate`)
- Ya vienen en el dataset: `HomeElo, AwayElo, Form3Home, Form5Home, Form3Away, Form5Away`. **Verificar en Fase 1 qué representan exactamente los `Form*`** (valen 0–13, parecen "puntos en últimos N partidos" — confirmar que son pre-partido).
- Derivar: `elo_diff = HomeElo - AwayElo`, `form_diff = Form5Home - Form5Away`.
- Construir (ningún kernel lo hizo → oportunidad, no hay qué copiar): rolling de goles a favor/en contra últimos 5, head-to-head histórico, `rest_days` (días desde último partido por equipo), racha de forma bien calculada, flag de localía explícito.
- Decidir con/sin **odds de mercado**: son casi un "leak del sabio" (el bookmaker ya sabe mucho). Modelar **con y sin odds** para medir contribución real y tener una versión "honesta".

### Nulos
- Elo (~35% nulo, pre-2006): NO `fillna(0)`. Opciones: filtrar a partir del año con Elo, o imputar con cuidado (mediana por liga/época). Decidir y documentar.
- NO usar `dropna(thresh=50%)` ciego.

### Validación
- **NO** `train_test_split` aleatorio. Usar **split temporal** por `MatchDate` (train años tempranos → test años recientes) o `TimeSeriesSplit`. Estratificar/reportar por clase dado el desbalance. La clase **D (empate) será la difícil** — reportar su F1 aparte.

### Modelos (Fase 3)
- Nuestro plan (LogReg/SVM/RF/XGBoost/LightGBM/CatBoost) es más completo que cualquier kernel. Reutilizar el espacio Optuna del #3 como base para los boosting.
