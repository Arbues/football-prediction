# `docs/presentation/` — Presentación de la defensa (Beamer)

Diapositivas de la sustentación oral, en **Beamer** (LaTeX). 22 diapositivas, siguiendo la
estructura estándar de la guía del curso (Sección 7).

## Archivos

| Archivo | Qué es |
|---|---|
| `presentation.tex` | Fuente LaTeX de las diapositivas. |
| `presentation.pdf` | PDF compilado — **este es el entregable de exposición**. |

## Compilar

```bash
cd docs/presentation
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex
```

Las figuras se toman de `../../figures/` y `../../figures/modeling/`.

## Estructura (22 diapositivas)

Portada · Agenda · Motivación · Problema formal (X, Y) · Objetivos · Fuente de datos ·
EDA I y II · Pipeline de preprocesamiento · Limpieza/imputación · Ingeniería de features ·
Metodología y validación · Modelo A y B (ecuaciones) · Resultados globales · La trampa de
la accuracy · Visualizaciones (ROC, matriz de confusión) · Significancia (Wilcoxon) ·
Interpretabilidad (SHAP) · Discusión · Limitaciones/validez · Conclusiones + trabajo
futuro · Cierre con enlace al repositorio.

## Reparto de la defensa (21 min = 7 min × 3)

- **Arbués** — Analista de datos e introducción (diapositivas 1–11).
- **Sergio** — Científico de datos e implementación / fundamentos matemáticos (12–14).
- **Luis** — Director de validación, resultados y conclusiones (15–22).
