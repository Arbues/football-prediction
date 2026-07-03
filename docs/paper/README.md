# `docs/paper/` — Artículo científico (IEEE)

Entregable 1 del proyecto: el manuscrito en formato **IEEE Conference** (doble columna,
8 páginas).

**Título:** *Clasificación Multiclase de Resultados de Fútbol a Escala Multi-Liga con
Validación Temporal y Análisis de Significancia Estadística*

**Autores:** Arbués Enrique Pérez Villegas · Luis Andre Trujillo Serva · Sergio Sebastian
Pezo Jimenez — Escuela de Ciencia de la Computación, UNI.

## Archivos

| Archivo | Qué es |
|---|---|
| `paper.tex` | Fuente LaTeX del artículo (único archivo a editar). |
| `paper.pdf` | PDF compilado — **este es el entregable**. |
| `IEEEtran.cls` | Plantilla oficial IEEE (no editar). |
| `guide/` | Plantilla y ejemplo de referencia de IEEE (`conference_101719.tex`). |
| `paper.aux`, `paper.log`, `paper.out`, `paper.fls`, `paper.fdb_latexmk` | Artefactos de compilación (regenerables). |

## Compilar

```bash
cd docs/paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex   # segunda pasada: referencias cruzadas
```

Requiere una distribución LaTeX (TeX Live o MiKTeX). El documento usa Computer Modern
(no requiere las fuentes Times de psnfss). La segunda pasada resuelve las referencias a
figuras, tablas y bibliografía.

## Verificación rápida

- Debe compilar sin `??` en el `.log` (referencias resueltas).
- Extensión: 8 páginas (rango exigido: 8–10).
- Las figuras se toman de `../../figures/` (ver `\graphicspath` en el preámbulo).

Todas las cifras del artículo se regeneran desde los scripts de `src/` y coinciden con
`results/modeling/`.
