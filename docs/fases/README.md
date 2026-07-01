# `docs/fases/` — Bitácora de avances del equipo

Este directorio es la **memoria compartida** del proyecto. Su objetivo: que cuando un
integrante retome el trabajo de otro, entienda **qué se hizo, por qué, y cómo tomarlo
en cuenta** — sin tener que leer todo el código ni preguntar por WhatsApp.

## Archivos

| Archivo | Qué contiene | Quién escribe |
|---|---|---|
| `BITACORA.md` | Log cronológico de avances. Cada bloque de trabajo = 1 entrada. | Todos, al terminar cada sesión de trabajo |
| `referencia-papers.md` | Resumen de los 10 papers académicos (estado del arte). | Generado — mantiene Luis |
| `referencia-kernels.md` | Resumen de los 5 kernels de Kaggle (features/modelos). | Generado — mantiene Sergio |

## Regla de oro

> **Antes de empezar a trabajar**: lee la última entrada de `BITACORA.md`.
> **Al terminar de trabajar**: agrega una entrada nueva ARRIBA (orden cronológico inverso).

## Plantilla de entrada (copiar en `BITACORA.md`)

```markdown
## [YYYY-MM-DD HH:MM] — <Integrante> — Fase X: <título corto>

**Qué hice:**
- ...

**Archivos tocados:**
- `ruta/al/archivo` — qué cambió y por qué

**Decisiones tomadas (y por qué):**
- ...

**Para el siguiente que trabaje acá:**
- Qué tomar en cuenta, qué NO tocar, qué quedó pendiente

**Bloqueos / dudas abiertas:**
- ...
```

## Convención de commits (para que el historial cuente lo mismo que la bitácora)

`git` se evalúa: el profesor mira el historial de contribuciones de cada integrante.
- Commits incrementales, nunca un commit masivo al final.
- Conventional commits: `feat(eda): ...`, `feat(preprocessing): ...`, `docs(paper): ...`, `fix(modeling): ...`
- Cada integrante commitea con SU cuenta.
