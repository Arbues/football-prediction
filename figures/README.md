# `figures/` — Gráficos científicos

Figuras exportadas en PNG de alta resolución, usadas en el artículo IEEE y la presentación.
Se regeneran al ejecutar los scripts de `src/`.

## Raíz — Figuras del EDA y preprocesamiento

Distribuciones (`dist_*`), balance de clases (`class_balance`), correlación
(`correlation_matrix`, `correlation_spearman`), outliers (`boxplots_*`,
`mahalanobis_outliers`), multicolinealidad (`vif_barplot`), PCA (`pca_cumvar`, `pca_scree`),
Elo/forma por resultado (`elo_by_result`, `form_by_result`) y análisis del empate
(`draw_analysis`).

## `modeling/` — Figuras del modelado

Curvas ROC, matrices de confusión, comparativas de modelos, importancia por permutación y
resúmenes SHAP (una por modelo/experimento). La figura `11_model_comparison.png` es la
comparativa maestra de F1 macro en validación y prueba.

> Las figuras referenciadas por el paper se resuelven vía `\graphicspath` a esta carpeta
> (ver `docs/paper/paper.tex`).
