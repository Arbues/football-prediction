# Guion de sustentación — Predicción de resultados de fútbol (CC442)

> **Proyecto:** Comparación empírica de clasificadores para predecir `FTResult` (H/D/A)
> sobre 230 554 partidos multi-liga (2000–2025), sin cuotas y con validación temporal.
> **Duración total:** 21 min (7 min por expositor). **No exceder.**
>
> **Expositores y roles (según rúbrica, Sección 6):**
> - **Arbués Enrique Pérez Villegas** — Analista de Datos e Introducción (slides 1–11)
> - **Sergio Sebastian Pezo Jimenez** — Científico de Datos e Implementación (slides 12–14)
> - **Luis Andre Trujillo Serva** — Director de Validación y Conclusiones (slides 15–22)
>
> **Regla de oro para la defensa:** el número que se defiende NO es la accuracy (~44 %),
> es el **F1 macro (0,42)** y la **significancia estadística**. Quien pregunte por qué la
> accuracy es "baja", ahí está la fortaleza del trabajo, no la debilidad.

---

## BLOQUE 1 — Arbués (7 min) · Datos, problema y EDA

### [Slide 1 · Portada] (~20 s)
"Buenos días. Somos el grupo de Minería de Datos y hoy presentamos un análisis comparativo
de modelos de aprendizaje supervisado para predecir el resultado de partidos de fútbol.
Soy Arbués Pérez y abro con el problema y los datos; luego Sergio detallará la metodología
y Luis cerrará con los resultados y su validación estadística."

### [Slide 3 · Motivación y contexto] (~50 s)
"Predecir el resultado de un partido —local, empate o visitante— es un problema clásico de
la analítica deportiva, con aplicaciones en gestión de clubes, mercados de apuestas y
generación de contenido. Es difícil por dos razones: el fútbol es de marcador bajo, así que
el azar pesa mucho; y el empate es una clase minoritaria y esquiva. Nuestra pregunta no es
solo 'qué modelo gana', sino 'cuánto se puede predecir honestamente, sin hacer trampa'."

### [Slide 4 · Definición formal del problema] (~50 s)
"Formalmente es una clasificación supervisada de tres clases. Dado un vector de
características X, construido **solo con información previa al partido**, buscamos una función
g que asigne la etiqueta: H victoria local, D empate, A victoria visitante. La restricción
clave es temporal: nada de lo que ocurre durante o después del partido puede entrar en X."

### [Slide 5 · Objetivos] (~40 s)
"El objetivo general es diseñar un pipeline reproducible y sin fuga de datos que compare
nueve familias de modelos. Los específicos: establecer líneas base, entrenar los modelos del
curso con su fundamentación matemática, añadir gradient boosting moderno y un ensamble, y
—esto es central— reportar métricas por clase y una prueba de significancia."

### [Slide 6 · Descripción del dataset] (~50 s)
"Usamos el dataset público *Club Football Match Data*: 230 554 partidos de múltiples ligas
entre 2000 y 2025, con 48 columnas originales que incluyen ratings Elo, forma reciente,
estadísticas del partido y cuotas de mercado. La variable objetivo es el resultado final. Es
un conjunto de gran escala y multi-liga, lo que lo distingue de la mayoría de la literatura,
que trabaja con una sola liga y unos pocos miles de partidos."

### [Slide 7 · EDA parte I — distribución y desbalance] (~1 min)
"El análisis exploratorio nos dio cuatro hallazgos que condicionan todo lo demás. El primero:
hay desbalance moderado. El local gana el 44,6 %, el visitante el 28,9 % y el empate solo el
26,5 %. La razón de desbalance es 1,68. Esto tiene una consecuencia inmediata que Luis
retomará: **optimizar la accuracy sería un error**, porque un modelo que siempre diga 'local'
ya acierta el 44 %. Por eso desde el inicio optimizamos F1 macro."

### [Slide 8 · EDA parte II — Elo y correlaciones] (~1 min)
"Segundo hallazgo: la variable más informativa es la diferencia de Elo. Al ordenar los
partidos por ella, los tres resultados se separan de forma monótona, pero —y esto es clave—
el empate siempre queda en el medio, sin una región propia. Eso anticipa una **frontera de
decisión no lineal**, que favorece a árboles y kernels. Tercero: hay multicolinealidad
exacta; las variables de diferencia son combinaciones lineales de sus componentes, con
factores de inflación de varianza infinitos. Y cuarto: el Elo falta de forma estructural en
ligas no europeas, y hay deriva temporal, con una caída de la ventaja local durante la
pandemia."

### [Slide 9–10 · Pipeline y limpieza / imputación] (~1 min)
"Con eso definimos el preprocesamiento. Descartamos toda columna posterior al partido
—tiros, tarjetas, marcador— porque son fuga directa, y también las cuotas de mercado, por
decisión de diseño: queríamos medir el poder predictivo honesto, sin la muleta del
bookmaker. El Elo lo reconstruimos con una unión temporal que solo mira el pasado, y las
ausencias las marcamos con una bandera explícita en vez de imputar con cero, que
introduciría equipos artificialmente promedio."

### [Slide 11 · Feature engineering y split temporal] (~50 s)
"Construimos features de contraste —diferencia de Elo, de forma, de goles recientes,
historial directo, descanso, rachas— todas calculadas solo con partidos anteriores. La
estandarización se ajustó **solo en entrenamiento**. Y la partición es estrictamente
temporal: entrenamos hasta 2021, validamos con 2022–2023 y probamos con 2024–2025. Esto no
es un capricho: un K-fold aleatorio metería partidos del futuro en el entrenamiento y
rompería la validez del experimento. Con esto, cedo la palabra a Sergio, que explicará cómo
convertimos estos datos en modelos."

---

## BLOQUE 2 — Sergio (7 min) · Metodología, modelos e implementación

### [Slide 12 · Arquitectura de la metodología] (~1 min)
"Gracias, Arbués. Sobre esa matriz de 62 características entrenamos nueve familias de
modelos. El protocolo de validación tiene dos niveles. Para elegir hiperparámetros usamos
`TimeSeriesSplit` **dentro** del entrenamiento: cada pliegue entrena con el pasado y valida
con el futuro inmediato. Con el conjunto de validación de 2022–2023 elegimos entre familias,
y el conjunto de prueba de 2024–2025 queda intacto hasta el final. La métrica que optimizamos
es el F1 macro, que promedia el F1 de las tres clases con igual peso, de modo que el empate
cuenta tanto como las clases fáciles."

### [Slide 13 · Modelo A — fundamentos] (~1 min 40 s)
"Déjenme anclar los modelos a su matemática. Empezamos por líneas base: un clasificador
ingenuo y Naive Bayes, cuyo supuesto de independencia lo penaliza porque nuestras variables
están correlacionadas. Luego los del curso. La regresión logística multinomial usa softmax
con regularización L2, que es exactamente la regularización de Tikhonov: vuelve invertible la
matriz singular que causaba la colinealidad. El SVM maximiza el margen, y con kernel de base
radial —la exponencial de la distancia al cuadrado— habilita fronteras no lineales, que es lo
que el empate pide. El árbol de decisión parte el espacio maximizando la ganancia de
información, es decir, la reducción de entropía."

### [Slide 14 · Modelo B — ensambles y boosting] (~1 min 40 s)
"Sobre esa base construimos los ensambles. El Random Forest promedia muchos árboles sobre
réplicas bootstrap; la clave es que, al restringir cada partición a un subconjunto aleatorio
de variables, **descorrelaciona** los árboles y así reduce la varianza del promedio. AdaBoost,
en cambio, reduce el sesgo, combinando aprendices débiles y subiendo el peso de los ejemplos
mal clasificados. El paso moderno es el gradient boosting: XGBoost, LightGBM y CatBoost
ajustan en cada ronda un árbol al gradiente de la pérdida, con una aproximación de segundo
orden y regularización. Y la contribución del trabajo es el ensamble final: un stacking, donde
un meta-modelo aprende a combinar las predicciones de un lineal, un bagging y un boosting.
Por el teorema del jurado de Condorcet, si los modelos son diversos y mejores que el azar, su
consenso mejora."

### [Slide · Optimización e implementación] (~1 min 20 s)
"Dos puntos de rigor. Primero, los hiperparámetros no se eligen a dedo: cada uno se barre con
una curva de validación y se elige donde la validación deja de mejorar, controlando el
compromiso sesgo–varianza. Segundo, reproducibilidad total: fijamos la semilla 42 en Python,
NumPy y en cada estimador, búsqueda y pliegue. El código es modular, un script por modelo con
un núcleo común, y el boosting corre en GPU: XGBoost y CatBoost en CUDA. Todo esto está en un
cuaderno que ejecuta de principio a fin sin errores. Con la metodología clara, le paso la
palabra a Luis para los resultados."

---

## BLOQUE 3 — Luis (7 min) · Resultados, significancia y conclusiones

### [Slide 15 · Resultados experimentales globales] (~1 min 20 s)
"Gracias, Sergio. Esta es la tabla maestra en el conjunto de prueba, ordenada por F1 macro.
El modelo propuesto, el stacking, alcanza 0,419, empatado en la cima con LightGBM. Todo el
pelotón competente se agrupa alrededor de 0,42 de F1 macro y 0,44 de accuracy. Aquí viene el
punto más importante de la defensa, y quiero adelantarme a la pregunta obvia: ¿por qué la
accuracy es 'solo' 44 %?"

### [Slide · La trampa de la accuracy] (~1 min 20 s)
"Porque la accuracy **engaña** con datos desbalanceados. Fíjense: el clasificador ingenuo,
que siempre predice 'local', ya tiene 0,434 de accuracy, casi igual que nuestro mejor modelo.
Pero su F1 macro es 0,20 y su Kappa es cero: es inútil. Nuestro modelo, con la misma
accuracy aparente, **duplica el F1 macro** —de 0,20 a 0,42—, lleva el Kappa de 0 a 0,15 y el
AUC de 0,50 a 0,61. O sea: donde el modelo demuestra que aprende no es en la accuracy, es en
las métricas robustas al desbalance. Este es exactamente el error que la matriz de la rúbrica
advierte, y nosotros lo evitamos por diseño."

### [Slide 16 · Visualizaciones — confusión y ROC] (~1 min)
"La matriz de confusión confirma el diagnóstico del EDA: el empate se lo reparten local y
visitante; casi nunca hay una región donde el empate sea la predicción mayoritaria. Su
sensibilidad tiene un techo estructural. El caso más elocuente es el SVM: en su versión
lineal el recall del empate es prácticamente cero, mientras que con kernel radial sube a 0,31,
el mejor de todos. La misma familia, con y sin kernel, prueba que la frontera del empate es no
lineal."

### [Slide 17 · Análisis estadístico — Wilcoxon] (~1 min)
"¿Y cómo sabemos que la ventaja no es azar? Con la prueba de Wilcoxon de rangos con signo,
comparando el F1 macro del modelo propuesto contra cada línea base, por bloque temporal
mensual, sobre 17 bloques del conjunto de prueba. El stacking supera al clasificador ingenuo
con un p-valor de 7,6 por 10 a la menos 6, y a Naive Bayes con 6,7 por 10 a la menos 4. Ambos
por debajo de 0,05: la mejora es estadísticamente significativa y consistente en el tiempo."

### [Slide 18 · Interpretabilidad — SHAP] (~50 s)
"Para interpretar el modelo usamos valores SHAP sobre el boosting. El resultado es claro: la
diferencia de Elo domina, muy por encima del historial directo, la forma y los goles
recientes. Y lo confirmamos con un método independiente, la importancia por permutación del
Random Forest, que apunta a la misma variable. Dos técnicas distintas, la misma conclusión."

### [Slide 18–19 · Discusión y amenazas a la validez] (~1 min)
"La lectura unificada: la frontera es no lineal, por eso el boosting y el ensamble lideran;
pero sin cuotas de mercado la señal tiene un techo bajo, y todos los modelos competentes
convergen. El empate es un muro estructural. Sobre las amenazas a la validez: la interna, la
fuga temporal, la mitigamos con la partición por fecha —y la ausencia de accuracies
sospechosamente altas lo confirma—; la externa, que el modelo se entrenó sin cuotas y su
generalización a dinámicas muy distintas, como la pandemia, no está garantizada."

### [Slide 20–21 · Conclusiones y trabajo futuro] (~50 s)
"Tres conclusiones. Primera: el empate es un límite estructural, no una falla de los
algoritmos. Segunda: la frontera es no lineal y el reponderado por clase superó al
sobremuestreo SMOTE, que casi no ayudó. Tercera: el modelo propuesto supera a las líneas base
con significancia estadística. Como trabajo futuro, incorporar las cuotas para medir cuánto
sube el techo, calibrar la incertidumbre del empate y explorar modelos secuenciales de la
forma."

### [Slide 22 · Cierre] (~20 s)
"En resumen: construimos un pipeline honesto y reproducible que, sin trampas, exprime la
señal disponible y lo demuestra estadísticamente. El código y el artículo están en nuestro
repositorio. Gracias; quedamos atentos a sus preguntas."

---

## Banco de preguntas cruzadas probables (preparación)

> El jurado puede dirigir una pregunta a cualquiera, incluso sobre la parte de otro.

**1. ¿Por qué su accuracy es tan baja (44 %)?**
No es baja para el problema: el piso honesto es el 43 % del clasificador mayoritario, no el
33 % del azar uniforme. Sobre ese piso, lo que mejora no es la accuracy sino el F1 macro (de
0,20 a 0,42), el Kappa y el AUC. La accuracy es la métrica equivocada bajo desbalance.

**2. ¿Por qué no usaron las cuotas de apuestas, si mejoran la predicción?**
Precisamente por eso: la cuota es un "leak del sabio", ya contiene el conocimiento del
mercado. Queríamos medir el poder predictivo del modelo, no copiar al bookmaker. Es una
versión honesta; medir el salto con cuotas queda como trabajo futuro.

**3. ¿Por qué TimeSeriesSplit y no un K-fold estándar?**
Porque los datos tienen orden temporal. Un K-fold aleatorio entrenaría con partidos del
futuro para predecir el pasado, lo que infla artificialmente el desempeño. La literatura de
escala (Bunker, Berrar) lo advierte explícitamente.

**4. ¿Por qué el stacking es "el mejor" si su F1 casi empata con LightGBM?**
Por dos razones: gana en validación, que es donde se hace la selección, y su ventaja sobre
las líneas base es estadísticamente significativa. Además, combina familias diversas, lo que
lo hace más robusto que un único modelo.

**5. ¿SMOTE no debería haber ayudado con el empate?**
En teoría sí, pero falló: el empate no tiene región propia en el espacio, así que interpolar
entre empates —y encima sobre variables one-hot— genera puntos que caen en zonas ya
dominadas por otras clases. El reponderado por clase fue superior.

**6. (A Arbués, sobre la parte de Sergio) ¿Qué es la regularización L2 y por qué la usan?**
Es el término lambda por la norma de los pesos al cuadrado; penaliza pesos grandes. La usamos
porque, con las variables colineales, la matriz del sistema es singular, y la L2 —que es la
regularización de Tikhonov— la vuelve invertible y estabiliza la solución.

**7. (A Luis, sobre la parte de Arbués) ¿Cómo garantizan que no hay fuga de datos?**
Con tres medidas: descartamos toda variable post-partido, reconstruimos el Elo mirando solo
el pasado, y ajustamos el escalador únicamente en entrenamiento. La señal de que funcionó es
que no aparecen accuracies sospechosamente altas, mayores al 65 %.

**8. ¿Cuál es el aporte real frente a la literatura?**
Ningún trabajo compara seis clasificadores modernos —incluidos LightGBM y CatBoost— con
métricas por clase y prueba de significancia sobre más de 200 000 partidos multi-liga. Los que
tienen esa escala usan métricas probabilísticas y pocos modelos; los que usan nuestras
métricas trabajan con unos pocos miles de partidos.

---

### Notas de ejecución (para el ensayo)
- **Cronometrar**: 7 min exactos por persona. Si Arbués se pasa, Luis se queda sin tiempo.
- **Transiciones**: usar las frases puente marcadas ("le paso la palabra a…").
- **No leer las slides**: las diapositivas llevan diagramas y ecuaciones; el texto va aquí.
- **Números que hay que saber de memoria**: 230 554 partidos; H/D/A = 44,6/28,9/26,5 %;
  F1 macro propuesto 0,419; accuracy ~0,44; piso Dummy F1 0,20; Wilcoxon p < 0,001.
