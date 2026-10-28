# Guía de presentación (Fase 14)

Rellena cada diapositiva con tus resultados reales. Una idea principal por diapositiva.

| # | Diapositiva | Qué decir (lenguaje sencillo) | Material |
|---|---|---|---|
| 1 | Contexto | CrediAvanza concede microcréditos y necesita priorizar solicitudes según riesgo. | — |
| 2 | Problema | ¿Podemos estimar qué solicitudes tienen más riesgo de incumplir? | — |
| 3 | Objetivo | Modelo + niveles bajo/medio/alto + demo. | — |
| 4 | Dataset | Home Credit, 307.511 solicitudes, 122 columnas, TARGET binaria; no trae 3 niveles. | `eda_target.png` |
| 5 | EDA | Desequilibrio, nulos, valor anómalo de DAYS_EMPLOYED, variables más asociadas. | gráficos `eda_*` |
| 6 | Limpieza | Qué se hizo con nulos, outliers, anómalo, ratios. | `src/limpieza.py` |
| 7 | Variables | Cuáles se usan y cuáles se excluyen y por qué (fuga, equidad, redundancia). | README §4 |
| 8 | Modelos | KNN (vecinos similares) y Árbol (reglas si/no); escalado, profundidad, sobreajuste. | — |
| 9 | Evaluación | Por qué no accuracy; recall, F1, ROC-AUC, matriz de confusión. | `matrices_confusion.png` |
| 10 | Comparación | Tabla CV vs prueba; modelo elegido y por qué. | `metricas_prueba.csv` |
| 11 | Demo | Simular 3 solicitantes (bajo, medio, alto) en Streamlit. | `app.py` |
| 12 | Conclusiones | Solo con tus números. Recomendación de integración: apoyo a analistas, no decisión automática. | — |
| 13 | Limitaciones | Ver README §7. | — |
| 14 | Mejoras | Ver README §8. | — |

**Propuesta de integración (pide el enunciado):** el modelo ordena las solicitudes; riesgo bajo → revisión ágil, medio → revisión estándar con documentación adicional, alto → revisión manual reforzada. Siempre con supervisión humana.

# Posibles preguntas del profesorado

1. **¿Por qué no usaste accuracy?** Porque ~92 % de las solicitudes son TARGET=0; un modelo que siempre dice 0 tendría accuracy alto sin detectar ningún incumplimiento. (Tienes el modelo de referencia en la tabla de CV para demostrarlo.)
2. **¿TARGET tiene tres niveles?** No, es binaria. Los niveles son categorías operativas por percentiles de la puntuación, validadas viendo que la tasa real sube de bajo a alto. No son etiquetas originales.
3. **¿Qué es la fuga de información y cómo la evitaste?** Usar información del test o del futuro al entrenar. Dividí antes de ajustar, imputación/escalado/codificación dentro del Pipeline, umbral elegido con validación cruzada y no con la prueba, y excluí identificadores.
4. **¿Por qué el árbol usa class_weight y KNN no?** KNN no admite `class_weight`; compensé ajustando el umbral de decisión con validación.
5. **¿Por qué escalar para KNN y no para el árbol?** KNN usa distancias, las variables de mayor escala dominarían. El árbol compara umbrales por variable.
6. **¿Cómo evitaste el sobreajuste del árbol?** Limité `max_depth` y `min_samples_leaf` con validación cruzada. Compara CV y prueba en tu tabla.
7. **¿Por qué excluiste el género?** Es un atributo sensible; la decisión es de equidad. Aun así, otras variables pueden actuar como proxy; no hay auditoría completa (limitación).
8. **¿La puntuación es una probabilidad?** No está calibrada; mira `calibracion.png`.
9. **¿Importancia de variables = causa?** No, indica uso predictivo, no causalidad.
10. **¿Por qué no usaste las demás tablas?** Para construir primero una base sólida; agregarlas exige cuidar claves, granularidad y fechas. Es mejora futura.
11. **¿Por qué una submuestra?** KNN tiene coste alto con muchas filas. Es configurable y se declara como limitación.
12. **¿Qué harías con los falsos negativos?** Son incumplimientos no detectados; se priorizó recall/F1 y se ajusta el umbral según el coste de negocio.
