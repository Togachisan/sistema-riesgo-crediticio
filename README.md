# Sistema de evaluación de riesgo crediticio para una fintech (CrediAvanza)

Proyecto final · Coders IA (80 h) · Fundación Somos F5.
Modelo de Machine Learning que estima el riesgo de incumplimiento de solicitantes de microcréditos y lo traduce a niveles operativos **bajo / medio / alto**, con una demo en Streamlit.

> Proyecto académico. No es una herramienta para conceder o denegar créditos reales.

## 1. Dataset
Home Credit Default Risk (Kaggle: `youngdaniel/loan-dataset`). Se usa solo `application_train.csv` (307.511 filas × 122 columnas).
`TARGET` es **binaria** (1 = incumplimiento registrado según el dataset). **El dataset no contiene niveles bajo/medio/alto**: se construyen como categorías operativas (ver §5).
Los CSV no se incluyen en el repositorio (tamaño y condiciones de uso); descárgalos de Kaggle y colócalos en `data/`.

## 2. Estructura
```
data/            CSV originales (no versionados)
notebooks/       analisis_exploratorio.ipynb  (EDA con celdas explicativas)
src/limpieza.py        características, preprocesado (Pipeline), división train/test
src/entrenamiento.py   KNN y Árbol, validación cruzada, evaluación, guardado
src/evaluacion.py      métricas, umbral, gráficos
src/niveles_riesgo.py  niveles bajo/medio/alto y calibración
models/          modelo_final.joblib + metadata.json
reports/         tablas CSV y graficos/
app.py           demo Streamlit
docs/            guía de presentación y preguntas del profesorado
```

## 3. Instalación y ejecución (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.limpieza          # informe de calidad de datos
jupyter notebook                # abrir notebooks/analisis_exploratorio.ipynb y ejecutar todo
python -m src.entrenamiento     # entrena, evalúa y guarda el modelo
python -m src.niveles_riesgo    # niveles bajo/medio/alto
streamlit run app.py            # demo
```

## 4. Metodología
- **Limpieza:** `DAYS_EMPLOYED` contiene un valor centinela (365243) que se convierte en nulo + indicador; edades y antigüedad en años; logaritmo en importes; ratios crédito/ingreso, cuota/ingreso y plazo aproximado. Nulos imputados (mediana / "Sin dato") dentro del Pipeline, sin eliminar filas.
- **Variables:** 17 numéricas + 8 categóricas disponibles al solicitar. Se excluyen `SK_ID_CURR`, `CODE_GENDER` (decisión de equidad), bloque de vivienda `_AVG/_MODE/_MEDI` (redundante, muchos nulos) y `FLAG_DOCUMENT_*`. `EXT_SOURCE_*` son puntuaciones externas cuyo cálculo no está documentado: se usan, con esa salvedad.
- **División:** 80/20 estratificada (`random_state=42`). La prueba solo se usa al final.
- **Modelos:** KNN (con escalado) y Árbol de decisión (`class_weight="balanced"`), con `GridSearchCV` (ROC-AUC, 5 pliegues). Para acelerar KNN se ajusta con una submuestra estratificada de `MAX_FILAS_ENTRENAMIENTO` filas (configurable en `src/limpieza.py`).
- **Umbral:** se elige maximizando F1 sobre predicciones *out-of-fold* (no sobre la prueba).
- **Métricas:** accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, validación cruzada. Se compara contra un modelo de referencia que predice siempre la clase mayoritaria.

## 5. Tres niveles de riesgo
Cortes por percentiles (50 % / 85 %) de la puntuación de validación → Bajo / Medio / Alto. Son una **propuesta ajustable**, no un estándar. Se valida comprobando que la tasa real de incumplimiento en la prueba crece de Bajo a Alto (`reports/niveles_riesgo_test.csv`). La puntuación no está calibrada (`reports/graficos/calibracion.png`).

## 6. Resultados
Ajuste y validación con una submuestra estratificada de 60.000 filas; prueba: 61.503 solicitudes no vistas (8,07 % con TARGET=1).

**Validación cruzada (5 pliegues, umbral 0,5)**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Referencia (siempre clase 0) | 0,9193 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| KNN (k=101) | 0,9193 | 0,0000 | 0,0000 | 0,0000 | 0,7169 |
| Árbol (profundidad 8, hoja mín. 200) | 0,6278 | 0,1374 | 0,6821 | 0,2287 | 0,6980 |

**Conjunto de prueba**

| Modelo / umbral | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN, 0,5 | 0,9193 | 0,0000 | 0,0000 | 0,0000 | 0,7201 |
| KNN, 0,1485 (F1 de validación) | 0,8360 | 0,2115 | 0,3782 | 0,2713 | 0,7201 |
| Árbol, 0,5 | 0,6403 | 0,1414 | 0,6814 | 0,2342 | 0,7101 |
| Árbol, 0,6165 (F1 de validación) | 0,8083 | 0,1934 | 0,4338 | 0,2676 | 0,7101 |

**Niveles operativos de riesgo (prueba)**

| Nivel | Solicitudes | % | Tasa real de incumplimiento |
|---|---|---|---|
| Bajo | 30.082 | 48,9 | 3,49 % |
| Medio | 20.878 | 33,9 | 8,66 % |
| Alto | 10.543 | 17,1 | 19,98 % |

Lectura: el accuracy de 0,92 de la referencia y de KNN a 0,5 no detecta ningún incumplimiento. ROC-AUC y F1 son similares entre modelos; el árbol detecta más incumplimientos (mayor recall) a costa de precisión. Las tres variables `EXT_SOURCE_*` concentran cerca del 80 % de la importancia del árbol (asociación, no causalidad).

## 7. Limitaciones
Datos de otra entidad y contexto; TARGET binaria; puntuación no calibrada; submuestra para KNN; sin auditoría de equidad completa; asociación ≠ causalidad; no se usaron las tablas auxiliares (`bureau`, etc.), que podrían mejorar el rendimiento.

## 8. Mejoras futuras
Incorporar `bureau`, `previous_application` e `installments_payments` (agregando por `SK_ID_CURR` y vigilando fugas); calibración de probabilidades; modelos de boosting; análisis de equidad por grupos; ajuste de cortes con costes de negocio.
