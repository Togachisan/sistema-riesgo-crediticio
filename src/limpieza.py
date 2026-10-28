"""
limpieza.py
-----------
Fases 4, 5 y 6: carga, ingeniería de características, preprocesamiento y división.

Idea clave (evitar fuga de información):
  * crear_caracteristicas() NO aprende nada de los datos (solo operaciones fila a fila),
    por eso es seguro aplicarla antes de dividir.
  * Todo lo que SÍ aprende (medianas, escalado, categorías) vive dentro del Pipeline
    y se ajusta únicamente con los datos de entrenamiento.

Uso como informe de calidad (desde la raíz del proyecto):   python -m src.limpieza
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_TRAIN = RAIZ / "data" / "application_train.csv"
SEMILLA = 42
TAM_TEST = 0.20
MAX_FILAS_ENTRENAMIENTO = 60_000   # None = usar todo el entrenamiento (KNN será mucho más lento)
DIAS_ANOMALO = 365243              # valor centinela conocido de DAYS_EMPLOYED (se verifica en el EDA)

# --------------------------------------------------------------------------
# Selección de variables (Fase 5)
# Criterio: información disponible en el momento de la solicitud, interpretable
# y que pueda pedirse en un formulario. Se excluyen:
#   - SK_ID_CURR (identificador) y TARGET (objetivo)
#   - CODE_GENDER (atributo sensible; decisión de equidad, ver README)
#   - Bloque de vivienda *_AVG/_MODE/_MEDI (muy redundante y con 50-70 % de nulos)
#   - FLAG_DOCUMENT_* y FLAG_* de contacto (poca señal esperable; se revisa en el EDA)
# --------------------------------------------------------------------------
NUM_CRUDAS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
]
CAT = [
    "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
COLUMNAS_ENTRADA = NUM_CRUDAS + CAT   # columnas "crudas" que necesita el modelo

NUM_FINALES = [
    "LOG_AMT_INCOME_TOTAL", "LOG_AMT_CREDIT", "LOG_AMT_ANNUITY", "LOG_AMT_GOODS_PRICE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EDAD_ANIOS", "ANIOS_EMPLEO", "EMPLEO_ANOMALO",
    "RATIO_CREDITO_INGRESO", "RATIO_CUOTA_INGRESO", "PLAZO_APROX_CUOTAS",
]


def crear_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones fila a fila (sin aprender de los datos)."""
    X = df[COLUMNAS_ENTRADA].copy()
    for c in NUM_CRUDAS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # DAYS_*: días relativos a la solicitud (negativos). Los pasamos a años positivos.
    anomalo = X["DAYS_EMPLOYED"] == DIAS_ANOMALO
    X["EMPLEO_ANOMALO"] = anomalo.astype(int)                       # conserva la información del valor raro
    X["ANIOS_EMPLEO"] = -X["DAYS_EMPLOYED"].mask(anomalo) / 365.25  # el centinela pasa a NaN
    X["EDAD_ANIOS"] = -X["DAYS_BIRTH"] / 365.25

    # Ratios de endeudamiento (análogos al "porcentaje de endeudamiento" del enunciado)
    ingreso = X["AMT_INCOME_TOTAL"].where(X["AMT_INCOME_TOTAL"] > 0)
    anualidad = X["AMT_ANNUITY"].where(X["AMT_ANNUITY"] > 0)
    X["RATIO_CREDITO_INGRESO"] = X["AMT_CREDIT"] / ingreso
    X["RATIO_CUOTA_INGRESO"] = X["AMT_ANNUITY"] / ingreso
    X["PLAZO_APROX_CUOTAS"] = X["AMT_CREDIT"] / anualidad

    # Importes muy asimétricos con valores extremos -> logaritmo
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        X["LOG_" + c] = np.log1p(X[c].clip(lower=0))

    X = X.replace([np.inf, -np.inf], np.nan)
    for c in CAT:                       # tipo object homogéneo para el imputador/codificador
        X[c] = X[c].astype(object)
    return X[NUM_FINALES + CAT]


def construir_preprocesador(escalar: bool) -> ColumnTransformer:
    """Imputación + (escalado) + codificación. Se ajusta SOLO con entrenamiento."""
    pasos_num = [("imputador", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:                          # KNN lo necesita; el árbol no
        pasos_num.append(("escalador", StandardScaler()))
    pasos_cat = [
        ("imputador", SimpleImputer(strategy="constant", fill_value="Sin dato")),
        ("codificador", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(pasos_num), NUM_FINALES),
        ("cat", Pipeline(pasos_cat), CAT),
    ])


def construir_pipeline(modelo, escalar: bool) -> Pipeline:
    """Pipeline completo: características -> preprocesado -> modelo."""
    return Pipeline([
        ("caracteristicas", FunctionTransformer(crear_caracteristicas)),
        ("prep", construir_preprocesador(escalar)),
        ("modelo", modelo),
    ])


# --------------------------------------------------------------------------
# Carga y división (Fase 6)
# --------------------------------------------------------------------------
def cargar_datos():
    """Devuelve X (columnas crudas necesarias) e y (TARGET)."""
    df = pd.read_csv(RUTA_TRAIN, usecols=["TARGET"] + COLUMNAS_ENTRADA)
    return df[COLUMNAS_ENTRADA], df["TARGET"]


def dividir_datos(X, y):
    """80/20 estratificado. El 20 % de prueba no se toca hasta la evaluación final."""
    return train_test_split(X, y, test_size=TAM_TEST, stratify=y, random_state=SEMILLA)


def muestra_entrenamiento(X_tr, y_tr):
    """Submuestra estratificada del entrenamiento (solo para acelerar KNN/validación)."""
    if MAX_FILAS_ENTRENAMIENTO is None or len(X_tr) <= MAX_FILAS_ENTRENAMIENTO:
        return X_tr, y_tr
    X_m, _, y_m, _ = train_test_split(
        X_tr, y_tr, train_size=MAX_FILAS_ENTRENAMIENTO, stratify=y_tr, random_state=SEMILLA
    )
    return X_m, y_m


# --------------------------------------------------------------------------
# Informe de calidad de datos
# --------------------------------------------------------------------------
if __name__ == "__main__":
    X, y = cargar_datos()
    print(f"Filas: {len(X):,} | columnas crudas usadas: {X.shape[1]}")
    print(f"DAYS_EMPLOYED == {DIAS_ANOMALO}: {(X['DAYS_EMPLOYED'] == DIAS_ANOMALO).sum():,} filas")
    F = crear_caracteristicas(X)
    print(f"Características finales: {F.shape[1]} ({len(NUM_FINALES)} numéricas + {len(CAT)} categóricas)")
    print("\n% de nulos en las características finales:")
    print((F.isna().mean() * 100).round(2).sort_values(ascending=False).head(12))
    print("\nResumen de las numéricas:")
    print(F[NUM_FINALES].describe().T[["min", "50%", "max"]].round(3))
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y)
    print(f"\nTrain: {len(X_tr):,} (TARGET=1: {y_tr.mean()*100:.2f} %)")
    print(f"Test:  {len(X_te):,} (TARGET=1: {y_te.mean()*100:.2f} %)")
