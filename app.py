"""
app.py - Demostración académica en Streamlit (Fase 12)
Ejecutar desde la raíz del proyecto:   streamlit run app.py
Requiere haber ejecutado antes:        python -m src.entrenamiento
"""

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.limpieza import DIAS_ANOMALO, RAIZ
from src.niveles_riesgo import asignar_nivel

st.set_page_config(page_title="CrediAvanza - Riesgo crediticio", page_icon="💳", layout="wide")


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(RAIZ / "models" / "modelo_final.joblib")
    meta = json.loads((RAIZ / "models" / "metadata.json").read_text(encoding="utf-8"))
    return modelo, meta


try:
    modelo, meta = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró el modelo. Ejecuta primero:  python -m src.entrenamiento")
    st.stop()

cat = meta["categorias"]

# ------------------------------------------------------------------ Cabecera
st.title("💳 Sistema de evaluación de riesgo crediticio")
st.caption("Proyecto final Coders IA · Fundación Somos F5 · Caso: CrediAvanza (fintech de microcréditos)")

with st.expander("¿Qué problema resuelve y cómo funciona?", expanded=True):
    st.markdown(
        f"""
**Problema.** CrediAvanza necesita apoyo para priorizar el análisis de solicitudes según su riesgo.

**Modelo.** {meta['modelo']} entrenado con datos históricos de *Home Credit* (Kaggle). Predice la variable
`TARGET` (1 = el dataset registra dificultades de pago). El resultado es una **puntuación**; los niveles
**Bajo / Medio / Alto** son categorías *operativas* creadas con percentiles de esa puntuación
(no son etiquetas originales del dataset).

**Importante.** Es una demostración académica. No debe usarse para conceder o denegar créditos reales.
"""
    )

# ------------------------------------------------------------------ Formulario
with st.form("formulario"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Solicitante")
        edad = st.number_input("Edad (años)", 18, 100, 35)
        anios_empleo = st.number_input("Años en el empleo actual", 0.0, 60.0, 5.0, step=0.5)
        sin_dato_empleo = st.checkbox("Antigüedad laboral no informada (valor anómalo del dataset)")
        hijos = st.number_input("Número de hijos", 0, 20, 0)
        miembros = st.number_input("Miembros de la familia", 1, 25, 2)
        estado = st.selectbox("Estado civil", cat["NAME_FAMILY_STATUS"])
        educ = st.selectbox("Nivel educativo", cat["NAME_EDUCATION_TYPE"])
    with c2:
        st.subheader("Situación financiera")
        ingreso = st.number_input("Ingreso total", 1.0, 1e9, 150000.0, step=5000.0)
        credito = st.number_input("Importe del crédito solicitado", 1.0, 1e9, 500000.0, step=10000.0)
        anualidad = st.number_input("Cuota (anualidad) del crédito", 1.0, 1e8, 25000.0, step=1000.0)
        bienes = st.number_input("Precio de los bienes a financiar", 0.0, 1e9, 450000.0, step=10000.0)
        tipo_ingreso = st.selectbox("Tipo de ingreso", cat["NAME_INCOME_TYPE"])
        ocup = st.selectbox("Ocupación", ["(no informada)"] + cat["OCCUPATION_TYPE"])
        contrato = st.selectbox("Tipo de contrato", cat["NAME_CONTRACT_TYPE"])
    with c3:
        st.subheader("Otros datos")
        coche = st.selectbox("¿Tiene coche?", cat["FLAG_OWN_CAR"])
        inmueble = st.selectbox("¿Tiene inmueble?", cat["FLAG_OWN_REALTY"])
        vivienda = st.selectbox("Tipo de vivienda", cat["NAME_HOUSING_TYPE"])
        rating = st.selectbox("Valoración de la región (1 = mejor, 3 = peor)", [1, 2, 3], index=1)
        pob = st.number_input("Población relativa de la región (0-1)", 0.0, 1.0, 0.02, step=0.005, format="%.4f")
        st.markdown("**Puntuaciones externas (0-1)**")
        ext = {}
        for i in (1, 2, 3):
            ok = st.checkbox(f"EXT_SOURCE_{i} disponible", value=True, key=f"ok{i}")
            ext[i] = st.number_input(f"EXT_SOURCE_{i}", 0.0, 1.0, 0.5, step=0.01, key=f"v{i}",
                                     disabled=not ok) if ok else np.nan
    enviar = st.form_submit_button("Calcular riesgo", type="primary")

# ------------------------------------------------------------------ Predicción
if enviar:
    fila = pd.DataFrame([{
        "AMT_INCOME_TOTAL": ingreso, "AMT_CREDIT": credito, "AMT_ANNUITY": anualidad,
        "AMT_GOODS_PRICE": bienes, "CNT_CHILDREN": hijos, "CNT_FAM_MEMBERS": float(miembros),
        "REGION_POPULATION_RELATIVE": pob, "REGION_RATING_CLIENT": rating,
        "EXT_SOURCE_1": ext[1], "EXT_SOURCE_2": ext[2], "EXT_SOURCE_3": ext[3],
        "DAYS_BIRTH": -edad * 365.25,
        "DAYS_EMPLOYED": DIAS_ANOMALO if sin_dato_empleo else -anios_empleo * 365.25,
        "NAME_CONTRACT_TYPE": contrato, "FLAG_OWN_CAR": coche, "FLAG_OWN_REALTY": inmueble,
        "NAME_INCOME_TYPE": tipo_ingreso, "NAME_EDUCATION_TYPE": educ, "NAME_FAMILY_STATUS": estado,
        "NAME_HOUSING_TYPE": vivienda, "OCCUPATION_TYPE": np.nan if ocup.startswith("(") else ocup,
    }])[meta["columnas_entrada"]]          # mismo orden de columnas que en el entrenamiento

    puntuacion = float(modelo.predict_proba(fila)[0, 1])
    nivel = str(asignar_nivel(puntuacion, meta["umbrales_riesgo"]))
    alerta = puntuacion >= meta["umbral_decision_f1"]

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Nivel de riesgo (operativo)", nivel)
    r2.metric("Puntuación del modelo", f"{puntuacion:.3f}")
    r3.metric("Predicción binaria (TARGET)", "1 · posible incumplimiento" if alerta else "0 · sin alerta")
    {"Bajo": st.success, "Medio": st.warning, "Alto": st.error}[nivel](
        f"Riesgo **{nivel.lower()}** según los cortes definidos para este proyecto."
    )
    st.info(
        "La puntuación **no es una probabilidad calibrada** de impago: sirve para ordenar solicitudes por riesgo. "
        "El nivel resulta de comparar la puntuación con percentiles de validación "
        f"(medio ≥ {meta['umbrales_riesgo']['medio']:.3f}, alto ≥ {meta['umbrales_riesgo']['alto']:.3f})."
    )

# ------------------------------------------------------------------ Limitaciones
with st.expander("Limitaciones del sistema"):
    st.markdown("\n".join(f"- {t}" for t in meta["limitaciones"]))
    st.caption(f"Modelo v{meta['version']} · {meta['fecha']} · scikit-learn {meta['scikit_learn']}")
