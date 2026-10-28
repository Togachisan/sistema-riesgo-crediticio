"""
niveles_riesgo.py
-----------------
Fase 10: categorías OPERATIVAS de riesgo (bajo / medio / alto) a partir de la
puntuación (probabilidad estimada) del modelo binario.

IMPORTANTE (honestidad metodológica):
  * TARGET es binaria. El dataset NO trae etiquetas de riesgo bajo/medio/alto.
  * Los niveles son una propuesta de negocio, NO etiquetas originales ni un estándar oficial.
  * Los cortes se definen por percentiles de las predicciones de validación (entrenamiento).
    Los percentiles de CORTES_RIESGO son una decisión ajustable por CrediAvanza.
  * La verificación de que los niveles tienen sentido se hace en el conjunto de prueba:
    la tasa REAL de incumplimiento debe aumentar de Bajo a Medio y de Medio a Alto.
  * La puntuación no está calibrada (se usó class_weight o vecinos): no es una probabilidad real.

Ejecutar (después de entrenar):   python -m src.niveles_riesgo
"""

import json

import numpy as np
import pandas as pd

# Percentiles de la puntuación de validación que separan Bajo | Medio | Alto.
# Propuesta inicial: 50 % de solicitudes en riesgo bajo, 35 % medio, 15 % alto.
CORTES_RIESGO = (0.50, 0.85)
NIVELES = ["Bajo", "Medio", "Alto"]


def calcular_umbrales(puntuaciones_validacion) -> dict:
    """Convierte los percentiles en valores de puntuación concretos."""
    c_medio, c_alto = np.quantile(puntuaciones_validacion, CORTES_RIESGO)
    return {"medio": float(c_medio), "alto": float(c_alto)}


def asignar_nivel(proba, umbrales: dict):
    """Devuelve 'Bajo' / 'Medio' / 'Alto' (escalar o array)."""
    p = np.asarray(proba, dtype=float)
    return np.where(p >= umbrales["alto"], "Alto", np.where(p >= umbrales["medio"], "Medio", "Bajo"))


def main():
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    from src.evaluacion import GRAFICOS
    from src.limpieza import RAIZ, cargar_datos, dividir_datos

    modelo = joblib.load(RAIZ / "models" / "modelo_final.joblib")
    meta = json.loads((RAIZ / "models" / "metadata.json").read_text(encoding="utf-8"))
    umbrales = meta["umbrales_riesgo"]

    X, y = cargar_datos()
    _, X_te, _, y_te = dividir_datos(X, y)          # misma división (misma semilla)
    proba = modelo.predict_proba(X_te)[:, 1]
    niveles = pd.Categorical(asignar_nivel(proba, umbrales), categories=NIVELES, ordered=True)

    tabla = (pd.DataFrame({"nivel": niveles, "incumple": y_te.values})
             .groupby("nivel", observed=True)["incumple"].agg(solicitudes="count", incumplimientos="sum"))
    tabla["% de solicitudes"] = (tabla["solicitudes"] / tabla["solicitudes"].sum() * 100).round(1)
    tabla["tasa real de incumplimiento %"] = (tabla["incumplimientos"] / tabla["solicitudes"] * 100).round(2)
    tabla["veces la tasa global"] = (tabla["tasa real de incumplimiento %"] / (y_te.mean() * 100)).round(2)
    print(f"Cortes de puntuación (de validación): {umbrales}")
    print(f"Tasa global de incumplimiento en test: {y_te.mean()*100:.2f} %\n")
    print(tabla.to_string())

    tasas = tabla["tasa real de incumplimiento %"].tolist()
    monotono = all(a < b for a, b in zip(tasas, tasas[1:]))
    print("\n¿La tasa real aumenta de Bajo a Alto?:", "SÍ" if monotono else "NO -> los niveles no son defendibles así")
    tabla.to_csv(RAIZ / "reports" / "niveles_riesgo_test.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tabla.index.astype(str), tabla["tasa real de incumplimiento %"], color=["#4caf50", "#ffb300", "#e53935"])
    ax.axhline(y_te.mean() * 100, color="k", ls="--", label="Tasa global")
    ax.set_ylabel("% con TARGET = 1")
    ax.set_title("Tasa real de incumplimiento por nivel (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / "niveles_riesgo.png", dpi=150)
    plt.close(fig)

    # Calibración: si la curva no sigue la diagonal, la puntuación NO es una probabilidad real
    obs, prev = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prev, obs, "o-", label="Modelo")
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    ax.set_xlabel("Puntuación media del modelo")
    ax.set_ylabel("Fracción real de incumplimientos")
    ax.set_title("Curva de calibración (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / "calibracion.png", dpi=150)
    plt.close(fig)
    print("\nGráficos guardados en reports/graficos/ (niveles_riesgo.png, calibracion.png)")


if __name__ == "__main__":
    main()
