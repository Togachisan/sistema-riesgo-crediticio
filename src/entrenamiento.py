"""
entrenamiento.py
----------------
Fases 7, 8 y 11: entrena KNN y Árbol de decisión, los compara y guarda el modelo final.

Flujo (sin fuga de información):
  1. Se separa un 20 % de PRUEBA (estratificado) que no se toca hasta el final.
  2. Con el 80 % restante (o una submuestra estratificada de él) se hace:
     - validación cruzada y búsqueda de hiperparámetros (GridSearchCV, métrica ROC-AUC),
     - predicciones "out-of-fold" para elegir el umbral de decisión (máx. F1) y los cortes de riesgo.
  3. Solo al final se evalúa en PRUEBA, una vez por modelo.

Ejecutar desde la raíz del proyecto:   python -m src.entrenamiento
"""

import json
import warnings
from datetime import date

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.evaluacion import (GRAFICOS, graficar_comparacion, graficar_importancias, graficar_matrices,
                            graficar_roc, mejor_umbral_f1, metricas)
from src.limpieza import (CAT, COLUMNAS_ENTRADA, MAX_FILAS_ENTRENAMIENTO, RAIZ, SEMILLA, cargar_datos,
                          construir_pipeline, dividir_datos, muestra_entrenamiento)
from src.niveles_riesgo import CORTES_RIESGO, calcular_umbrales

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

PUNTUACIONES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
VERSION_MODELO = "1.0"


def main():
    (RAIZ / "models").mkdir(exist_ok=True)
    X, y = cargar_datos()
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y)
    X_fit, y_fit = muestra_entrenamiento(X_tr, y_tr)
    print(f"Entrenamiento disponible: {len(X_tr):,} | usado para ajustar/validar: {len(X_fit):,} | prueba: {len(X_te):,}")
    print(f"% TARGET=1 -> fit: {y_fit.mean()*100:.2f} | prueba: {y_te.mean()*100:.2f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    # ---- Referencia: modelo que siempre predice la clase mayoritaria ----------
    base = construir_pipeline(DummyClassifier(strategy="prior"), escalar=False)
    r = cross_validate(base, X_fit, y_fit, cv=cv, scoring=PUNTUACIONES)
    filas_cv = {"Referencia (clase mayoritaria)": {m: r[f"test_{m}"].mean() for m in PUNTUACIONES}}

    # ---- Candidatos --------------------------------------------------------------
    candidatos = {
        "KNN": (KNeighborsClassifier(n_jobs=-1), True,
                {"modelo__n_neighbors": [15, 51, 101]}),
        "Árbol": (DecisionTreeClassifier(random_state=SEMILLA, class_weight="balanced"), False,
                  {"modelo__max_depth": [3, 5, 8, 12], "modelo__min_samples_leaf": [50, 200]}),
    }

    mejores, oof, proba_test, umbrales_f1, cv_auc = {}, {}, {}, {}, {}
    tabla_busqueda = []
    for nombre, (estimador, escalar, rejilla) in candidatos.items():
        print(f"=== {nombre}: búsqueda de hiperparámetros (validación cruzada 5 pliegues) ===")
        gs = GridSearchCV(construir_pipeline(estimador, escalar), rejilla, scoring="roc_auc", cv=cv, n_jobs=1)
        gs.fit(X_fit, y_fit)
        res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]]
        res.insert(0, "modelo", nombre)
        tabla_busqueda.append(res)
        print(res.drop(columns="modelo").to_string(index=False))
        print(f"Mejores parámetros: {gs.best_params_} | ROC-AUC CV: {gs.best_score_:.4f}\n")

        mejor = gs.best_estimator_
        mejores[nombre] = mejor
        cv_auc[nombre] = gs.best_score_

        # Validación cruzada del mejor modelo (umbral 0.5)
        r = cross_validate(mejor, X_fit, y_fit, cv=cv, scoring=PUNTUACIONES)
        filas_cv[nombre + " (umbral 0.5)"] = {m: r[f"test_{m}"].mean() for m in PUNTUACIONES}

        # Predicciones out-of-fold -> umbral de decisión elegido SIN usar la prueba
        oof[nombre] = cross_val_predict(mejor, X_fit, y_fit, cv=cv, method="predict_proba")[:, 1]
        umbrales_f1[nombre] = mejor_umbral_f1(y_fit, oof[nombre])
        proba_test[nombre] = mejor.predict_proba(X_te)[:, 1]

    # ---- Validación cruzada: tabla ---------------------------------------------
    tabla_cv = pd.DataFrame(filas_cv).T.round(4)
    print("=== VALIDACIÓN CRUZADA (media de 5 pliegues, sobre datos de entrenamiento) ===")
    print(tabla_cv.to_string())
    tabla_cv.to_csv(RAIZ / "reports" / "validacion_cruzada.csv")
    pd.concat(tabla_busqueda).to_csv(RAIZ / "reports" / "busqueda_hiperparametros.csv", index=False)

    # ---- Evaluación final en PRUEBA ----------------------------------------------
    filas_test = {}
    for nombre in candidatos:
        filas_test[f"{nombre} (umbral 0.5)"] = metricas(y_te, proba_test[nombre], 0.5)
        filas_test[f"{nombre} (umbral F1 de validación)"] = metricas(y_te, proba_test[nombre], umbrales_f1[nombre])
    tabla_test = pd.DataFrame(filas_test).T
    print("\n=== CONJUNTO DE PRUEBA (datos no vistos) ===")
    print(tabla_test.round(4).to_string())
    tabla_test.round(4).to_csv(RAIZ / "reports" / "metricas_prueba.csv")

    graficar_matrices(y_te, proba_test, umbrales_f1)
    graficar_roc(y_te, proba_test)
    graficar_comparacion(tabla_test)
    importancias = graficar_importancias(mejores["Árbol"])
    importancias.head(20).to_csv(RAIZ / "reports" / "importancia_arbol.csv", header=["importancia"])
    print("\nTop 10 variables del árbol (importancia != causalidad):")
    print(importancias.head(10).round(4).to_string())

    # ---- Selección y guardado del modelo final (Fase 11) -------------------------
    final = max(cv_auc, key=cv_auc.get)          # elegido por ROC-AUC de validación, NO por la prueba
    print(f"\nModelo final (mayor ROC-AUC de validación cruzada): {final}")
    umbrales_riesgo = calcular_umbrales(oof[final])

    joblib.dump(mejores[final], RAIZ / "models" / "modelo_final.joblib")
    metadata = {
        "version": VERSION_MODELO,
        "fecha": date.today().isoformat(),
        "modelo": final,
        "hiperparametros": {k: v for k, v in mejores[final].get_params().items()
                            if k.startswith("modelo__") and isinstance(v, (int, float, str, type(None)))},
        "scikit_learn": sklearn.__version__,
        "filas_entrenamiento_usadas": int(len(X_fit)),
        "umbral_decision_f1": umbrales_f1[final],
        "umbrales_riesgo": umbrales_riesgo,
        "cortes_percentil_riesgo": list(CORTES_RIESGO),
        "columnas_entrada": COLUMNAS_ENTRADA,
        "categorias": {c: sorted(X[c].dropna().unique().tolist()) for c in CAT},
        "metricas_prueba_umbral_f1": {k: float(v) for k, v in
                                      tabla_test.loc[f"{final} (umbral F1 de validación)"].items()},
        "limitaciones": [
            "Proyecto académico: no es una herramienta para conceder o denegar créditos reales.",
            "La puntuación no está calibrada: no es una probabilidad real de impago.",
            "Datos históricos de Home Credit: no representan necesariamente a CrediAvanza.",
            "TARGET es binaria; los niveles bajo/medio/alto son categorías operativas propuestas.",
            "Posibles sesgos: no se ha realizado una auditoría de equidad completa.",
        ],
    }
    (RAIZ / "models" / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Comprobación: el modelo se recarga y predice igual sin reentrenar
    recargado = joblib.load(RAIZ / "models" / "modelo_final.joblib")
    diff = abs(recargado.predict_proba(X_te.head(200))[:, 1] - proba_test[final][:200]).max()
    print(f"Comprobación de recarga: diferencia máxima de predicción = {diff:.2e} (debe ser 0)")
    print("Guardado: models/modelo_final.joblib y models/metadata.json")
    print(f"Gráficos en: {GRAFICOS}")


if __name__ == "__main__":
    main()
