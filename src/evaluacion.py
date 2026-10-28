"""
evaluacion.py
-------------
Fases 8 y 9: métricas, umbral, matrices de confusión, ROC e importancia de variables.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guardar gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

GRAFICOS = Path(__file__).resolve().parent.parent / "reports" / "graficos"
GRAFICOS.mkdir(parents=True, exist_ok=True)


def mejor_umbral_f1(y_true, proba) -> float:
    """Umbral que maximiza F1. Debe calcularse con predicciones de VALIDACIÓN, nunca con test."""
    p, r, umbrales = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(umbrales[np.argmax(f1)])


def metricas(y_true, proba, umbral=0.5) -> dict:
    pred = (np.asarray(proba) >= umbral).astype(int)
    return {
        "umbral": round(float(umbral), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def graficar_matrices(y_true, probas: dict, umbrales: dict, nombre="matrices_confusion.png"):
    """probas/umbrales: {'KNN': ..., 'Árbol': ...}. Una matriz por modelo."""
    fig, ejes = plt.subplots(1, len(probas), figsize=(5.5 * len(probas), 4.5))
    ejes = np.atleast_1d(ejes)
    for ax, (nom, pr) in zip(ejes, probas.items()):
        cm = confusion_matrix(y_true, (pr >= umbrales[nom]).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1], ["Real 0", "Real 1"])
        ax.set_title(f"{nom} (umbral = {umbrales[nom]:.3f})")
    fig.tight_layout()
    fig.savefig(GRAFICOS / nombre, dpi=150)
    plt.close(fig)


def graficar_roc(y_true, probas: dict, nombre="curvas_roc.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for nom, pr in probas.items():
        fpr, tpr, _ = roc_curve(y_true, pr)
        ax.plot(fpr, tpr, label=f"{nom} (AUC = {roc_auc_score(y_true, pr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_title("Curvas ROC en el conjunto de prueba")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / nombre, dpi=150)
    plt.close(fig)


def graficar_comparacion(tabla: pd.DataFrame, nombre="comparacion_metricas.png"):
    """tabla: una fila por modelo/umbral, columnas = métricas."""
    cols = ["precision", "recall", "f1", "roc_auc"]
    ax = tabla[cols].plot.bar(figsize=(9, 5), rot=20)
    ax.set_title("Comparación de métricas en el conjunto de prueba")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(GRAFICOS / nombre, dpi=150)
    plt.close()


def graficar_importancias(pipeline_arbol, top=15, nombre="importancia_arbol.png") -> pd.Series:
    nombres = pipeline_arbol.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipeline_arbol.named_steps["modelo"].feature_importances_, index=nombres)
    imp = imp.sort_values(ascending=False)
    ax = imp.head(top)[::-1].plot.barh(figsize=(8, 6))
    ax.set_title("Importancia de variables (árbol de decisión)")
    ax.set_xlabel("Importancia (reducción de impureza). No implica causalidad.")
    plt.tight_layout()
    plt.savefig(GRAFICOS / nombre, dpi=150)
    plt.close()
    return imp
