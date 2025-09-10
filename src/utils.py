from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def ensure_dir(path: str) -> None:
    """Cria diretório se não existir."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    average: str = "binary",
                    labels: List[str] | None = None) -> Dict:
    """
    Retorna métricas padrão (accuracy, precision, recall, f1) + relatório.
    - Para problema multiclasse use average="macro" (ou "weighted").
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    if average == "binary":
        prec = metrics.precision_score(y_true, y_pred, zero_division=0)
        rec = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    else:
        prec = metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)

    report = metrics.classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "classification_report": report
    }


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: List[str],
                          title: str,
                          out_path: str) -> None:
    """Gera e salva a matriz de confusão (normalizada por linha)."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig = plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_norm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)

    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
