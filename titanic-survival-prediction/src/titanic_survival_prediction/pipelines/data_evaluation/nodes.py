"""
This is a boilerplate pipeline 'data_evaluation'
generated using Kedro 1.7.0
"""

import pandas as pd
from typing import Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def _positive_class_scores(model: Any, features: pd.DataFrame) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def evaluate_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    threshold=0.5
):

    # Probabilidades da classe positiva
    scores = {
        "train": _positive_class_scores(model, X_train),
        "valid": _positive_class_scores(model, X_valid),
        "test": _positive_class_scores(model, X_test),
    }

    # Conversão de probabilidade -> classe
    predictions = {
        dataset: (score >= threshold).astype(int)
        for dataset, score in scores.items()
    }

    # Targets
    targets = {
        "train": y_train,
        "valid": y_valid,
        "test": y_test
    }

    # Métricas
    metrics = {}

    for dataset in ["train", "valid", "test"]:

        y_true = targets[dataset]
        y_prob = scores[dataset]
        y_pred = predictions[dataset]

        # ROC-AUC
        auc = roc_auc_score(y_true, y_prob)

        # Gini
        gini = 2 * auc - 1

        metrics[dataset] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            "roc_auc": auc,
            "gini": gini
        }

    # Exibição
    print("\n--- Métricas de Avaliação ---")

    for dataset, dataset_metrics in metrics.items():

        print(f"\n{dataset.upper()}")

        for metric, value in dataset_metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")

    return metrics
