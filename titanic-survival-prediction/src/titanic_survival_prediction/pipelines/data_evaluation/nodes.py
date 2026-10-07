"""
This is a boilerplate pipeline 'data_evaluation'
generated using Kedro 1.7.0
"""

import lightgbm as lgb
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test, threshold=0.5):
    
    # Probabilidades
    preds = {
        "train": model.predict(X_train),
        "valid": model.predict(X_valid),
        "test": model.predict(X_test)
    }

    # Conversão de probabilidade -> classe
    predictions = {
        dataset: (prob >= threshold).astype(int)
        for dataset, prob in preds.items()
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
        y_pred = predictions[dataset]

        metrics[dataset] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }

    # Exibição
    print("\n--- Métricas de Avaliação ---")

    for dataset, dataset_metrics in metrics.items():
        print(f"\n{dataset.upper()}")

        for metric, value in dataset_metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")

    return metrics