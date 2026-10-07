"""
This is a boilerplate pipeline 'data_hyperparameters'
generated using Kedro 1.7.0
"""

import lightgbm as lgb
import pandas as pd
from typing import Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold

def optimize_hyperparameters(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    parameters: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Node do Kedro para realizar a otimização de hiperparâmetros usando GridSearchCV
    com LGBMClassifier.
    """
    # Extrai o param_grid vindo do parameters.yml
    param_grid = parameters.get("param_grid", {})

    # Instancia o classificador do LightGBM
    clf = lgb.LGBMClassifier(
        objective="binary",
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbosity=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=clf, 
        param_grid=param_grid, 
        cv=cv,
        n_jobs=-1,
        scoring="roc_auc",
        refit=True,
    )

    print("Iniciando busca de hiperparâmetros com CV estratificada no treino original...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Melhores parâmetros encontrados: {best_params}")
    print(f"ROC-AUC médio na validação cruzada: {grid_search.best_score_:.4f}")
    print(f"Gini médio na validação cruzada: {2 * grid_search.best_score_ - 1:.4f}")

    return best_model, best_params

def fit_model_hyperparameters(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    best_params: dict[str, Any],
) -> lgb.LGBMClassifier:
    """
    Node do Kedro para treinar o modelo final com os melhores hiperparâmetros encontrados.
    """
    # Instancia o classificador do LightGBM com os melhores parâmetros
    final_model = lgb.LGBMClassifier(
        **best_params,
        objective="binary",
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbosity=-1,
    )

    print("Treinando o modelo final com early stopping na validação...")
    final_model.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    return final_model