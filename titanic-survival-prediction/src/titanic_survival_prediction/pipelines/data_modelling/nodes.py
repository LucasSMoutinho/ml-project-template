"""
This is a boilerplate pipeline 'data_modelling'
generated using Kedro 1.7.0
"""

import lightgbm as lgb
import pandas as pd
from typing import Dict, Any

def train_lightgbm_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_valid: pd.DataFrame, 
    y_valid: pd.Series, 
    parameters: Dict[str, Any]
) -> lgb.Booster:
    """
    Node do Kedro para treinar o modelo LightGBM usando os parâmetros da pipeline.
    """
    # Cria os datasets do LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # Treinamento utilizando o dicionário de parâmetros vindo do parameters_data_modelling.yml
    model = lgb.train(
            params=parameters,
            train_set=train_data,
            num_boost_round=1000,                      # Controla o total máximo de rodadas (combinado com o early stopping)
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )

    return model