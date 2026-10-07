"""
This is a boilerplate pipeline 'data_hyperparameters'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import optimize_hyperparameters, fit_model_hyperparameters

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=optimize_hyperparameters,
            inputs=[
                "X_train",
                "y_train",
                "params:hyperparameters_options" # Parâmetros definidos no YAML
            ],
            outputs=["best_lgbm_model", "best_hyperparameters"],
            name="optimize_hyperparameters_node",
        ),
        node(func=fit_model_hyperparameters,
            inputs=[
                "X_train",
                "y_train",
                "X_valid",
                "y_valid",
                "best_hyperparameters"
            ],
            outputs="final_model",
            name="fit_model_hyperparameters_node",
        )   
    ])