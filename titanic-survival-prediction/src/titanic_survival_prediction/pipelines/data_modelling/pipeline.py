"""
This is a boilerplate pipeline 'data_modelling'
generated using Kedro 1.7.0
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_lightgbm_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_lightgbm_model,
            inputs=[
                "X_train",
                "y_train",
                "X_valid", 
                "y_valid", 
                "params:params_lightgbm"
            ],
            outputs="lgbm_model",
            name="train_lightgbm_node",
        )
    ])