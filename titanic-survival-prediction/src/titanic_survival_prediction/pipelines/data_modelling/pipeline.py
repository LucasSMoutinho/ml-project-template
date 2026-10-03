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
                "X_train_balanced", 
                "y_train_balanced", 
                "X_test",
                "y_test", 
                "params:params_lightgbm"
            ],
            outputs="lgbm_model",
            name="train_lightgbm_node",
        )
    ])