"""
This is a boilerplate pipeline 'data_evaluation'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline  # noqa


from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs=[
                "lgbm_model",  # O objeto modelo salvo pelo node anterior (ou carregado via .pkl no catalog)
                "X_train", 
                "y_train",
                "X_valid", 
                "y_valid",
                "X_test", 
                "y_test"
            ],
            outputs="model_metrics",
            name="evaluate_model_node",
        )
    ])