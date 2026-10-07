from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    balance_and_format_train_data,
    split_data,
    summarize_data_quality,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=summarize_data_quality,
            inputs=["water_potability", "params:model_options"],
            outputs="data_quality_report",
            name="summarize_data_quality_node",
        ),
        node(
            func=split_data,
            inputs=["water_potability", "params:model_options"],
            outputs=[
                "X_train",
                "X_test",
                "X_valid",
                "y_train",
                "y_test",
                "y_valid",
            ],
            name="split_data_node",
        ),
        node(
            func=balance_and_format_train_data,
            inputs=[
                "X_train",
                "y_train",
                "params:model_options"       # O dicionário de parâmetros contendo as 'features' e o 'target_column'
            ],
            outputs=[
                "X_train_balanced",          # Dataset de saída catalogado
                "y_train_balanced"           # Dataset de saída catalogado
            ],
            name="balance_and_format_train_data_node",
        )
])      
