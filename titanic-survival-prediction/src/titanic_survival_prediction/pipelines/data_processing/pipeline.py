from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, balance_and_format_train_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["water_potability", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
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
