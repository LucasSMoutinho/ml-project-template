from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, balance_train_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["water_potability", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=balance_train_data,
            inputs=["X_train", "y_train"],
            outputs=["X_train_balanced", "y_train_balanced"],
            name="balance_data_node",
        )
])      