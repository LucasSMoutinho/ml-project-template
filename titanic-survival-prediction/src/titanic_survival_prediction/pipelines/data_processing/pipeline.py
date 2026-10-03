from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["seu_dataframe_processado", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        )
    ])  