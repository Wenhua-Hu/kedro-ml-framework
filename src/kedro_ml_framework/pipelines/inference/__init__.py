from kedro.pipeline import Pipeline, node

from kedro_ml_framework.pipelines.inference.nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=predict,
            inputs=["model_bundle", "inference_data"],
            outputs="predictions",
            name="predict",
        )
    ])
