from kedro.pipeline import Pipeline, node

from kedro_ml_framework.pipelines.training.nodes import train_project_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_project_model,
            inputs=[
                "dataset_X",
                "dataset_y",
                "dataset_metadata",
                "dataset_feature_pipeline",
                "params:training.feature_columns",
                "params:training.model_params",
                "params:training.calibration.method",
            ],
            outputs=["model_bundle", "model_metadata", "evaluation"],
            name="train_model",
        )
    ])
