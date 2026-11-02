from kedro.pipeline import Pipeline, node, pipeline

from kedro_ml_framework.pipelines.data_processing.nodes import build_project_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_project_dataset,
                inputs=["raw_data", "params:dataset_id"],
                outputs=[
                    "dataset_X",
                    "dataset_y",
                    "dataset_metadata",
                    "dataset_feature_pipeline",
                ],
                name="build_dataset",
            )
        ]
    )
