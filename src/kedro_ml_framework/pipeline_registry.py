from kedro.pipeline import Pipeline

from kedro_ml_framework.pipelines.data_processing import create_pipeline
from kedro_ml_framework.pipelines.inference import create_pipeline as create_inference_pipeline
from kedro_ml_framework.pipelines.training import create_pipeline as create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_processing = create_pipeline()
    training = data_processing + create_training_pipeline()
    inference = create_inference_pipeline()
    return {
        "__default__": training,
        "data_processing": data_processing,
        "train": training,
        "inference": inference,
    }
