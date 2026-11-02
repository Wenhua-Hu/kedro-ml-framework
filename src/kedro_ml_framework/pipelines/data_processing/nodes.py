from kedro_ml_framework.business_logic.dataset import build_dataset


def build_project_dataset(raw_data, dataset_id):
    dataset = build_dataset(raw_data, dataset_id=dataset_id)
    return dataset.data, dataset.target, dataset.metadata, dataset.feature_pipeline
