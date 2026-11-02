from kedro_ml_framework.business_logic.features import PriceReturnFeatures, RollingVolatilityFeature
from kedro_ml_framework.business_logic.labels import ForwardReturnLabels
from ml_core.data.dataset_builder import DatasetBuilder
from ml_core.data.feature_pipeline import create_feature_pipeline


def build_dataset(raw_data, *, dataset_id):
    """Wire this project's plugins; replace these objects for the actual project."""
    builder = DatasetBuilder(
        feature_pipeline=create_feature_pipeline([
            PriceReturnFeatures(price_column="close", time_column="date", entity_column="symbol"),
            RollingVolatilityFeature(
                price_column="close",
                time_column="date",
                entity_column="symbol",
                window=20,
            ),
        ]),
        label_builder=ForwardReturnLabels(
            price_column="close",
            time_column="date",
            entity_column="symbol",
            horizons=(1, 5, 20),
            threshold=0.0,
        ),
    )
    return builder.build(raw_data, dataset_id=dataset_id, source={"name": "catalog:raw_data"})
