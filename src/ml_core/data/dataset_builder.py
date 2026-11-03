from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from ml_core.data.dataset import make_dataset


def _new_dataset_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}"


class DatasetBuilder:
    """Build candidate features and labels using injected project components.

    Feature steps here must be stateless: ``fit`` may be a no-op, but must not
    learn values from the dataset. Learned transformations belong in the
    estimator pipeline so cross-validation fits them separately in each fold.
    """

    def __init__(self, feature_pipeline, label_builder):
        self.feature_pipeline = feature_pipeline
        self.label_builder = label_builder

    def build(self, raw_data, *, dataset_id=None, source=None):
        # This pipeline builds reusable candidate columns before train/CV splits.
        # Its steps must not learn dataset-level state here.
        transformed = self.feature_pipeline.fit_transform(raw_data)
        y = self.label_builder.build(transformed)
        label_columns = list(y.columns) if isinstance(y, pd.DataFrame) else [y.name]
        # Keep incomplete targets in the candidate dataset. A training task may
        # use only one of several labels, so filtering belongs after target selection.
        X = transformed.drop(columns=label_columns, errors="ignore")

        metadata = {
            "dataset_id": dataset_id or _new_dataset_id(),
            "source": source or {},
            "feature_pipeline": type(self.feature_pipeline).__name__,
            "label_builder": type(self.label_builder).__name__,
            "feature_columns": list(X.columns),
            "label_columns": label_columns,
            "row_count": len(X),
        }
        return make_dataset(
            data=X,
            target=y,
            metadata=metadata,
            description="Feature and target dataset built by ml_core.",
            feature_pipeline=self.feature_pipeline,
        )
