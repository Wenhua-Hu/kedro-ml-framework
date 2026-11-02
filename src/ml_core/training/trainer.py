from ml_core.models.model_bundle import ModelBundle
from ml_core.training.model_pipeline import build_model_pipeline


class Trainer:
    """Fit an estimator pipeline and return it with feature metadata."""

    def __init__(self, model, *, feature_selector=None, preprocessor=None):
        self.model = model
        self.feature_selector = feature_selector
        self.preprocessor = preprocessor

    def fit(
        self,
        X,
        y,
        *,
        feature_pipeline=None,
        sample_weight=None,
    ):
        X = self.feature_selector.select(X) if self.feature_selector else X
        model_pipeline = build_model_pipeline(self.model, self.preprocessor)
        fit_params = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
        model_pipeline.fit(X, y, **fit_params)
        target_names = list(y.columns) if hasattr(y, "columns") else [y.name]
        return ModelBundle(
            model_pipeline=model_pipeline,
            feature_pipeline=feature_pipeline,
            feature_names=list(X.columns),
            target_names=target_names,
        )
