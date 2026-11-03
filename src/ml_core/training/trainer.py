from ml_core.models.model_bundle import ModelBundle
from ml_core.training.model_pipeline import build_model_pipeline
from ml_core.training.scoring import evaluate_predictions, prediction_scores
from sklearn.utils import Bunch


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
        validation_data=None,
        evaluator=None,
    ):
        X = self.feature_selector.select(X) if self.feature_selector else X
        model_pipeline = build_model_pipeline(self.model, self.preprocessor)
        fit_params = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
        model_pipeline.fit(X, y, **fit_params)
        target_names = (
            list(y.columns)
            if hasattr(y, "columns")
            else [getattr(y, "name", None) or "target"]
        )
        bundle = ModelBundle(
            model_pipeline=model_pipeline,
            feature_pipeline=feature_pipeline,
            feature_names=list(X.columns),
            target_names=target_names,
        )
        evaluation_metrics = None
        if validation_data is not None and evaluator is not None:
            X_valid, y_valid = validation_data
            X_valid = self.feature_selector.select(X_valid) if self.feature_selector else X_valid
            y_pred = model_pipeline.predict(X_valid)
            y_score, score_labels = prediction_scores(model_pipeline, X_valid)
            evaluation_metrics = evaluate_predictions(
                evaluator,
                y_valid,
                y_pred,
                y_score=y_score,
                y_score_labels=score_labels,
            )
        return Bunch(model_bundle=bundle, evaluation_metrics=evaluation_metrics)
