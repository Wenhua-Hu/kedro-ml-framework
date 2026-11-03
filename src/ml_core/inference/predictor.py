import numpy as np
import pandas as pd


class Predictor:
    """Apply the saved feature pipeline, preprocessing, and model to new data."""

    def __init__(self, model_bundle):
        self.model_bundle = model_bundle

    def _transform_features(self, raw_data):
        bundle = self.model_bundle
        return (
            bundle.feature_pipeline.transform(raw_data)
            if bundle.feature_pipeline
            else raw_data
        )

    def predict(self, raw_data):
        return self.predict_features(self._transform_features(raw_data))

    def predict_proba(self, raw_data):
        """Return sklearn-native class probabilities for raw input data.

        The fitted estimator must implement ``predict_proba``. For multi-output
        classifiers, sklearn's list-of-arrays return format is preserved.
        """
        return self.predict_proba_features(self._transform_features(raw_data))

    def predict_features(self, features):
        bundle = self.model_bundle
        selected = features.loc[:, bundle.feature_names]
        predictions = np.asarray(bundle.model_pipeline.predict(selected))
        if predictions.ndim == 1:
            name = bundle.target_names[0] if bundle.target_names else None
            return pd.Series(predictions, index=selected.index, name=name)

        columns = bundle.target_names if bundle.target_names else None
        return pd.DataFrame(predictions, index=selected.index, columns=columns)

    def predict_proba_features(self, features):
        """Return sklearn-native probabilities for already generated features."""
        bundle = self.model_bundle
        if not hasattr(bundle.model_pipeline, "predict_proba"):
            estimator = bundle.model_pipeline
            if hasattr(estimator, "steps") and estimator.steps:
                estimator = estimator.steps[-1][1]
            raise AttributeError(
                f"{type(estimator).__name__} does not support predict_proba"
            )
        selected = features.loc[:, bundle.feature_names]
        return bundle.model_pipeline.predict_proba(selected)
