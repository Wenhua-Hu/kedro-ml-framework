import numpy as np
import pandas as pd


class Predictor:
    """Apply the saved feature pipeline, preprocessing, and model to new data."""

    def __init__(self, model_bundle):
        self.model_bundle = model_bundle

    def predict(self, raw_data):
        bundle = self.model_bundle
        features = (
            bundle.feature_pipeline.transform(raw_data)
            if bundle.feature_pipeline
            else raw_data
        )
        return self.predict_features(features)

    def predict_features(self, features):
        bundle = self.model_bundle
        selected = features.loc[:, bundle.feature_names]
        predictions = np.asarray(bundle.model_pipeline.predict(selected))
        if predictions.ndim == 1:
            name = bundle.target_names[0] if bundle.target_names else None
            return pd.Series(predictions, index=selected.index, name=name)

        columns = bundle.target_names if bundle.target_names else None
        return pd.DataFrame(predictions, index=selected.index, columns=columns)
