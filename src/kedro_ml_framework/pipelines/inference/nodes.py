from ml_core.inference import Predictor


def predict(model_bundle, inference_data):
    """Predict from raw input using the complete, catalog-loaded model bundle."""
    return Predictor(model_bundle).predict(inference_data)
