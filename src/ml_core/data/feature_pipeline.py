from sklearn.pipeline import make_pipeline


def create_feature_pipeline(features):
    """Build an sklearn Pipeline from project feature transformers."""
    return make_pipeline(*features)
