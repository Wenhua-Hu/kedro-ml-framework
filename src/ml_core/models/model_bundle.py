class ModelBundle:
    """Container for fitted feature generation, selected columns, and model pipeline."""

    def __init__(
        self,
        model_pipeline,
        *,
        feature_pipeline=None,
        feature_names=None,
        target_names=None,
    ):
        self.model_pipeline = model_pipeline
        self.feature_pipeline = feature_pipeline
        self.feature_names = feature_names
        self.target_names = target_names
