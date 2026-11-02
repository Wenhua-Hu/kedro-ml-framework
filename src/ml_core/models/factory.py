def create_model(model_class, params=None):
    """Construct a project-selected estimator with its own keyword parameters."""
    return model_class(**(params or {}))
