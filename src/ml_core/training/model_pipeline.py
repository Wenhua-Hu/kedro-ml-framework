from sklearn.pipeline import Pipeline


def build_model_pipeline(model, preprocessor=None):
    steps = []
    if preprocessor is not None:
        steps.append(("preprocessing", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps)
