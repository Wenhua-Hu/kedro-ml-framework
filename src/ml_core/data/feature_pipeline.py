from sklearn.pipeline import make_pipeline


def create_feature_pipeline(features):
    """Build a pipeline for stateless candidate-feature generation.

    DatasetBuilder runs this pipeline before train/CV splitting, so its
    ``fit`` methods must not learn values from the input data. Put learned
    transformations in the estimator's preprocessing pipeline instead; CV
    then fits them independently on each training fold. Components only need
    sklearn's BaseEstimator parameter interface when callers want clone or
    parameter-search support.
    """
    return make_pipeline(*features)
