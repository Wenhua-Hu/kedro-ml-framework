"""Project-level post-fit probability calibration helpers."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from ml_core.models import ModelBundle


def calibrate_model_bundle(
    model_bundle,
    X_calibration,
    y_calibration,
    *,
    method="sigmoid",
):
    """Return a bundle whose fitted classifier produces calibrated probabilities.

    Calibration data must be disjoint from the data used to fit the bundled
    model. ``X_calibration`` contains generated feature columns, as in the
    dataset returned by ``DatasetBuilder``. Supports one classification target.
    """
    X_calibration = X_calibration.loc[:, model_bundle.feature_names]
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=FrozenEstimator(model_bundle.model_pipeline),
        method=method,
    )
    calibrated_pipeline.fit(X_calibration, y_calibration)

    return ModelBundle(
        model_pipeline=calibrated_pipeline,
        feature_pipeline=model_bundle.feature_pipeline,
        feature_names=model_bundle.feature_names,
        target_names=model_bundle.target_names,
    )
