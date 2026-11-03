"""Project-level helpers for estimator construction and fitting."""

from ml_core.models import create_model
from ml_core.training import FeatureSelector, Trainer, build_model_pipeline
from sklearn.utils.class_weight import compute_sample_weight


def balanced_sample_weights(y):
    """Project choice: give each class equal total weight for one target."""
    return compute_sample_weight(class_weight="balanced", y=y)


def create_estimator_pipeline(model_class, model_params, preprocessor=None):
    model = create_model(model_class, model_params)
    return build_model_pipeline(model, preprocessor)


def train_model(
    X_train,
    y_train,
    *,
    model_class,
    model_params,
    feature_columns,
    feature_pipeline,
    preprocessor=None,
    sample_weight=None,
    validation_data=None,
    evaluator=None,
):
    trainer = Trainer(
        model=create_model(model_class, model_params),
        feature_selector=FeatureSelector(feature_columns),
        preprocessor=preprocessor,
    )
    return trainer.fit(
        X_train,
        y_train,
        feature_pipeline=feature_pipeline,
        sample_weight=sample_weight,
        validation_data=validation_data,
        evaluator=evaluator,
    )
