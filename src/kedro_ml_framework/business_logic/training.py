from datetime import datetime, timezone
from uuid import uuid4

from ml_core.models import create_model
from ml_core.training import FeatureSelector, Trainer, build_model_pipeline
from ml_core.data import select_training_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def standard_scaler(columns):
    return ColumnTransformer(
        [("standard_scaler", StandardScaler(), columns)],
        remainder="passthrough",
    )


def min_max_scaler(columns):
    return ColumnTransformer(
        [("min_max_scaler", MinMaxScaler(), columns)],
        remainder="passthrough",
    )


def create_preprocessor(transformers):
    return ColumnTransformer(transformers, remainder="passthrough")


def create_project_preprocessor():
    return create_preprocessor([
        (
            "standard",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            ["return_1d", "return_5d"],
        ),
        (
            "min_max",
            make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler()),
            ["volatility_20d"],
        ),
    ])


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
    )


def train_project_model(
    dataset_X,
    dataset_y,
    dataset_metadata,
    dataset_feature_pipeline,
    feature_columns,
    model_params,
):
    X_train, y_train = select_training_data(
        dataset_X,
        dataset_y,
        target_columns=["forward_up_5d"],
    )
    y_train = y_train["forward_up_5d"].astype(int)

    bundle = train_model(
        X_train,
        y_train,
        model_class=RandomForestClassifier,
        model_params=model_params,
        feature_columns=feature_columns,
        feature_pipeline=dataset_feature_pipeline,
        preprocessor=create_project_preprocessor(),
        sample_weight=balanced_sample_weights(y_train),
    )
    created_at = datetime.now(timezone.utc)
    model_metadata = {
        "model_id": f"{created_at.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        "dataset_id": dataset_metadata["dataset_id"],
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "model_type": f"{RandomForestClassifier.__module__}.{RandomForestClassifier.__qualname__}",
        "model_params": dict(model_params),
        "feature_names": bundle.feature_names,
        "target_names": bundle.target_names,
    }
    return bundle, model_metadata
