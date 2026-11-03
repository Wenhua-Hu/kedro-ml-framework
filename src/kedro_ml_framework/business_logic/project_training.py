"""Orchestrate this project's temporal split, model fit, and metadata."""

from datetime import datetime, timezone
from uuid import uuid4

from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.calibration import calibrate_model_bundle
from kedro_ml_framework.business_logic.preprocessing import create_project_preprocessor
from kedro_ml_framework.business_logic.training import balanced_sample_weights, train_model
from ml_core.data import select_training_data
from sklearn.ensemble import RandomForestClassifier


def train_project_model(
    dataset_X,
    dataset_y,
    dataset_metadata,
    dataset_feature_pipeline,
    feature_columns,
    model_params,
    calibration_method="sigmoid",
):
    dates = dataset_X["date"].drop_duplicates().sort_values().reset_index(drop=True)
    evaluation_size = max(1, round(len(dates) * 0.2))
    calibration_size = max(1, round(len(dates) * 0.2))
    evaluation_start = len(dates) - evaluation_size
    calibration_end = evaluation_start - 5  # purge the five-day target horizon
    calibration_start = calibration_end - calibration_size
    training_end = calibration_start - 5  # purge before calibration period

    training_dates = dates.iloc[:training_end]
    calibration_dates = dates.iloc[calibration_start:calibration_end]
    evaluation_dates = dates.iloc[evaluation_start:]

    training_mask = dataset_X["date"].isin(training_dates)
    calibration_mask = dataset_X["date"].isin(calibration_dates)
    evaluation_mask = dataset_X["date"].isin(evaluation_dates)
    X_train_raw, y_train_raw = dataset_X.loc[training_mask], dataset_y.loc[training_mask]
    X_calibration_raw = dataset_X.loc[calibration_mask]
    y_calibration_raw = dataset_y.loc[calibration_mask]
    X_evaluation_raw = dataset_X.loc[evaluation_mask]
    y_evaluation_raw = dataset_y.loc[evaluation_mask]

    X_train, y_train = select_training_data(
        X_train_raw,
        y_train_raw,
        target_columns=["forward_up_5d"],
    )
    y_train = y_train["forward_up_5d"].astype(int)
    X_calibration, y_calibration = select_training_data(
        X_calibration_raw,
        y_calibration_raw,
        target_columns=["forward_up_5d"],
    )
    y_calibration = y_calibration["forward_up_5d"].astype(int)
    X_evaluation, y_evaluation = select_training_data(
        X_evaluation_raw,
        y_evaluation_raw,
        target_columns=["forward_up_5d"],
    )
    y_evaluation = y_evaluation["forward_up_5d"].astype(int)

    training_result = train_model(
        X_train,
        y_train,
        model_class=RandomForestClassifier,
        model_params=model_params,
        feature_columns=feature_columns,
        feature_pipeline=dataset_feature_pipeline,
        preprocessor=create_project_preprocessor(),
        sample_weight=balanced_sample_weights(y_train),
    )
    model_bundle = calibrate_model_bundle(
        training_result.model_bundle,
        X_calibration,
        y_calibration,
        method=calibration_method,
    )
    evaluation_X = X_evaluation.loc[:, model_bundle.feature_names]
    evaluation_model = model_bundle.model_pipeline
    evaluation_metrics = BinaryClassificationEvaluator().evaluate(
        y_evaluation,
        evaluation_model.predict(evaluation_X),
        y_score=evaluation_model.predict_proba(evaluation_X),
        y_score_labels=evaluation_model.classes_,
    )

    created_at = datetime.now(timezone.utc)
    model_metadata = {
        "model_id": f"{created_at.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
        "dataset_id": dataset_metadata["dataset_id"],
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "model_type": f"{RandomForestClassifier.__module__}.{RandomForestClassifier.__qualname__}",
        "model_params": dict(model_params),
        "feature_names": model_bundle.feature_names,
        "target_names": model_bundle.target_names,
        "probability_calibration": {
            "method": calibration_method,
            "calibration_start": str(calibration_dates.iloc[0]),
            "calibration_end": str(calibration_dates.iloc[-1]),
        },
    }
    return (
        model_bundle,
        model_metadata,
        evaluation_metrics,
    )
