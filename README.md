# kedro-ml-framwork

A Kedro project template that keeps workflow, reusable ML components, and project-specific logic separate.

## Structure

```text
src/
├── ml_core/                   # Reusable Python package; no Kedro or business imports
│   ├── data/                  # Feature composition, labels, dataset and metadata
│   ├── training/              # Feature selection and model training
│   ├── models/                # Inference bundle
│   ├── inference/             # Prediction orchestration
└── kedro_ml_framework/
    ├── pipelines/             # Kedro workflow orchestration
    └── business_logic/        # This project's features and labels
```

`ml_core` is a top-level Python package so it can be extracted without changing its imports. Feature objects implement sklearn's `fit(data, y=None)` / `transform(data)` contract and are composed with `sklearn.pipeline.Pipeline`; label builders implement `build(data)`, and estimators implement `fit` / `predict`. The example Feature transformers inherit `BaseEstimator` and `TransformerMixin`, so sklearn can clone them and expose their constructor parameters. This does not run parameter search; projects can use those capabilities only when needed. Other injected transformers can remain duck-typed unless clone/search support is required. Components are passed in by project code; there is no YAML feature language or automatic plugin discovery. `DatasetBuilder` runs the feature pipeline on the full source data before train/CV splitting, so these feature steps must be stateless: `fit()` must not learn data-dependent values. They may calculate row-wise or rolling candidate columns from the available input rows. Put learned transformations such as imputation, scaling, or learned encodings in the estimator's preprocessing pipeline; CV then fits them separately on each training fold.

Training responsibilities are split across the project modules: `business_logic/preprocessing.py` declares scaler choices and the project preprocessor; `business_logic/training.py` constructs and fits estimators; `business_logic/project_training.py` owns this project's temporal split and model metadata. The generic `create_preprocessor()` accepts sklearn `ColumnTransformer` entries, so other projects can choose different scalers and columns. Trainer combines the preprocessor and estimator into one sklearn `Pipeline`, fits it only on training data, and returns a scikit-learn `Bunch`: `model_bundle` contains the fitted pipeline, feature-generation pipeline, and ordered `feature_names` / `target_names`; `evaluation_metrics` contains optional held-out metrics. Predictor uses only the bundle at inference. Tree models such as LightGBM can omit scaling. Trainer does not own experiment tracking; Kedro-MLflow handles the Kedro run, while metric and model outputs use its catalog datasets when those need to be logged to MLflow.

Kedro-MLflow is configured in `conf/base/mlflow.yml` and uses the local `mlruns/` directory by default. Kedro 0.19 auto-registers the plugin hook. Run `kedro run --pipeline=train --params="dataset_id=run-001"` to build the dataset and train the example model. Training uses chronological train, calibration, and evaluation periods, with a five-date purge around each boundary. It fits the classifier on the first period, calibrates it with the next period, evaluates calibrated probabilities on the last period, and writes the calibrated bundle to `model_bundle/inference_bundle.pkl` with sibling metadata. The calibration method defaults to sigmoid and can be changed with `training.calibration.method`. Run `kedro run --pipeline=inference` to produce class predictions. Set `inference.method` to `predict_proba` in `conf/base/parameters.yml`, or run `kedro run --pipeline=inference --params="inference.method=predict_proba"`, to write calibrated probabilities instead. The inference node loads the bundle, applies its feature pipeline, selects the exact training columns, then invokes the chosen method on the bundled model pipeline. Probability outputs are written as a DataFrame with class names in its columns, compatible with the configured Parquet catalog. To log the bundle or metrics to MLflow as well, change the corresponding outputs to Kedro-MLflow catalog datasets.

Cross-validation is separate from `Trainer`, which fits one final model. `ml_core.training.CrossValidator` accepts an estimator factory, any splitter implementing `split(X, y, groups)`, and an evaluator; it returns per-fold metrics, numeric mean/std, OOF predictions/scores, and OOF metrics. The core does not select a CV strategy. Use sklearn `KFold` for independent samples, `StratifiedKFold` for single-label classification, `GroupKFold` for grouped samples, or `TimeSeriesSplit` for ordered rows. For this project's panel data, `business_logic/splitters.py` provides `DateGroupedTimeSeriesSplit`, which keeps all symbols on a date together; for 1/5/20-period targets configure `gap=20` trading dates. Project code can run CV to compare a configuration, then call `Trainer` once to fit the final model. `create_estimator_pipeline()` builds the same preprocessing/model Pipeline for each CV fold.

For independent, single-target binary classification, `StratifiedKFold` keeps class proportions similar across folds:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from ml_core.data import select_training_data
from ml_core.training import CrossValidator, FeatureSelector
from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.training import (
    balanced_sample_weights,
    create_estimator_pipeline,
)

feature_columns = ["feature_a", "feature_b"]
X_task, y_task = select_training_data(
    dataset.data, dataset.target, target_columns=["forward_up_5d"]
)
X_cv = FeatureSelector(feature_columns).select(X_task)
y_cv = y_task["forward_up_5d"]  # one binary target column
cv = CrossValidator(
    estimator_factory=lambda: create_estimator_pipeline(
        RandomForestClassifier,
        {"n_estimators": 200, "random_state": 42},
    ),
    splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evaluator=BinaryClassificationEvaluator(),
)
report = cv.evaluate(
    X_cv,
    y_cv,
    sample_weight_factory=balanced_sample_weights,
)
report.fold_metrics       # metrics for each fold
report.summary            # mean and std across folds
report.oof_metrics        # metrics on all out-of-fold predictions
report.oof_predictions    # index, y_true, y_pred, and fold for each OOF row
```

Use this for independent samples. For chronological data, including thresholded stock labels, keep the time-based splitter to preserve the past-to-future evaluation order.

The example's `balanced_sample_weights(y)` is a project-level choice for a single imbalanced classification target. Pass it as `sample_weight_factory` during CV so each fold computes weights from its own training labels. Validation metrics are calculated on the original validation rows. For final training, compute one weight vector from the full training labels and pass it as `sample_weight`:

```python
training_result = train_model(
    X_train,
    y_train,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 200, "random_state": 42},
    feature_columns=feature_columns,
    feature_pipeline=dataset.feature_pipeline,
    sample_weight=balanced_sample_weights(y_train),
)
bundle = training_result.model_bundle
```

The estimator must accept `sample_weight` in `fit`. If it has a native setting such as `class_weight`, configure that in `model_params` instead; choosing between these approaches remains project code.

```python
from sklearn.ensemble import RandomForestClassifier
from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.splitters import DateGroupedTimeSeriesSplit
from kedro_ml_framework.business_logic.training import (
    balanced_sample_weights,
    create_estimator_pipeline,
)
from kedro_ml_framework.business_logic.preprocessing import create_project_preprocessor
from ml_core.data import select_training_data
from ml_core.training import CrossValidator, FeatureSelector

X_task, y_task = select_training_data(
    dataset.data, dataset.target, target_columns=["forward_up_5d"]
)
X_cv = FeatureSelector(["return_1d", "return_5d", "volatility_20d"]).select(X_task)
y_cv = y_task["forward_up_5d"]
cv = CrossValidator(
    estimator_factory=lambda: create_estimator_pipeline(
        RandomForestClassifier,
        {"n_estimators": 200, "random_state": 42},
        preprocessor=create_project_preprocessor(),
    ),
    splitter=DateGroupedTimeSeriesSplit(n_splits=5, gap=5),
    evaluator=BinaryClassificationEvaluator(),
)
report = cv.evaluate(
    X_cv,
    y_cv,
    groups=X_task.loc[X_cv.index, "date"],
    sample_weight_factory=balanced_sample_weights,
)
```

### Train without CV

Pass a held-out validation set and an evaluator to `train_model()`. The trainer fits only on `X_train` / `y_train` and predicts the validation set. It returns a scikit-learn `Bunch` with separate `model_bundle` and `evaluation_metrics` fields, so evaluation results stay outside the inference artifact. In the Kedro example, the bundle and metrics are also returned as separate catalog outputs.

```python
from sklearn.ensemble import RandomForestClassifier
from ml_core.inference import Predictor
from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.training import (
    balanced_sample_weights,
    train_model,
)
from kedro_ml_framework.business_logic.preprocessing import create_project_preprocessor

training_result = train_model(
    X_train,
    y_train,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 200, "random_state": 42},
    feature_columns=["return_1d", "return_5d", "volatility_20d"],
    feature_pipeline=dataset.feature_pipeline,
    preprocessor=create_project_preprocessor(),
    sample_weight=balanced_sample_weights(y_train),
    validation_data=(X_valid, y_valid),
    evaluator=BinaryClassificationEvaluator(),
)
bundle = training_result.model_bundle
predictions = Predictor(bundle).predict(new_raw_data)
probabilities = Predictor(bundle).predict_proba(new_raw_data)
metrics = training_result.evaluation_metrics
```

`predict_proba()` applies the same feature pipeline and training-time column selection as `predict()`, then calls the fitted estimator pipeline's `predict_proba`. It returns sklearn's native format: a probability array for one target, or a list of arrays for multi-output classification. Models without `predict_proba` raise `AttributeError`.

The project Kedro training flow calibrates its classifier by default. Other projects can use the same helper as optional post-fit processing: fit the ordinary model on an earlier period, then pass a later, disjoint calibration period to `calibrate_model_bundle()`. It returns a new `ModelBundle` whose model pipeline contains sklearn's fitted calibrator; `Predictor.predict_proba()` then returns calibrated probabilities. Keep a still later period for evaluation if you need to measure calibration quality. For forward labels, leave a date gap around each split at least as long as the target horizon. This uses `FrozenEstimator` and requires scikit-learn 1.6 or newer.

```python
from kedro_ml_framework.business_logic import calibrate_model_bundle

# X_calibration contains DatasetBuilder-generated feature columns.
calibrated_bundle = calibrate_model_bundle(
    training_result.model_bundle,
    X_calibration,
    y_calibration,
    method="sigmoid",  # or "isotonic" when calibration data is ample
)
probabilities = Predictor(calibrated_bundle).predict_proba(new_raw_data)
```

The helper only handles the post-fit calibration step; project code owns the chronological train/calibration/evaluation splits. It does not alter `Trainer` or add another CV layer. Do not use metrics computed before calibration as a report of calibrated probabilities.

The classification evaluator returns accuracy, precision, recall, F1, and a confusion matrix with its class order. For binary classification, it also calculates ROC-AUC and average precision (`pr_auc`) when class scores are available and the validation set contains both classes. For multi-class classification, use `ClassificationEvaluator(average="macro", auc_average="macro")`; it calculates macro/micro/weighted class metrics, one-vs-rest ROC-AUC, and averaged per-class average precision when the validation split contains every class. The regression evaluator returns MAE and RMSE, including per-target values for multi-output regression.

An evaluator is a duck-typed object with `evaluate(y_true, y_pred)` that returns a metrics dictionary. A score-aware evaluator can also declare keyword arguments `y_score` and/or `y_score_labels`. `y_score` is the estimator's native `predict_proba` or `decision_function` output; `y_score_labels` gives the class order for its score columns when available. `evaluate_predictions()` detects which optional keywords the evaluator accepts and passes only those, so evaluators that implement just the two-argument contract remain usable.

These are distinct support cases: single-target multi-class classification (one mutually exclusive class per row) is supported, including CV alignment of class-score columns across folds. Multi-output regression is supported with per-target MAE/RMSE. Multiple independent classification targets can use sklearn's `MultiOutputClassifier` with `MultiOutputClassificationEvaluator`; CV aligns each target's class-score columns independently and returns per-target metrics plus their mean. `Predictor.predict_proba()` preserves sklearn's list-of-arrays probability format for multi-output classifiers.

The included Kedro training example uses the earliest period to fit the model, the next 20% of unique dates to fit the calibrator, and the last 20% to evaluate calibrated probabilities. It excludes five dates before both calibration and evaluation periods, matching the five-day forward label horizon. This temporal split is project-level logic; other projects can choose a different calibration strategy. The Kedro training node saves the calibrated bundle under the `model_bundle` catalog name. Run `kedro run --pipeline=inference --params="inference.method=predict_proba"` to load it and produce calibrated probabilities. The standalone Python example passes its bundle directly to `Predictor`.

### Train with CV

Run CV on the training period only. It reports fold mean/std and OOF metrics; then fit one final model with `Trainer` through `train_model()`.

```python
from sklearn.ensemble import RandomForestClassifier
from ml_core.data import select_training_data
from ml_core.training import CrossValidator, FeatureSelector
from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.splitters import DateGroupedTimeSeriesSplit
from kedro_ml_framework.business_logic.training import (
    balanced_sample_weights,
    create_estimator_pipeline,
    train_model,
)
from kedro_ml_framework.business_logic.preprocessing import create_project_preprocessor

# These are the training-period rows after reserving the final holdout period.
feature_columns = ["return_1d", "return_5d", "volatility_20d"]
X_task, y_task = select_training_data(
    X_training_period,
    y_training_period,
    target_columns=["forward_up_5d"],
)
X_cv = FeatureSelector(feature_columns).select(X_task)
y_cv = y_task["forward_up_5d"]

cv = CrossValidator(
    estimator_factory=lambda: create_estimator_pipeline(
        RandomForestClassifier,
        {"n_estimators": 200, "random_state": 42},
        preprocessor=create_project_preprocessor(),
    ),
    splitter=DateGroupedTimeSeriesSplit(n_splits=5, gap=20),
    evaluator=BinaryClassificationEvaluator(),
)
report = cv.evaluate(
    X_cv,
    y_cv,
    groups=X_task.loc[X_cv.index, "date"],
    sample_weight_factory=balanced_sample_weights,
)

training_result = train_model(
    X_task,
    y_cv,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 200, "random_state": 42},
    feature_columns=feature_columns,
    feature_pipeline=dataset.feature_pipeline,
    preprocessor=create_project_preprocessor(),
    sample_weight=balanced_sample_weights(y_cv),
)
bundle = training_result.model_bundle
```

## Dataset and experiment artifacts

Each built dataset consists of `X.parquet`, `y.parquet`, `metadata.json`, and `feature_pipeline.pkl` in one dataset directory. The training pipeline consumes these outputs together; `inference_bundle.pkl` embeds that same feature pipeline alongside the fitted preprocessing/model pipeline and selected columns. Its `model_metadata.json` records `model_id`, `dataset_id`, UTC creation time, model type/parameters, and ordered feature/target names. If not supplied, the dataset ID is generated as a UTC timestamp plus a short UUID (for example, `20261102_194512_a1b2c3d4`). Kedro's Data Catalog owns the actual I/O, so the same entries can point to local files or S3 paths. The example catalog uses a local path; change the paths for S3 and provide credentials through the environment. Pass a fresh `dataset_id` runtime parameter for each build so an existing dataset is not overwritten.

Metadata describes the dataset, candidate feature names, label names, and source. Kedro-MLflow tracks Kedro runs and parameters; use its metric and model catalog datasets for outputs that should be logged to MLflow. The bundle keeps feature transformation, preprocessing, and the estimator together for inference.

## Project-specific code

`DatasetBuilder` returns a scikit-learn `Bunch`, following the built-in dataset loader pattern. You can use either `dataset.data` / `dataset.target` or `dataset.X` / `dataset.y`; both pairs refer to the same objects. It also provides `dataset.frame`, `dataset.feature_names`, `dataset.target_names`, `dataset.metadata`, and `dataset.DESCR`. The data and target retain DataFrame/Series types, including for multiple targets.

`DatasetBuilder` keeps rows with missing target values because a later experiment may use only one of several labels. Select the targets for the current task and drop rows missing those targets with `select_training_data`. For feature missing values, the project preprocessor below uses a median imputer before each scaler; the imputer is fitted inside the model Pipeline, so each CV fold learns it only from that fold's training rows. If a feature is required to exist (for example, a rolling feature whose lookback window is incomplete), list it in `required_feature_columns` to drop those rows instead of imputing it.

```python
from ml_core.data import select_training_data

# One-label experiment: unrelated missing labels do not discard these rows.
X_train, y_train = select_training_data(
    dataset.data,
    dataset.target,
    target_columns=["forward_up_5d"],
)

# Multi-target selection: keep rows with all selected targets.
X_multi, y_multi = select_training_data(
    dataset.data,
    dataset.target,
    target_columns=["forward_up_1d", "forward_up_5d"],
)

# Optional: drop rows where a feature's history window is incomplete.
X_ready, y_ready = select_training_data(
    dataset.data,
    dataset.target,
    target_columns=["forward_up_5d"],
    required_feature_columns=["volatility_20d"],
)
```

Selecting multiple target columns only prepares the data; it does not by itself mean multi-label classification metrics are supported. For independent binary targets, each row may have several positive labels, which differs from a single multi-class target where each row has one class.

The included business example expects ordered price data with `date`, `symbol`, and `close` columns. It composes price-return and rolling-volatility features, then generates three binary targets (`forward_up_1d`, `forward_up_5d`, and `forward_up_20d`) using a zero return threshold. Set `threshold=None` on `ForwardReturnLabels` to produce the corresponding regression targets instead. `DatasetBuilder` keeps all transformed columns except target columns in `data`; it does not select a training feature subset. Key columns such as `symbol` and `date`, and label-construction inputs such as `close`, remain available in the dataset and are excluded by the training `feature_columns` configuration. The builder aligns data and targets by index and preserves missing targets; `select_training_data` drops rows missing the target columns selected for a particular training task. `ExistingLabels` shows how to use one or several label columns already present in the source data instead.

Add or remove feature objects in `business_logic/dataset.py`, or implement another object with `transform(data)`. For a different target definition, implement `build(data)` and return a Series for one target or a DataFrame for multiple targets. `ml_core` does not import business code, Kedro, or a storage implementation, so it can later be extracted as a Python package. For multiple targets, choose an estimator that supports multi-output or wrap the estimator with an appropriate adapter.

This is a starter layout and a small set of core components. Model choice, feature logic, evaluator, and environment-specific storage remain project decisions. Supply a new `dataset_id` to each Kedro run, for example `kedro run --params="dataset_id=run-001"`; the catalog uses it in the dataset path.
