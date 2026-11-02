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

`ml_core` is a top-level Python package so it can be extracted without changing its imports. Feature objects implement sklearn's `fit(data, y=None)` / `transform(data)` contract and are composed with `sklearn.pipeline.Pipeline`; label builders implement `build(data)`, and estimators implement `fit` / `predict`. Components are passed in by project code; there is no YAML feature language or automatic plugin discovery. Feature steps in this example are stateless; learned preprocessing belongs in the training-time preprocessing pipeline so it is fit only on training data.

For training, `business_logic/training.py` shows the wiring: select experiment features with `FeatureSelector`, construct the chosen estimator with `create_model(ModelClass, params)`, and pass those pieces to `Trainer`. `create_project_preprocessor()` is where this project's scaler choices and columns are declared; pass its result as `preprocessor` when calling `train_model()`. The generic `create_preprocessor()` accepts sklearn `ColumnTransformer` entries, so other projects can choose different scalers and columns. Trainer combines the preprocessor and estimator into one sklearn `Pipeline`, fits it only on training data, and returns a `ModelBundle` containing that fitted pipeline, the feature-generation pipeline, and ordered `feature_names` / `target_names`. Predictor reuses the feature pipeline and selected feature names at inference. Tree models such as LightGBM can omit scaling. Trainer does not own experiment tracking; Kedro-MLflow handles the Kedro run, while metric and model outputs use its catalog datasets when those need to be logged to MLflow.

Kedro-MLflow is configured in `conf/base/mlflow.yml` and uses the local `mlruns/` directory by default. Kedro 0.19 auto-registers the plugin hook. Run `kedro run --pipeline=train --params="dataset_id=run-001"` to build the dataset and train the example model. Training writes `model_bundle/inference_bundle.pkl` and a sibling `model_bundle/model_metadata.json`. Run `kedro run --pipeline=inference` to load that bundle and `inference_data`; `Predictor` applies the bundled feature pipeline, selects the exact training columns, then runs the bundled preprocessing/model pipeline. To log the bundle or metrics to MLflow as well, change the corresponding outputs to Kedro-MLflow catalog datasets.

Cross-validation is separate from `Trainer`, which fits one final model. `ml_core.training.CrossValidator` accepts an estimator factory, any splitter implementing `split(X, y, groups)`, and an evaluator; it returns per-fold metrics, mean/std, OOF predictions, and OOF metrics. The core does not select a CV strategy. Use sklearn `KFold` for independent samples, `StratifiedKFold` for single-label classification, `GroupKFold` for grouped samples, or `TimeSeriesSplit` for ordered rows. For this project's panel data, `business_logic/splitters.py` provides `DateGroupedTimeSeriesSplit`, which keeps all symbols on a date together; for 1/5/20-period targets configure `gap=20` trading dates. Project code can run CV to compare a configuration, then call `Trainer` once to fit the final model. `create_estimator_pipeline()` builds the same preprocessing/model Pipeline for each CV fold.

For independent, single-target binary classification, `StratifiedKFold` keeps class proportions similar across folds:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from ml_core.data import select_training_data
from ml_core.training import CrossValidator, FeatureSelector
from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.training import create_estimator_pipeline
from kedro_ml_framework.business_logic.training import balanced_sample_weights

feature_columns = ["feature_a", "feature_b"]
X_task, y_task = select_training_data(
    dataset.data, dataset.target, target_columns=["forward_up_5d"]
)
X_cv = FeatureSelector(feature_columns).select(X_task)
y_cv = y_task["forward_up_5d"]  # one binary target column
sample_weight = balanced_sample_weights(y_cv)
cv = CrossValidator(
    estimator_factory=lambda: create_estimator_pipeline(
        RandomForestClassifier,
        {"n_estimators": 200, "random_state": 42},
    ),
    splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    evaluator=BinaryClassificationEvaluator(),
)
report = cv.evaluate(X_cv, y_cv, sample_weight=sample_weight)
report.fold_metrics       # metrics for each fold
report.summary            # mean and std across folds
report.oof_metrics        # metrics on all out-of-fold predictions
report.oof_predictions    # index, y_true, y_pred, and fold for each OOF row
```

Use this for independent samples. For chronological data, including thresholded stock labels, keep the time-based splitter to preserve the past-to-future evaluation order.

The example's `balanced_sample_weights(y)` is a project-level choice for a single imbalanced classification target. `CrossValidator` slices those weights by each fold's training rows and passes them to the estimator; validation metrics remain calculated on the original validation rows. For final training, pass the weights the same way:

```python
bundle = train_model(
    X_train,
    y_train,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 200, "random_state": 42},
    feature_columns=feature_columns,
    feature_pipeline=dataset.feature_pipeline,
    sample_weight=balanced_sample_weights(y_train),
)
```

The estimator must accept `sample_weight` in `fit`. If it has a native setting such as `class_weight`, configure that in `model_params` instead; choosing between these approaches remains project code.

```python
from sklearn.ensemble import RandomForestClassifier
from kedro_ml_framework.business_logic.evaluation import BinaryClassificationEvaluator
from kedro_ml_framework.business_logic.splitters import DateGroupedTimeSeriesSplit
from kedro_ml_framework.business_logic.training import (
    balanced_sample_weights,
    create_estimator_pipeline,
    create_project_preprocessor,
)
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
    sample_weight=balanced_sample_weights(y_cv),
)
```

### Train without CV

Assume `X_train` and `y_train` have already been selected from the training period.

```python
from sklearn.ensemble import RandomForestClassifier
from ml_core.inference import Predictor
from kedro_ml_framework.business_logic.training import (
    balanced_sample_weights,
    create_project_preprocessor,
    train_model,
)

bundle = train_model(
    X_train,
    y_train,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 200, "random_state": 42},
    feature_columns=["return_1d", "return_5d", "volatility_20d"],
    feature_pipeline=dataset.feature_pipeline,
    preprocessor=create_project_preprocessor(),
    sample_weight=balanced_sample_weights(y_train),
)
predictions = Predictor(bundle).predict(new_raw_data)
```

When this bundle is saved under the `model_bundle` catalog name, run `kedro run --pipeline=inference`; Kedro loads the pickle and injects the bundle into the inference node. The standalone Python example above passes the returned bundle directly to the same `Predictor` API.

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
    create_project_preprocessor,
    train_model,
)

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
    sample_weight=balanced_sample_weights(y_cv),
)

bundle = train_model(
    X_task,
    y_cv,
    model_class=RandomForestClassifier,
    model_params={"n_estimators": 200, "random_state": 42},
    feature_columns=feature_columns,
    feature_pipeline=dataset.feature_pipeline,
    preprocessor=create_project_preprocessor(),
    sample_weight=balanced_sample_weights(y_cv),
)
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

# Multi-label experiment: rows must have all selected labels.
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

The included business example expects ordered price data with `date`, `symbol`, and `close` columns. It composes price-return and rolling-volatility features, then generates three binary targets (`forward_up_1d`, `forward_up_5d`, and `forward_up_20d`) using a zero return threshold. Set `threshold=None` on `ForwardReturnLabels` to produce the corresponding regression targets instead. `DatasetBuilder` keeps all transformed columns except target columns in `data`; it does not select a training feature subset. Key columns such as `symbol` and `date`, and label-construction inputs such as `close`, remain available in the dataset and are excluded by the training `feature_columns` configuration. The builder aligns data and targets by index and preserves missing targets; `select_training_data` drops rows missing the target columns selected for a particular training task. `ExistingLabels` shows how to use one or several label columns already present in the source data instead.

Add or remove feature objects in `business_logic/dataset.py`, or implement another object with `transform(data)`. For a different target definition, implement `build(data)` and return a Series for one target or a DataFrame for multiple targets. `ml_core` does not import business code, Kedro, or a storage implementation, so it can later be extracted as a Python package. For multiple targets, choose an estimator that supports multi-output or wrap the estimator with an appropriate adapter.

This is a starter layout and a small set of core components. Model choice, feature logic, evaluator, and environment-specific storage remain project decisions. Supply a new `dataset_id` to each Kedro run, for example `kedro run --params="dataset_id=run-001"`; the catalog uses it in the dataset path.
