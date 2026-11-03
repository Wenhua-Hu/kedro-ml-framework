import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ml_core.training.scoring import (
    align_score_columns,
    evaluate_predictions,
    prediction_scores,
)


class CrossValidator:
    """Evaluate a fresh estimator from a factory on each supplied CV split.

    ``X`` must be a pandas DataFrame and ``y`` a pandas Series or DataFrame,
    since the validator slices folds by row position while preserving labels
    and indexes. ``groups`` and sample weights must follow the same row order.
    """

    def __init__(self, estimator_factory, splitter, evaluator):
        self.estimator_factory = estimator_factory
        self.splitter = splitter
        self.evaluator = evaluator

    def evaluate(
        self,
        X,
        y,
        *,
        groups=None,
        sample_weight=None,
        sample_weight_factory=None,
    ):
        """Fit and evaluate one estimator per split of the supplied data.

        ``X`` must be a pandas DataFrame; ``y`` must be a pandas Series or
        DataFrame with the same row order. ``groups`` and array-like
        ``sample_weight`` must be aligned to those rows. A
        ``sample_weight_factory`` receives each fold's training targets.
        """
        if sample_weight is not None and sample_weight_factory is not None:
            raise ValueError("Pass sample_weight or sample_weight_factory, not both.")

        target_names = list(y.columns) if isinstance(y, pd.DataFrame) else [y.name or "target"]
        class_labels = None
        oof_values = [None] * len(X)
        oof_scores = [None] * len(X)
        fold_ids = np.full(len(X), -1, dtype=int)
        fold_rows = []

        for fold, (train_positions, valid_positions) in enumerate(
            self.splitter.split(X, y, groups), start=1
        ):
            estimator = self.estimator_factory()
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]

            fit_params = {}
            if sample_weight_factory is not None:
                train_weight = sample_weight_factory(y_train)
            elif sample_weight is not None:
                train_weight = (
                    sample_weight.iloc[train_positions]
                    if hasattr(sample_weight, "iloc")
                    else np.asarray(sample_weight)[train_positions]
                )
            else:
                train_weight = None

            if train_weight is not None:
                weight_parameter = (
                    "model__sample_weight"
                    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps
                    else "sample_weight"
                )
                fit_params[weight_parameter] = train_weight
            estimator.fit(X_train, y_train, **fit_params)
            predictions = self._as_target_frame(
                estimator.predict(X_valid), y_valid, target_names
            )
            scores, score_labels = prediction_scores(estimator, X_valid)
            if scores is not None and score_labels is not None:
                if isinstance(scores, list) and isinstance(y, pd.DataFrame):
                    class_labels = [np.unique(y.iloc[:, index]) for index in range(y.shape[1])]
                    scores = [
                        align_score_columns(target_scores, source_classes, target_classes)
                        for target_scores, source_classes, target_classes in zip(
                            scores, score_labels, class_labels
                        )
                    ]
                    score_labels = class_labels
                elif not isinstance(y, pd.DataFrame):
                    class_labels = np.unique(y)
                    scores = align_score_columns(scores, score_labels, class_labels)
                    score_labels = class_labels
            fold_ids[valid_positions] = fold
            prediction_values = predictions.to_numpy()
            for position, prediction in zip(valid_positions, prediction_values):
                oof_values[position] = prediction
            if scores is not None:
                if isinstance(scores, list):
                    for target_index, target_scores in enumerate(scores):
                        for position, score in zip(valid_positions, target_scores):
                            if oof_scores[position] is None:
                                oof_scores[position] = [None] * len(scores)
                            oof_scores[position][target_index] = score
                else:
                    for position, score in zip(valid_positions, np.asarray(scores)):
                        oof_scores[position] = score
            fold_metrics_row = evaluate_predictions(
                self.evaluator,
                y_valid,
                predictions,
                y_score=scores,
                y_score_labels=score_labels,
            )
            fold_rows.append(fold_metrics_row)

        fold_metrics = pd.DataFrame(fold_rows, index=pd.Index(range(1, len(fold_rows) + 1), name="fold"))
        numeric_fold_metrics = fold_metrics.select_dtypes(include="number")
        summary = pd.DataFrame({
            "mean": numeric_fold_metrics.mean(),
            "std": numeric_fold_metrics.std(),
        })

        oof_positions = np.flatnonzero(fold_ids >= 0)
        y_oof = y.iloc[oof_positions]
        oof_values = np.asarray([oof_values[position] for position in oof_positions])
        y_pred_oof = self._as_target_frame(
            oof_values, y_oof, target_names
        )
        if oof_scores[oof_positions[0]] is not None:
            if isinstance(oof_scores[oof_positions[0]], list):
                scores_oof = [
                    np.asarray([oof_scores[position][target_index] for position in oof_positions])
                    for target_index in range(len(oof_scores[oof_positions[0]]))
                ]
            else:
                scores_oof = np.asarray([oof_scores[position] for position in oof_positions])
            oof_metrics = evaluate_predictions(
                self.evaluator,
                y_oof,
                y_pred_oof,
                y_score=scores_oof,
                y_score_labels=class_labels,
            )
        else:
            scores_oof = None
            oof_metrics = evaluate_predictions(self.evaluator, y_oof, y_pred_oof)
        oof_predictions = Bunch(
            index=X.index.take(oof_positions),
            y_true=y_oof,
            y_pred=y_pred_oof,
            y_score=scores_oof,
            y_score_labels=class_labels,
            fold=fold_ids[oof_positions],
        )
        return Bunch(
            fold_metrics=fold_metrics,
            summary=summary,
            oof_predictions=oof_predictions,
            oof_metrics=oof_metrics,
        )

    @staticmethod
    def _as_target_frame(values, target, target_names):
        if isinstance(target, pd.Series):
            if isinstance(values, pd.DataFrame):
                values = values.iloc[:, 0]
            if isinstance(values, pd.Series):
                return pd.Series(values.to_numpy(), index=target.index, name=target.name)
            return pd.Series(np.asarray(values).reshape(-1), index=target.index, name=target.name)

        if isinstance(values, pd.Series):
            values = values.to_frame()
        if isinstance(values, pd.DataFrame):
            return pd.DataFrame(values.to_numpy(), index=target.index, columns=target_names)

        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return pd.DataFrame(values, index=target.index, columns=target_names)
