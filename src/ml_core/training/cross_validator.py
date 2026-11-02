import numpy as np
import pandas as pd
from sklearn.utils import Bunch


class CrossValidator:
    """Evaluate a fresh estimator from a factory on each supplied CV split."""

    def __init__(self, estimator_factory, splitter, evaluator):
        self.estimator_factory = estimator_factory
        self.splitter = splitter
        self.evaluator = evaluator

    def evaluate(self, X, y, *, groups=None, sample_weight=None):
        target_names = list(y.columns) if isinstance(y, pd.DataFrame) else [y.name or "target"]
        oof_values = [None] * len(X)
        fold_ids = np.full(len(X), -1, dtype=int)
        fold_rows = []

        for fold, (train_positions, valid_positions) in enumerate(
            self.splitter.split(X, y, groups), start=1
        ):
            estimator = self.estimator_factory()
            X_train, X_valid = X.iloc[train_positions], X.iloc[valid_positions]
            y_train, y_valid = y.iloc[train_positions], y.iloc[valid_positions]

            fit_params = {}
            if sample_weight is not None:
                train_weight = (
                    sample_weight.iloc[train_positions]
                    if hasattr(sample_weight, "iloc")
                    else np.asarray(sample_weight)[train_positions]
                )
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
            fold_ids[valid_positions] = fold
            prediction_values = predictions.to_numpy()
            for position, prediction in zip(valid_positions, prediction_values):
                oof_values[position] = prediction
            fold_rows.append(self.evaluator.evaluate(y_valid, predictions))

        fold_metrics = pd.DataFrame(fold_rows, index=pd.Index(range(1, len(fold_rows) + 1), name="fold"))
        summary = pd.DataFrame({
            "mean": fold_metrics.mean(),
            "std": fold_metrics.std(),
        })

        oof_positions = np.flatnonzero(fold_ids >= 0)
        y_oof = y.iloc[oof_positions]
        oof_values = np.asarray([oof_values[position] for position in oof_positions])
        y_pred_oof = self._as_target_frame(
            oof_values, y_oof, target_names
        )
        oof_metrics = self.evaluator.evaluate(y_oof, y_pred_oof)
        oof_predictions = Bunch(
            index=X.index.take(oof_positions),
            y_true=y_oof,
            y_pred=y_pred_oof,
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
