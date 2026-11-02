"""Example evaluator for the multi-horizon regression labels in this project."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class RegressionEvaluator:
    def evaluate(self, y_true, y_pred):
        actual = np.asarray(y_true)
        predicted = np.asarray(y_pred)
        if actual.ndim == 1:
            actual = actual.reshape(-1, 1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)

        columns = list(y_true.columns) if hasattr(y_true, "columns") else ["target"]
        metrics = {}
        for index, column in enumerate(columns):
            error = actual[:, index] - predicted[:, index]
            metrics[f"mae_{column}"] = float(np.abs(error).mean())
            metrics[f"rmse_{column}"] = float(np.sqrt(np.square(error).mean()))
        return metrics


class BinaryClassificationEvaluator:
    def evaluate(self, y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }
