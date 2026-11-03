"""Project-level regression and classification evaluators."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


class RegressionEvaluator:
    def evaluate(self, y_true, y_pred):
        actual = np.asarray(y_true)
        predicted = np.asarray(y_pred)
        if actual.ndim == 1:
            actual = actual.reshape(-1, 1)
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)

        columns = (
            list(y_true.columns)
            if hasattr(y_true, "columns")
            else [getattr(y_true, "name", None) or "target"]
        )
        metrics = {}
        for index, column in enumerate(columns):
            error = actual[:, index] - predicted[:, index]
            metrics[f"mae_{column}"] = float(np.abs(error).mean())
            metrics[f"rmse_{column}"] = float(np.sqrt(np.square(error).mean()))
        metrics["mae"] = float(mean_absolute_error(actual, predicted))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(actual, predicted)))
        return metrics


class ClassificationEvaluator:
    def __init__(self, *, average="macro", positive_label=None, auc_average="macro"):
        self.average = average
        self.positive_label = positive_label
        self.auc_average = auc_average

    def evaluate(self, y_true, y_pred, *, y_score=None, y_score_labels=None):
        observed_labels = np.unique(y_true)
        score_labels = (
            np.asarray(y_score_labels)
            if y_score_labels is not None
            else observed_labels
        )
        positive_label = self.positive_label
        if positive_label is None:
            positive_label = 1 if 1 in score_labels else score_labels[-1]

        average = "binary" if len(score_labels) == 2 else self.average
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(
                y_true, y_pred, labels=labels, average=average,
                pos_label=positive_label, zero_division=0,
            )),
            "recall": float(recall_score(
                y_true, y_pred, labels=labels, average=average,
                pos_label=positive_label, zero_division=0,
            )),
            "f1": float(f1_score(
                y_true, y_pred, labels=labels, average=average,
                pos_label=positive_label, zero_division=0,
            )),
            "confusion_matrix": {
                "labels": labels.tolist(),
                "values": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            },
        }

        if y_score is None:
            return metrics

        scores = np.asarray(y_score)
        if len(score_labels) == 2 and len(observed_labels) == 2:
            if scores.ndim == 2:
                positive_scores = scores[:, list(score_labels).index(positive_label)]
            else:
                positive_scores = scores.reshape(-1)
                if score_labels[-1] != positive_label:
                    positive_scores = (
                        1 - positive_scores
                        if np.all((positive_scores >= 0) & (positive_scores <= 1))
                        else -positive_scores
                    )
            binary_true = (np.asarray(y_true) == positive_label).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(binary_true, positive_scores))
            metrics["pr_auc"] = float(
                average_precision_score(binary_true, positive_scores)
            )
        elif (
            len(score_labels) > 2
            and scores.ndim == 2
            and set(observed_labels) == set(score_labels)
        ):
            binary_targets = label_binarize(y_true, classes=score_labels)
            class_auc = np.asarray([
                roc_auc_score(binary_targets[:, index], scores[:, index])
                for index in range(len(score_labels))
            ])
            if self.auc_average == "weighted":
                metrics["roc_auc"] = float(np.average(
                    class_auc, weights=binary_targets.sum(axis=0)
                ))
            elif self.auc_average == "micro":
                metrics["roc_auc"] = float(roc_auc_score(
                    binary_targets.ravel(), scores.ravel()
                ))
            else:
                metrics["roc_auc"] = float(class_auc.mean())
            metrics["pr_auc"] = float(average_precision_score(
                binary_targets, scores, average=self.auc_average
            ))
        return metrics


class BinaryClassificationEvaluator(ClassificationEvaluator):
    """Backward-compatible name for the classification evaluator."""


class MultiOutputClassificationEvaluator:
    """Evaluate each independent classification target in a DataFrame."""

    def __init__(self, *, average="macro", positive_label=None, auc_average="macro"):
        self.target_evaluator = ClassificationEvaluator(
            average=average,
            positive_label=positive_label,
            auc_average=auc_average,
        )

    def evaluate(self, y_true, y_pred, *, y_score=None, y_score_labels=None):
        if not hasattr(y_true, "columns") or not hasattr(y_pred, "columns"):
            raise TypeError("MultiOutputClassificationEvaluator expects DataFrame targets.")

        metrics = {}
        values_by_metric = {}
        for index, target_name in enumerate(y_true.columns):
            target_scores = y_score[index] if isinstance(y_score, list) else None
            target_labels = (
                y_score_labels[index]
                if isinstance(y_score_labels, list)
                else y_score_labels
            )
            target_metrics = self.target_evaluator.evaluate(
                y_true.iloc[:, index],
                y_pred.iloc[:, index],
                y_score=target_scores,
                y_score_labels=target_labels,
            )
            for name, value in target_metrics.items():
                if name == "confusion_matrix":
                    metrics[f"{target_name}_{name}"] = value
                else:
                    metrics[f"{target_name}_{name}"] = value
                    values_by_metric.setdefault(name, []).append(value)

        for name, values in values_by_metric.items():
            metrics[f"mean_{name}"] = float(np.mean(values))
        return metrics
