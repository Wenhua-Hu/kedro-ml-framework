from inspect import Parameter, signature

import numpy as np


def prediction_scores(estimator, X):
    """Return estimator scores and the class order for score columns, if any."""
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
    else:
        return None, None

    final_estimator = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    classes = getattr(final_estimator, "classes_", None)
    if isinstance(scores, list):
        return scores, classes
    return np.asarray(scores), None if classes is None else np.asarray(classes)


def align_score_columns(scores, source_classes, target_classes):
    """Align class-score columns to a shared class order across CV folds."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or source_classes is None:
        return scores

    aligned = np.zeros((scores.shape[0], len(target_classes)), dtype=scores.dtype)
    target_positions = {label: position for position, label in enumerate(target_classes)}
    for source_position, label in enumerate(source_classes):
        if label in target_positions:
            aligned[:, target_positions[label]] = scores[:, source_position]
    return aligned


def evaluate_predictions(evaluator, y_true, y_pred, *, y_score=None, y_score_labels=None):
    """Call an injected evaluator using its minimal supported contract.

    Every evaluator must implement ``evaluate(y_true, y_pred)``. Score-aware
    evaluators may additionally accept keyword-only ``y_score`` and
    ``y_score_labels``; these optional arguments are passed only when declared
    by the evaluator. ``y_score_labels`` gives the class order for score
    columns when the estimator exposes it.
    """
    if y_score is None:
        return evaluator.evaluate(y_true, y_pred)

    parameters = signature(evaluator.evaluate).parameters
    accepts_kwargs = any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values())
    kwargs = {}
    if "y_score" in parameters or accepts_kwargs:
        kwargs["y_score"] = y_score
    if y_score_labels is not None and ("y_score_labels" in parameters or accepts_kwargs):
        kwargs["y_score_labels"] = y_score_labels
    return evaluator.evaluate(y_true, y_pred, **kwargs)
