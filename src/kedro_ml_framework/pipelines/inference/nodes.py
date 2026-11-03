import numpy as np
import pandas as pd

from ml_core.inference import Predictor


def predict(model_bundle, inference_data, prediction_method="predict"):
    """Run the selected prediction method and return a catalog-friendly frame.

    Probability prediction requires a classifier with ``predict_proba``. For
    multi-output classifiers, columns are named ``target__probability_class``.
    """
    predictor = Predictor(model_bundle)
    if prediction_method == "predict":
        return predictor.predict(inference_data)
    if prediction_method != "predict_proba":
        raise ValueError("prediction_method must be 'predict' or 'predict_proba'.")

    probabilities = predictor.predict_proba(inference_data)
    model = model_bundle.model_pipeline
    if hasattr(model, "steps") and model.steps:
        model = model.steps[-1][1]
    classes = model.classes_

    if isinstance(probabilities, list):
        target_names = model_bundle.target_names or [
            f"target_{index}" for index in range(len(probabilities))
        ]
        values = np.column_stack(probabilities)
        columns = [
            f"{target_name}__probability_{class_label}"
            for target_name, target_classes in zip(target_names, classes)
            for class_label in target_classes
        ]
    else:
        values = probabilities
        columns = [f"probability_{class_label}" for class_label in classes]

    return pd.DataFrame(values, index=inference_data.index, columns=columns)
