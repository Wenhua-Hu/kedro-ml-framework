"""Project-specific preprocessing choices for model pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
