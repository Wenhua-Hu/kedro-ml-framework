import pandas as pd


def select_training_data(X, y, *, target_columns=None, required_feature_columns=()):
    """Select targets and rows usable by this training task.

    Missing targets are dropped only for the selected target columns. Feature
    rows are dropped only when explicitly listed as required by the caller.
    """
    if target_columns is not None:
        target_columns = [target_columns] if isinstance(target_columns, str) else list(target_columns)
        if isinstance(y, pd.DataFrame):
            y = y.loc[:, target_columns]
        elif len(target_columns) != 1 or y.name != target_columns[0]:
            raise KeyError(target_columns)

    valid_rows = y.notna().all(axis=1) if isinstance(y, pd.DataFrame) else y.notna()
    if required_feature_columns:
        valid_rows &= X.loc[:, list(required_feature_columns)].notna().all(axis=1)

    return X.loc[valid_rows], y.loc[valid_rows]
