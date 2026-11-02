import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


class DateGroupedTimeSeriesSplit:
    """Split panel rows by unique dates, keeping every entity/date together."""

    def __init__(self, n_splits=5, *, test_size=None, gap=0, max_train_size=None):
        self.time_splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
        )

    def split(self, X, y=None, groups=None):
        dates = pd.Index(groups).drop_duplicates().sort_values()
        for train_dates, test_dates in self.time_splitter.split(dates):
            train_groups = dates.take(train_dates)
            test_groups = dates.take(test_dates)
            train_positions = np.flatnonzero(pd.Index(groups).isin(train_groups))
            test_positions = np.flatnonzero(pd.Index(groups).isin(test_groups))
            yield train_positions, test_positions
