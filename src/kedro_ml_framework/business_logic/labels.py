"""Example existing-column and computed multi-target label builders."""

import pandas as pd


class ExistingLabels:
    def __init__(self, columns):
        self.columns = columns

    def build(self, data):
        labels = data.loc[:, self.columns]
        return labels[self.columns[0]] if len(self.columns) == 1 else labels


class ForwardReturnLabels:
    def __init__(self, *, price_column, time_column, horizons, entity_column=None, threshold=None):
        self.price_column = price_column
        self.time_column = time_column
        self.entity_column = entity_column
        self.horizons = horizons
        self.threshold = threshold

    def build(self, data):
        ordered = data.sort_values(self.time_column, ascending=True)
        labels = pd.DataFrame(index=ordered.index)

        for horizon in self.horizons:
            if self.entity_column:
                future_price = ordered.groupby(self.entity_column, sort=False)[self.price_column].shift(-horizon)
            else:
                future_price = ordered[self.price_column].shift(-horizon)

            returns = future_price / ordered[self.price_column] - 1
            if self.threshold is None:
                name = f"forward_return_{horizon}d"
                target = returns
            else:
                name = f"forward_up_{horizon}d"
                target = returns.gt(self.threshold).where(returns.notna()).astype("Int64")
            labels.loc[ordered.index, name] = target

        return labels.reindex(data.index)
