"""Example project-specific feature transformers for ordered price data."""

from sklearn.base import BaseEstimator, TransformerMixin


class PriceReturnFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, *, price_column, time_column, entity_column=None, periods=(1, 5, 20)):
        self.price_column = price_column
        self.time_column = time_column
        self.entity_column = entity_column
        self.periods = periods

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        ordered = data.sort_values(self.time_column, ascending=True)
        result = data.copy()
        for period in self.periods:
            if self.entity_column:
                values = ordered.groupby(self.entity_column, sort=False)[self.price_column].pct_change(
                    period, fill_method=None
                )
            else:
                values = ordered[self.price_column].pct_change(period, fill_method=None)
            result.loc[ordered.index, f"return_{period}d"] = values
        return result


class RollingVolatilityFeature(BaseEstimator, TransformerMixin):
    def __init__(self, *, price_column, time_column, window, entity_column=None):
        self.price_column = price_column
        self.time_column = time_column
        self.entity_column = entity_column
        self.window = window

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        ordered = data.sort_values(self.time_column, ascending=True)
        if self.entity_column:
            volatility = ordered.groupby(self.entity_column, sort=False)[self.price_column].transform(
                lambda price: price.pct_change(fill_method=None).rolling(self.window).std()
            )
        else:
            volatility = ordered[self.price_column].pct_change(fill_method=None).rolling(self.window).std()

        result = data.copy()
        result.loc[ordered.index, f"volatility_{self.window}d"] = volatility
        return result
