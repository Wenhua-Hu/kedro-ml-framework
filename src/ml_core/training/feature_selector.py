class FeatureSelector:
    def __init__(self, columns):
        self.columns = columns

    def select(self, X):
        return X.loc[:, self.columns].copy()
