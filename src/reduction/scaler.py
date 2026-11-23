import numpy as np

class StandardScaler:
    """z = (x - mean) / std, fitted on train only."""
    def fit(self, X):
        raise NotImplementedError
    def transform(self, X):
        raise NotImplementedError
    def fit_transform(self, X):
        return self.fit(X).transform(X)
