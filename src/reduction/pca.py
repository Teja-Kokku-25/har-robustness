import numpy as np

class PCA:
    """Center -> covariance -> np.linalg.eigh -> sort descending -> keep components
    reaching `var` cumulative explained variance."""
    def __init__(self, var: float = 0.95):
        self.var = var
    def fit(self, X, y=None):
        # set self.mean_, self.components_ (d, k), self.explained_variance_ratio_ (k,)
        raise NotImplementedError
    def transform(self, X):
        raise NotImplementedError
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
