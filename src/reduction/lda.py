import numpy as np

class LDA:
    """Fisher LDA: S_W, S_B -> eigenvectors of pinv(S_W + eps*I) @ S_B -> top n_components."""
    def __init__(self, n_components: int = 5, eps: float = 1e-6):
        self.n_components, self.eps = n_components, eps
    def fit(self, X, y):
        raise NotImplementedError
    def transform(self, X):
        raise NotImplementedError
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
