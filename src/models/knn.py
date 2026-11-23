import numpy as np

class KNN:
    """Euclidean k-NN; distances vectorized as |a|^2 + |b|^2 - 2ab (in batches); majority vote."""
    def __init__(self, k: int = 5, batch: int = 1000):
        self.k, self.batch = k, batch
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
