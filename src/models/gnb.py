import numpy as np

class GaussianNB:
    """Per-class mean/var (+ var_smoothing); predict argmax of log prior + sum of log N(x | mu, var)."""
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
