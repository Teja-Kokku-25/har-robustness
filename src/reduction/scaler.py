"""From-scratch StandardScaler (z-score normalization)."""
import numpy as np


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance:
        z = (x - mean) / std
    Fitted on training data only to prevent data leakage.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        # Guard against zero-variance (constant) columns
        self.std_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
