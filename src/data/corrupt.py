"""Controlled corruption: gaussian noise, missing values, OOD samples."""
import numpy as np

def corrupt(X, X_raw, kind: str, level: float, seed: int = 42):
    """Return a corrupted copy of X (n, d).
    kind="gaussian": X + N(0, (level * feature_std)^2)
    kind="missing" : set `level` fraction of entries to NaN (impute later with train-fold mean)
    kind="ood"     : pick `level` fraction of samples, rotate (scipy Rotation.random) and
                     scale (gain 0.7-1.3) their raw xyz signals, then re-extract features.
    level=0 must return X unchanged.
    """
    raise NotImplementedError

def mean_impute(X_train, X_test):
    """Fill NaNs in both using column means computed on X_train only."""
    raise NotImplementedError
