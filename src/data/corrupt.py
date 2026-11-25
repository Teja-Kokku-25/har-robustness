"""Controlled corruption: gaussian noise, missing values, OOD samples."""
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from src.data.features import extract_features

KINDS = ("gaussian", "missing", "ood")
_TRIPLETS = (slice(0, 3), slice(3, 6), slice(6, 9))  # body_acc, body_gyro, total_acc


def corrupt(X, X_raw, kind: str, level: float, seed: int = 42):
    """Return a corrupted copy of X (n, d); X itself is never modified.

    kind="gaussian": X + N(0, (level * feature_std)^2), std taken per column of X.
    kind="missing" : set `level` fraction of all n*d entries to NaN (impute later with
                     ``mean_impute`` on the train fold). Every column keeps >= 1 observed value.
    kind="ood"     : pick round(level * n) samples, apply one random 3-D rotation per window
                     (shared by the body_acc / body_gyro / total_acc triplets, as for a
                     re-oriented phone) and a random gain in [0.7, 1.3] to their raw signals,
                     then re-extract their features.
    level <= 0 returns X.copy().
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if level <= 0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_out = np.array(X, dtype=np.float64, copy=True)
    n, d = X_out.shape

    if kind == "gaussian":
        scale = level * X_out.std(axis=0, keepdims=True)
        return X_out + rng.normal(0.0, scale, size=X_out.shape)

    if kind == "missing":
        k = min(int(round(level * n * d)), n * d)
        mask = np.zeros(n * d, dtype=bool)
        mask[rng.choice(n * d, size=k, replace=False)] = True
        mask = mask.reshape(n, d)
        # Guarantee no column is entirely NaN: un-mask one random row in any such column.
        full = np.flatnonzero(mask.all(axis=0))
        if full.size:
            mask[rng.integers(0, n, size=full.size), full] = False
        X_out[mask] = np.nan
        return X_out

    # kind == "ood"
    if X_raw is None or len(X_raw) != n:
        raise ValueError("ood corruption needs X_raw aligned with X (same n)")
    m = int(round(level * n))
    if m == 0:
        return X_out
    idx = rng.choice(n, size=m, replace=False)
    W = np.array(X_raw[idx], dtype=np.float64)  # (m, 128, 9)
    R = Rotation.random(m, random_state=rng).as_matrix()  # (m, 3, 3)
    for sl in _TRIPLETS:
        W[:, :, sl] = np.einsum("mij,mtj->mti", R, W[:, :, sl])
    W *= rng.uniform(0.7, 1.3, size=(m, 1, 1))
    F = extract_features(W)
    if F.shape[1] != d:
        raise ValueError(f"X has {d} columns but extract_features gives {F.shape[1]}")
    X_out[idx] = F
    return X_out


def mean_impute(X_train, X_test):
    """Fill NaNs in both using column means computed on X_train only (0.0 if all-NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN column
        mu = np.nanmean(X_train, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)

    def _fill(A):
        A = np.array(A, dtype=np.float64, copy=True)
        r, c = np.nonzero(np.isnan(A))
        A[r, c] = mu[c]
        return A

    return _fill(X_train), _fill(X_test)
