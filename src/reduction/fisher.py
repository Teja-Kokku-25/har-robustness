"""Fisher's Criterion and scatter matrix utilities."""
import numpy as np


def scatter_matrices(Z: np.ndarray, y: np.ndarray):
    """Compute Within-Class Scatter (S_W) and Between-Class Scatter (S_B).

    Given data Z of shape (n, d) and integer class labels y of shape (n,):
      S_W = sum_{c} sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T
      S_B = sum_{c} N_c (mu_c - mu)(mu_c - mu)^T
    Returns (S_W, S_B) each of shape (d, d).
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y)
    if Z.ndim == 1:
        Z = Z[:, None]
    _, d = Z.shape
    classes = np.unique(y)
    overall_mean = Z.mean(axis=0)

    S_W = np.zeros((d, d), dtype=np.float64)
    S_B = np.zeros((d, d), dtype=np.float64)

    for c in classes:
        Z_c = Z[y == c]
        n_c = Z_c.shape[0]
        if n_c == 0:
            continue
        mu_c = Z_c.mean(axis=0)
        diff_w = Z_c - mu_c
        S_W += diff_w.T @ diff_w
        diff_b = (mu_c - overall_mean).reshape(d, 1)
        S_B += n_c * (diff_b @ diff_b.T)

    return S_W, S_B


def fisher_score(Z: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Compute multivariate Fisher's Criterion J = trace(S_B) / trace(S_W) on embedding Z.

    Measures the ratio of between-class spread to within-class spread.
    Higher values indicate better class separability.
    """
    S_W, S_B = scatter_matrices(Z, y)
    tr_w = float(np.trace(S_W))
    tr_b = float(np.trace(S_B))
    return tr_b / max(tr_w, eps)
