"""From-scratch Principal Component Analysis (PCA) via eigendecomposition."""
import numpy as np


class PCA:
    """Unsupervised linear dimensionality reduction via covariance eigendecomposition.

    Steps:
      1. Center data: X_c = X - mean(X)
      2. Sample covariance matrix: Cov = (X_c^T @ X_c) / (n - 1)
      3. Symmetric eigendecomposition: np.linalg.eigh(Cov)
      4. Sort eigenvalues/eigenvectors in descending order
      5. Retain smallest k components whose cumulative explained variance >= `var`
         (or exact integer k if `var >= 1` is an int).
    """

    def __init__(self, var: float = 0.95):
        self.var = var
        self.mean_ = None
        self.components_ = None              # shape (d, k): columns are principal axes
        self.explained_variance_ = None      # shape (k,)
        self.explained_variance_ratio_ = None  # shape (k,)
        self.all_explained_variance_ratio_ = None  # shape (d,) for scree plots
        self.n_components_ = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # Sample covariance matrix (d x d)
        cov = (X_c.T @ X_c) / max(n - 1, 1)

        # eigh returns eigenvalues in ascending order for real symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        # Deterministic sign convention: largest-magnitude entry in each eigenvector is positive
        max_abs_idx = np.argmax(np.abs(eigvecs), axis=0)
        signs = np.sign(eigvecs[max_abs_idx, np.arange(d)])
        signs[signs == 0] = 1.0
        eigvecs = eigvecs * signs

        total_var = eigvals.sum()
        if total_var <= 0:
            ratios = np.zeros_like(eigvals)
        else:
            ratios = eigvals / total_var
        self.all_explained_variance_ratio_ = ratios

        if isinstance(self.var, (int, np.integer)) and self.var >= 1:
            k = min(int(self.var), d)
        else:
            cum = np.cumsum(ratios)
            k = int(np.searchsorted(cum, self.var, side="left")) + 1
            k = max(1, min(k, d))

        self.n_components_ = k
        self.components_ = eigvecs[:, :k]
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = ratios[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before calling transform().")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
