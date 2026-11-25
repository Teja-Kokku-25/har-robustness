"""From-scratch Multi-Class Fisher Linear Discriminant Analysis (LDA)."""
import numpy as np
from src.reduction.fisher import scatter_matrices


class LDA:
    """Supervised dimensionality reduction maximizing Fisher's Criterion.

    Steps:
      1. Compute within-class scatter S_W and between-class scatter S_B.
      2. Regularize S_W -> S_W + eps * I to handle collinear/noisy features.
      3. Solve the generalized eigenvalue problem S_B w = lambda S_W w
         via pinv(S_W + eps * I) @ S_B.
      4. Sort eigenvectors by decreasing real eigenvalue and keep top
         min(n_components, C - 1, d) discriminant directions.
    """

    def __init__(self, n_components: int = 5, eps: float = 1e-4):
        self.n_components = n_components
        self.eps = eps
        self.mean_ = None
        self.scalings_ = None               # shape (d, k): projection matrix W
        self.explained_variance_ratio_ = None  # relative discriminant power of top k axes
        self.eigenvalues_ = None
        self.n_components_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        _, d = X.shape
        classes = np.unique(y)
        max_k = min(self.n_components, len(classes) - 1, d)

        self.mean_ = X.mean(axis=0)
        S_W, S_B = scatter_matrices(X, y)

        # Ridge regularization ensures invertibility under collinear features or dropout
        S_W_reg = S_W + self.eps * np.eye(d, dtype=np.float64)
        M = np.linalg.pinv(S_W_reg) @ S_B

        eigvals, eigvecs = np.linalg.eig(M)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]

        # Normalize columns to unit length and apply deterministic sign
        norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        eigvecs = eigvecs / norms
        max_abs_idx = np.argmax(np.abs(eigvecs), axis=0)
        signs = np.sign(eigvecs[max_abs_idx, np.arange(d)])
        signs[signs == 0] = 1.0
        eigvecs = eigvecs * signs

        self.n_components_ = max_k
        self.scalings_ = eigvecs[:, :max_k]
        self.eigenvalues_ = eigvals[:max_k]
        tot = eigvals[: len(classes) - 1].sum()
        self.explained_variance_ratio_ = (
            eigvals[:max_k] / tot if tot > 0 else np.zeros(max_k)
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scalings_ is None:
            raise RuntimeError("LDA must be fitted before calling transform().")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.scalings_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
