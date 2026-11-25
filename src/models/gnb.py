import numpy as np


class GaussianNB:
    """Per-class mean/var (+ var_smoothing); predict argmax of log prior + sum of log N(x | mu, var)."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        eps = self.var_smoothing * np.var(X, axis=0).max()
        self.log_prior_, self.mu_, self.var_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.log_prior_.append(np.log(len(Xc) / len(X)))
            self.mu_.append(Xc.mean(axis=0))
            self.var_.append(np.var(Xc, axis=0) + eps)
        self.log_prior_ = np.array(self.log_prior_)
        self.mu_ = np.array(self.mu_)
        self.var_ = np.array(self.var_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        log_probs = np.empty((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            ll = np.log(2 * np.pi * self.var_[i]) + (X - self.mu_[i]) ** 2 / self.var_[i]
            log_probs[:, i] = self.log_prior_[i] - 0.5 * ll.sum(axis=1)
        return self.classes_[np.argmax(log_probs, axis=1)]
