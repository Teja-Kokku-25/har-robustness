import numpy as np


class KNN:
    """Euclidean k-NN; distances vectorized as |a|^2 + |b|^2 - 2ab (in batches); majority vote."""

    def __init__(self, k: int = 5, batch: int = 1000):
        self.k, self.batch = k, batch

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=int)
        self.train_sq_ = np.sum(self.X_train_ ** 2, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        k = min(self.k, len(self.X_train_))
        n_cls = int(self.y_train_.max()) + 1
        out = np.empty(len(X), dtype=int)
        for s in range(0, len(X), self.batch):
            Xb = X[s:s + self.batch]
            d2 = (np.sum(Xb ** 2, axis=1, keepdims=True)
                  + self.train_sq_[None, :]
                  - 2.0 * (Xb @ self.X_train_.T))
            if k < d2.shape[1]:
                idx = np.argpartition(d2, k, axis=1)[:, :k]
            else:
                idx = np.tile(np.arange(d2.shape[1]), (len(Xb), 1))
            # sort the k neighbours by distance -> deterministic tie-breaking
            order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1, kind="stable")
            idx = np.take_along_axis(idx, order, axis=1)
            labels = self.y_train_[idx]                      # (b, k), nearest first
            counts = np.zeros((len(Xb), n_cls), dtype=int)
            np.add.at(counts, (np.arange(len(Xb))[:, None], labels), 1)
            # majority vote; ties go to the lowest class id (same convention as sklearn)
            out[s:s + self.batch] = np.argmax(counts, axis=1)
        return out
