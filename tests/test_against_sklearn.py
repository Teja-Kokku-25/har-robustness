"""From-scratch implementations vs sklearn. Run: python -m pytest tests/"""
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(600, 20, n_informative=8, n_classes=6, n_clusters_per_class=1, random_state=0)

def test_pca_variance():
    from sklearn.decomposition import PCA as SK
    from src.reduction.pca import PCA
    assert np.allclose(PCA(0.95).fit(X).explained_variance_ratio_,
                       SK(0.95).fit(X).explained_variance_ratio_, atol=1e-6)

def test_gnb_accuracy():
    from sklearn.naive_bayes import GaussianNB as SK
    from src.models.gnb import GaussianNB
    assert abs((GaussianNB().fit(X, y).predict(X) == y).mean() - (SK().fit(X, y).predict(X) == y).mean()) < 0.01

def test_knn_accuracy():
    from sklearn.neighbors import KNeighborsClassifier as SK
    from src.models.knn import KNN
    assert abs((KNN(5).fit(X, y).predict(X) == y).mean() - (SK(5).fit(X, y).predict(X) == y).mean()) < 0.01
