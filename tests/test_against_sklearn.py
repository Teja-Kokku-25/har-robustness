"""From-scratch implementations vs sklearn. Run: python -m pytest tests/"""
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    600, 20, n_informative=8, n_classes=6, n_clusters_per_class=1, random_state=0
)


def test_scaler():
    from sklearn.preprocessing import StandardScaler as SK
    from src.reduction.scaler import StandardScaler

    Z_ours = StandardScaler().fit_transform(X)
    Z_sk = SK().fit_transform(X)
    assert np.allclose(Z_ours, Z_sk, atol=1e-6)


def test_pca_variance():
    from sklearn.decomposition import PCA as SK
    from src.reduction.pca import PCA

    ours = PCA(0.95).fit(X)
    sk = SK(0.95).fit(X)
    assert np.allclose(
        ours.explained_variance_ratio_, sk.explained_variance_ratio_, atol=1e-6
    )
    Z_ours = ours.transform(X)
    Z_sk = sk.transform(X)
    assert np.allclose(np.abs(Z_ours), np.abs(Z_sk), atol=1e-5)


def test_lda_and_fisher():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLDA
    from sklearn.neighbors import KNeighborsClassifier
    from src.reduction.lda import LDA
    from src.reduction.pca import PCA
    from src.reduction.fisher import fisher_score

    Z_lda = LDA(5).fit_transform(X, y)
    Z_sk = SKLDA(n_components=5).fit_transform(X, y)
    Z_pca = PCA(5).fit_transform(X)

    # Supervised LDA should achieve higher Fisher's Criterion than unsupervised PCA
    assert fisher_score(Z_lda, y) > fisher_score(Z_pca, y)

    # Downstream 5-NN accuracy on our LDA should closely match sklearn's LDA
    acc_ours = KNeighborsClassifier(5).fit(Z_lda, y).score(Z_lda, y)
    acc_sk = KNeighborsClassifier(5).fit(Z_sk, y).score(Z_sk, y)
    assert abs(acc_ours - acc_sk) < 0.02


def test_gnb_accuracy():
    from sklearn.naive_bayes import GaussianNB as SK
    from src.models.gnb import GaussianNB

    assert (
        abs(
            (GaussianNB().fit(X, y).predict(X) == y).mean()
            - (SK().fit(X, y).predict(X) == y).mean()
        )
        < 0.01
    )


def test_knn_accuracy():
    from sklearn.neighbors import KNeighborsClassifier as SK
    from src.models.knn import KNN

    assert (
        abs(
            (KNN(5).fit(X, y).predict(X) == y).mean()
            - (SK(5).fit(X, y).predict(X) == y).mean()
        )
        < 0.01
    )
