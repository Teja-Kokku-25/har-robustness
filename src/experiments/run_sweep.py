"""Full corruption sweep with stratified 5-fold CV. Writes results/results.csv.

Per fold (fit everything on train only, to avoid leakage):
  corrupt -> impute -> scale -> reduce (none/PCA/LDA) -> classify -> metrics + fisher
Columns: kind, level, method, clf, fold, acc, prec, rec, f1, fisher
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src import config as C
from src.data.load import load_raw
from src.data.features import extract_features
from src.data.corrupt import corrupt, mean_impute
from src.reduction.scaler import StandardScaler
from src.reduction.pca import PCA
from src.reduction.lda import LDA
from src.reduction.fisher import fisher_score
from src.models.knn import KNN
from src.models.gnb import GaussianNB


def make_reducer(method):
    return {"none": None, "pca": PCA(C.PCA_VAR), "lda": LDA(C.LDA_DIMS)}[method]


def make_clf(name):
    return {"knn": KNN(C.KNN_K), "gnb": GaussianNB()}[name]


def run_sweep(out_csv=os.path.join(C.RESULTS_DIR, "results.csv")):
    raise NotImplementedError


if __name__ == "__main__":
    run_sweep()
