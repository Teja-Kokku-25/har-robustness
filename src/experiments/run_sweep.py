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
    X_raw, y, _ = load_raw(C.DATA_DIR)
    X = extract_features(X_raw)

    # identical folds for every (kind, level, method, clf) configuration
    skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    splits = list(skf.split(X, y))

    rows = []
    for kind in C.KINDS:
        for level in C.LEVELS:
            X_corr = corrupt(X, X_raw, kind=kind, level=level, seed=C.SEED)
            for fold, (tr_idx, te_idx) in enumerate(splits):
                X_tr, X_te = X_corr[tr_idx], X_corr[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]
                # leak-free: impute and scale with train-only statistics
                X_tr, X_te = mean_impute(X_tr, X_te)
                scaler = StandardScaler().fit(X_tr)
                X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

                for method in C.METHODS:
                    if method == "none":
                        Z_tr, Z_te = X_tr_s, X_te_s
                    elif method == "pca":
                        red = PCA(C.PCA_VAR).fit(X_tr_s)
                        Z_tr, Z_te = red.transform(X_tr_s), red.transform(X_te_s)
                    else:
                        red = LDA(C.LDA_DIMS).fit(X_tr_s, y_tr)
                        Z_tr, Z_te = red.transform(X_tr_s), red.transform(X_te_s)
                    n_components = Z_tr.shape[1]
                    fisher = fisher_score(Z_te, y_te)

                    for clf_name in C.CLASSIFIERS:
                        clf = make_clf(clf_name).fit(Z_tr, y_tr)
                        y_pred = clf.predict(Z_te)
                        acc = accuracy_score(y_te, y_pred)
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_te, y_pred, average="macro", zero_division=0)
                        rows.append({
                            "kind": kind, "level": level, "method": method,
                            "clf": clf_name, "fold": fold,
                            "n_components": n_components,
                            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
                            "fisher": fisher,
                        })
            print(f"done kind={kind} level={level}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    run_sweep()
