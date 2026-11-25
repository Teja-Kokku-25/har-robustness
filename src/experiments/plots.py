"""Figures from results/results.csv -> results/figures/ (300-DPI PNGs).
- metric vs corruption level (one panel per kind, method x classifier)
- fisher score vs level
- confusion matrices at 0% and 50% Gaussian noise (seaborn heatmap)
- 2D scatter of first two components at 0/25/50% Gaussian noise
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src.data.load import load_raw
from src.data.features import extract_features
from src.data.corrupt import corrupt, mean_impute
from src.reduction.scaler import StandardScaler
from src.reduction.pca import PCA
from src.reduction.lda import LDA
from src.models.knn import KNN

FIG_DIR = os.path.join(C.RESULTS_DIR, "figures")
CSV = os.path.join(C.RESULTS_DIR, "results.csv")
DPI = 300

KIND_TITLES = {"gaussian": "Gaussian Noise", "missing": "Missing Data Dropout",
               "ood": "OOD Rotation+Gain"}
METHOD_NAMES = {"pca": "PCA", "lda": "LDA", "none": "None"}
FISHER_NAMES = {"pca": "PCA (95% var)", "lda": "LDA (5 dims)", "none": "None (63 dims)"}
CLF_NAMES = {"knn": "k-NN", "gnb": "GaussianNB"}
METHOD_COLORS = {"pca": "#1f77b4", "lda": "#d62728", "none": "#555555"}
CLF_STYLE = {"knn": "-", "gnb": "--"}
CLF_MARKER = {"knn": "o", "gnb": "s"}


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def load_results(csv=CSV):
    return pd.read_csv(csv)


def _band(ax, d, col, **kw):
    g = d.groupby("level")[col].agg(["mean", "std"]).fillna(0.0)
    ax.plot(g.index, g["mean"], **kw)
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                    color=kw.get("color"), alpha=0.12, linewidth=0)


def plot_metric_vs_corruption(df, col="acc", ylabel="Accuracy", fname="accuracy_vs_corruption.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, kind in zip(axes, C.KINDS):
        for method in ["pca", "lda", "none"]:
            for clf in C.CLASSIFIERS:
                d = df[(df.kind == kind) & (df.method == method) & (df.clf == clf)]
                _band(ax, d, col, color=METHOD_COLORS[method], ls=CLF_STYLE[clf],
                      marker=CLF_MARKER[clf], ms=4, lw=1.8,
                      label=f"{METHOD_NAMES[method]} + {CLF_NAMES[clf]}")
        ax.set_title(KIND_TITLES[kind])
        ax.set_xlabel("Corruption level")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"{ylabel} (mean ± 1 std, 5 folds)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{ylabel} vs. corruption level", y=1.02)
    fig.tight_layout()
    path = _save(fig, fname)
    return fig, path


def plot_accuracy(df):
    return plot_metric_vs_corruption(df, "acc", "Accuracy", "accuracy_vs_corruption.png")


def plot_f1(df):
    return plot_metric_vs_corruption(df, "f1", "Macro F1", "f1_vs_corruption.png")


def plot_fisher(df):
    # Fisher score does not depend on the classifier: use one classifier's rows
    d0 = df[df.clf == C.CLASSIFIERS[0]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, kind in zip(axes, C.KINDS):
        for method in ["pca", "lda", "none"]:
            d = d0[(d0.kind == kind) & (d0.method == method)]
            _band(ax, d, "fisher", color=METHOD_COLORS[method], marker="o", ms=4, lw=2,
                  label=FISHER_NAMES[method])
        ax.set_yscale("log")
        ax.set_title(KIND_TITLES[kind])
        ax.set_xlabel("Corruption level")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Fisher criterion tr(S_B)/tr(S_W)  (log, held-out fold)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Class separability vs. corruption level", y=1.02)
    fig.tight_layout()
    return fig, _save(fig, "fisher_vs_corruption.png")


# ---- figures that need a re-fit (not stored in results.csv) -----------------

def _prepare(kind, level, fold=0):
    """One stratified train/test split of the corrupted, imputed, scaled features."""
    X_raw, y, _ = load_raw(C.DATA_DIR)
    X = extract_features(X_raw)
    skf = StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)
    tr, te = list(skf.split(X, y))[fold]
    Xc = corrupt(X, X_raw, kind=kind, level=level, seed=C.SEED)
    Xtr, Xte = mean_impute(Xc[tr], Xc[te])
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), y[tr], y[te]


def plot_confusions(kind="gaussian", levels=(0.0, 0.5)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for j, level in enumerate(levels):
        Xtr, Xte, ytr, yte = _prepare(kind, level)
        for i, (name, red) in enumerate([("PCA", PCA(C.PCA_VAR).fit(Xtr)),
                                         ("LDA", LDA(C.LDA_DIMS).fit(Xtr, ytr))]):
            Ztr, Zte = red.transform(Xtr), red.transform(Xte)
            pred = KNN(C.KNN_K).fit(Ztr, ytr).predict(Zte)
            cm = confusion_matrix(yte, pred, labels=range(6))
            cmn = 100.0 * cm / cm.sum(axis=1, keepdims=True)
            acc = (pred == yte).mean()
            ax = axes[i, j]
            sns.heatmap(cmn, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                        xticklabels=C.ACTIVITIES, yticklabels=C.ACTIVITIES,
                        cbar=False, ax=ax, linewidths=0.5, linecolor="white")
            ax.set_title(f"k-NN on {name} - {int(level * 100)}% {KIND_TITLES[kind]} "
                         f"(acc {acc:.1%})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.tick_params(axis="x", rotation=45)
            # highlight the SITTING / STANDING block
            ax.add_patch(plt.Rectangle((3, 3), 2, 2, fill=False, ec="#d62728", lw=2.5))
    fig.suptitle("Row-normalised confusion matrices (% of true class); "
                 "red box = SITTING / STANDING", y=1.0)
    fig.tight_layout()
    return fig, _save(fig, "confusion_matrices_0_vs_50.png")


def plot_embeddings(kind="gaussian", levels=(0.0, 0.25, 0.5)):
    fig, axes = plt.subplots(2, len(levels), figsize=(5.2 * len(levels), 9.5))
    palette = sns.color_palette("tab10", 6)
    rng = np.random.RandomState(C.SEED)
    for j, level in enumerate(levels):
        Xtr, Xte, ytr, yte = _prepare(kind, level)
        for i, (name, red) in enumerate([("PCA(2)", PCA(2).fit(Xtr)),
                                         ("LDA(2)", LDA(2).fit(Xtr, ytr))]):
            Z = red.transform(Xte)
            ax = axes[i, j]
            order = rng.permutation(len(Z))
            for c in range(6):
                m = yte[order] == c
                ax.scatter(Z[order][m, 0], Z[order][m, 1], s=7, alpha=0.6,
                           color=palette[c], label=C.ACTIVITIES[c])
            ax.set_title(f"{name} - {int(level * 100)}% noise")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(alpha=0.25)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, markerscale=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("2-D embeddings of held-out fold under Gaussian noise", y=1.0)
    fig.tight_layout()
    return fig, _save(fig, "embeddings_2d_0_25_50.png")


def main():
    df = load_results()
    paths = [plot_accuracy(df)[1], plot_f1(df)[1], plot_fisher(df)[1],
             plot_confusions()[1], plot_embeddings()[1]]
    plt.close("all")
    for p in paths:
        print("saved", p)


if __name__ == "__main__":
    main()
