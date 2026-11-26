# Benchmarking PCA and LDA Robustness to Sensor Noise for Human Activity Recognition

CSE 455/555 Pattern Recognition, Fall 2026 — Track 2 (Dimensionality Reduction vs. Noise & Out-of-Distribution Data).

**Team:** Yuva Mani Teja Kokku, Harsha Adinarayanaraju Lolabattu, Litheesh Velakaturi Rambabu

## Overview
We benchmark how **Principal Component Analysis (PCA)** (`95%` variance retained) and **Linear Discriminant Analysis (LDA)** (`5` discriminant axes) preserve class separability and downstream classification accuracy (`k-NN` and `Gaussian Naive Bayes`) on the **UCI HAR Dataset** (`10,299` windows, `6` activities, `63` custom time- and frequency-domain features extracted from the `9` raw 50 Hz inertial channels) as sensor corruption sweeps from `0%` to `50%` across three modes:
1. **Additive Gaussian sensor noise** (`gaussian`)
2. **Random missing-value feature dropout + train-mean imputation** (`missing`)
3. **Out-of-distribution 3D sensor rotation + gain shift** (`ood`)

All core components—feature extraction, corruption injection, mean imputation, `StandardScaler`, `PCA`, `LDA`, `fisher_score`, `KNN`, and `GaussianNB`—are implemented from scratch in NumPy and verified against `scikit-learn`.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # downloads UCI HAR into data/UCI HAR Dataset/
```

## Run
```bash
python -m pytest tests/              # verifies from-scratch implementations vs. scikit-learn
python -m src.experiments.run_sweep  # runs 5-fold stratified CV sweep -> results/results.csv (540 rows)
python -m src.experiments.plots      # generates 300-DPI figures -> results/figures/*.png
```
Or open `notebooks/01_data_and_features.ipynb`, `notebooks/02_reduction_checks.ipynb`, and `notebooks/03_results.ipynb` and run all cells.

## Layout
| Folder | Contents | Owner |
|---|---|---|
| `src/data/` | Raw loader (`load.py`), 63-feature extractor (`features.py`), controlled corruption & train-mean imputation (`corrupt.py`) | Harsha Lolabattu (`features.py`, `corrupt.py`), Yuva Mani Teja Kokku (`load.py`) |
| `src/reduction/` | From-scratch `StandardScaler`, `PCA`, `LDA`, and Fisher's Criterion (`fisher_score`) | Yuva Mani Teja Kokku |
| `src/models/` | From-scratch vectorized `KNN` (`k=5`) and `GaussianNB` | Litheesh Velakaturi Rambabu |
| `src/experiments/` | Leak-free 5-fold stratified CV sweep (`run_sweep.py`) and 300-DPI figure suite (`plots.py`) | Litheesh Velakaturi Rambabu |
| `notebooks/` | `01_data_and_features.ipynb`, `02_reduction_checks.ipynb`, `03_results.ipynb` | Harsha (`01`), Teja (`02`), Litheesh (`03`) |
| `tests/` | Unit tests checking from-scratch `StandardScaler`, `PCA`, `LDA`, `KNN`, and `GaussianNB` against `scikit-learn` | All |

## Contributions
- **Harsha Adinarayanaraju Lolabattu (Part 1 — Data, Feature Extraction & Corruption):** Implemented vectorized extraction of 63 time- and frequency-domain features (mean, std, min, max, energy, non-DC PSD spectral entropy, dominant FFT frequency) from the 9 raw inertial channels in `src/data/features.py`; built the 3-mode corruption pipeline (`gaussian`, `missing`, `ood` 3D rotation + gain) and leak-free `mean_impute` in `src/data/corrupt.py`; authored `notebooks/01_data_and_features.ipynb`.
- **Yuva Mani Teja Kokku (Part 2 — Dimensionality Reduction & Fisher's Criterion):** Implemented the raw UCI HAR loader (`src/data/load.py`) and from-scratch `StandardScaler` (`src/reduction/scaler.py`), covariance-eigendecomposition `PCA` (`src/reduction/pca.py`), ridge-regularized multi-class Fisher `LDA` (`src/reduction/lda.py`), and scatter-matrix Fisher's Criterion (`src/reduction/fisher.py`); authored unit tests and `notebooks/02_reduction_checks.ipynb`.
- **Litheesh Velakaturi Rambabu (Part 3 — Classifiers, 5-Fold CV Sweep & Visualization):** Implemented from-scratch batched `KNN` (`src/models/knn.py`) and `GaussianNB` (`src/models/gnb.py`), the leak-free 5-fold stratified cross-validation sweep across 540 configurations (`src/experiments/run_sweep.py`), the 5 publication figure generators (`src/experiments/plots.py`), and the interactive results/demo notebook (`notebooks/03_results.ipynb`).
