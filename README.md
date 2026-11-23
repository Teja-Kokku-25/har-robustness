# Benchmarking PCA and LDA Robustness to Sensor Noise for Human Activity Recognition

CSE 455/555 Pattern Recognition, Fall 2026, Track 2.
Team: Yuva Mani Teja Kokku, Harsha Adinarayanaraju Lolabattu, Litheesh Velakaturi Rambabu

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # downloads UCI HAR into data/
```

## Run
```bash
python -m src.experiments.run_sweep  # writes results/results.csv
python -m src.experiments.plots      # writes results/figures/*.png
python -m pytest tests/              # from-scratch vs sklearn checks
```
Or open `notebooks/03_results.ipynb` and Run All.

## Layout
| Folder | Contents | Owner |
|---|---|---|
| `src/data/` | loading, feature extraction, corruption | TBD |
| `src/reduction/` | scaler, PCA, LDA, Fisher's criterion (from scratch) | TBD |
| `src/models/` | k-NN, Gaussian Naive Bayes (from scratch) | TBD |
| `src/experiments/` | CV sweep + plots | TBD |

## Contributions
- **Name:** ...
- **Name:** ...
- **Name:** ...
