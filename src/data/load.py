"""Load raw UCI HAR inertial windows."""
import os
import numpy as np

CHANNELS = [f"{s}_{a}" for s in ("body_acc", "body_gyro", "total_acc") for a in "xyz"]


def _load_split(data_dir, split):
    sig_dir = os.path.join(data_dir, split, "Inertial Signals")
    X = np.stack(
        [np.loadtxt(os.path.join(sig_dir, f"{ch}_{split}.txt")) for ch in CHANNELS],
        axis=-1,
    )  # (n, 128, 9)
    y = np.loadtxt(os.path.join(data_dir, split, f"y_{split}.txt"), dtype=int) - 1  # 1..6 -> 0..5
    subj = np.loadtxt(os.path.join(data_dir, split, f"subject_{split}.txt"), dtype=int)
    return X, y, subj


def load_raw(data_dir: str = "data/UCI HAR Dataset"):
    """Return X_raw (n, 128, 9), y (n,) in 0..5, subjects (n,).
    Concatenates train + test (10,299 windows)."""
    parts = [_load_split(data_dir, s) for s in ("train", "test")]
    X, y, subj = (np.concatenate(p) for p in zip(*parts))
    return X.astype(np.float32), y, subj
