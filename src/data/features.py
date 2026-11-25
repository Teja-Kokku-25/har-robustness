"""Hand-crafted time/frequency features from raw windows.

Each raw window is (128 time steps x 9 channels) sampled at FS = 50 Hz. For every
channel we compute 7 features, giving 9 x 7 = 63 columns. Columns are ordered
channel-major: all 7 features of ``body_acc_x``, then all 7 of ``body_acc_y``, ...
(see ``feature_names()``), i.e. column ``c * 7 + f`` is feature ``f`` of channel ``c``.
"""
import numpy as np

from src.data.load import CHANNELS

FS = 50  # Hz
FEATURES = ["mean", "std", "min", "max", "energy", "entropy", "dom_freq"]
_EPS = 1e-12


def extract_features(X_raw: np.ndarray) -> np.ndarray:
    """X_raw (n, 128, 9) -> X (n, 63) float64.

    Per channel: mean, std, min, max, energy (mean squared amplitude),
    spectral entropy (Shannon entropy, in nats, of the normalised non-DC power
    spectrum) and dominant non-DC FFT frequency in Hz. Fully vectorised over windows.
    """
    X_raw = np.asarray(X_raw, dtype=np.float64)
    if X_raw.ndim != 3 or X_raw.shape[2] != len(CHANNELS):
        raise ValueError(f"expected (n, T, {len(CHANNELS)}), got {X_raw.shape}")
    n, T, C = X_raw.shape

    mean = X_raw.mean(axis=1)
    std = X_raw.std(axis=1)
    mn = X_raw.min(axis=1)
    mx = X_raw.max(axis=1)
    energy = np.mean(X_raw ** 2, axis=1)

    # Frequency domain: drop the DC bin so the gravity / offset term does not dominate.
    mag = np.abs(np.fft.rfft(X_raw, axis=1))[:, 1:, :]  # (n, T//2, C)
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)[1:]

    psd = mag ** 2
    p = psd / (psd.sum(axis=1, keepdims=True) + _EPS)
    entropy = -np.sum(p * np.log(p + _EPS), axis=1)

    dom_freq = freqs[np.argmax(mag, axis=1)]

    # Stack to (n, C, 7) then flatten channel-major -> (n, C*7).
    X = np.stack([mean, std, mn, mx, energy, entropy, dom_freq], axis=2).reshape(n, C * len(FEATURES))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feature_names() -> list:
    """63 names in the exact column order of ``extract_features``."""
    return [f"{ch}_{f}" for ch in CHANNELS for f in FEATURES]
