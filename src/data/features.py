"""Hand-crafted time/frequency features from raw windows."""
import numpy as np
from scipy.stats import entropy

FS = 50  # Hz

def extract_features(X_raw: np.ndarray) -> np.ndarray:
    """X_raw (n, 128, 9) -> X (n, d). Per channel: mean, std, min, max, energy,
    entropy (histogram-based), dominant FFT frequency (np.fft.rfft / rfftfreq)."""
    raise NotImplementedError

def feature_names() -> list:
    raise NotImplementedError
