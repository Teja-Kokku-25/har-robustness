import numpy as np

def scatter_matrices(Z, y):
    """Return (S_W, S_B)."""
    raise NotImplementedError

def fisher_score(Z, y) -> float:
    """trace(S_B) / trace(S_W) on embedding Z. Higher = more separable."""
    raise NotImplementedError
