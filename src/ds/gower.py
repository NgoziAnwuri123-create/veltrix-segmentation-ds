"""Gower distance for mixed numeric/categorical data, and a sampled Gower silhouette.

Gower distance between two customers = mean over features of
    |x_i - x_j| / range   for numeric features (range taken from the full data, not the sample)
    0 if equal, else 1    for categorical features
It puts every candidate model (k-means, k-prototypes, mixtures) on one common yardstick.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def gower_matrix(num: np.ndarray, cat: np.ndarray, ranges: np.ndarray, chunk: int = 100) -> np.ndarray:
    """Pairwise Gower distances (n x n, float32) for numeric array `num` and integer-coded `cat`."""
    n = num.shape[0]
    p = num.shape[1] + cat.shape[1]
    num = (num / np.where(ranges > 0, ranges, 1)).astype(np.float32)
    out = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = np.abs(num[start:stop, None, :] - num[None, :, :]).sum(axis=2)
        d += (cat[start:stop, None, :] != cat[None, :, :]).sum(axis=2)
        out[start:stop] = d / p
    np.fill_diagonal(out, 0.0)
    return out


def gower_silhouette(X: pd.DataFrame, labels: np.ndarray, num_cols: list[str], cat_cols: list[str],
                     n: int = 4000, repeats: int = 3, seed: int = 42) -> float:
    """Mean silhouette over `repeats` random samples of `n` rows, using Gower distance on X."""
    num_all = X[num_cols].to_numpy(dtype=np.float64)
    ranges = num_all.max(axis=0) - num_all.min(axis=0)
    cat_all = np.column_stack([pd.factorize(X[c])[0] for c in cat_cols]) if cat_cols else np.empty((len(X), 0), int)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        if len(np.unique(labels[idx])) < 2:
            continue
        D = gower_matrix(num_all[idx], cat_all[idx], ranges)
        scores.append(silhouette_score(D, labels[idx], metric="precomputed"))
    return float(np.mean(scores)) if scores else float("nan")
