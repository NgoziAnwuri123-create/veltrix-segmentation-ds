"""Internal evaluation shared by every candidate segmentation (no answer key anywhere in this module).

    size_stats       segment shares: smallest, largest, number below MIN_SEGMENT_SHARE
    native_metrics   silhouette (sampled, Euclidean), Calinski-Harabasz, Davies-Bouldin in the model's own space
    common_metrics   Gower silhouette in the curated fv1 space: the yardstick comparable across model types
    stability        mean ARI between the reference labels and refits on random subsamples
"""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.ds.gower import gower_silhouette

MIN_SEGMENT_SHARE = 0.02      # every segment must hold at least 2% of customers to be actionable
STABILITY_TARGET = 0.80


def size_stats(labels: np.ndarray) -> dict:
    shares = pd.Series(labels).value_counts(normalize=True)
    return {"min_share": float(shares.min()), "max_share": float(shares.max()),
            "n_small_segments": int((shares < MIN_SEGMENT_SHARE).sum())}


def native_metrics(X: np.ndarray, labels: np.ndarray, sample: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    return {"silhouette_native": float(silhouette_score(X[idx], labels[idx])),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels))}


def common_metrics(X_curated: pd.DataFrame, labels: np.ndarray, num_cols: list[str], cat_cols: list[str],
                   seed: int = 42) -> dict:
    return {"silhouette_gower": gower_silhouette(X_curated, labels, num_cols, cat_cols, seed=seed)}


def stability(fit_predict: Callable[[np.ndarray, int], np.ndarray], n_rows: int, reference: np.ndarray,
              n_boot: int = 10, frac: float = 0.8, seed: int = 42) -> dict:
    """fit_predict(train_idx, seed) must fit on rows train_idx and return labels for all n_rows."""
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(n_rows, size=int(frac * n_rows), replace=False)
        aris.append(adjusted_rand_score(reference, fit_predict(idx, seed + 1 + b)))
    return {"stability_ari_mean": float(np.mean(aris)), "stability_ari_min": float(np.min(aris))}
