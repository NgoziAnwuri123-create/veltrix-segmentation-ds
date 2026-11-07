"""A fitted segmentation = preprocessor + clustering model behind one predict(DataFrame) -> labels call.

Encodings of the curated fv1 space:
    encoded  scaled numeric columns + one-hot categoricals x sqrt(emphasis)   (k-means, Gaussian mixture)
    mixed    scaled numeric columns + raw categorical strings                 (k-prototypes, gamma = 2 x emphasis)

Categorical emphasis e: a mismatch on one categorical variable adds 2e to the squared distance, and two random
customers differ by 2 on one standardised numeric feature in expectation. So e = 1 gives every variable the same
expected weight.

SegmenterPyfunc wraps a Segmenter for MLflow; scoring (d07) loads any candidate the same way.
"""
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.ds.preprocess import SegmentationPreprocessor


def encode(Xt: pd.DataFrame, pre: SegmentationPreprocessor, emphasis: float = 1.0) -> np.ndarray:
    w = np.sqrt(emphasis)
    blocks = [Xt[pre.num_cols_].to_numpy(dtype=np.float64)]
    for col in pre.cat_cols_:
        cats = pre.categories_[col]
        blocks.append((Xt[col].to_numpy()[:, None] == np.array(cats)[None, :]).astype(np.float64) * w)
    return np.hstack(blocks)


def mixed(Xt: pd.DataFrame, pre: SegmentationPreprocessor) -> tuple[np.ndarray, list[int]]:
    arr = np.column_stack([Xt[pre.num_cols_].to_numpy(dtype=np.float64), Xt[pre.cat_cols_].to_numpy()]).astype(object)
    return arr, list(range(len(pre.num_cols_), arr.shape[1]))


class Segmenter:
    """label_map (optional) renames raw cluster numbers, e.g. to segment ids ordered by size."""

    def __init__(self, pre: SegmentationPreprocessor, model, kind: str, emphasis: float = 1.0, label_map=None):
        self.pre, self.model, self.kind, self.emphasis = pre, model, kind, emphasis
        self.label_map = label_map

    def _relabel(self, raw: np.ndarray) -> np.ndarray:
        mapping = getattr(self, "label_map", None)   # pickles from d03 predate label_map
        return raw if mapping is None else np.vectorize(mapping.__getitem__)(raw)

    def _raw_predict(self, Xt: pd.DataFrame) -> np.ndarray:
        if self.kind == "kproto":
            arr, cat_idx = mixed(Xt, self.pre)
            return np.asarray(self.model.predict(arr, categorical=cat_idx))
        return np.asarray(self.model.predict(encode(Xt, self.pre, self.emphasis)))

    def predict_transformed(self, Xt: pd.DataFrame) -> np.ndarray:
        return self._relabel(self._raw_predict(Xt))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_transformed(self.pre.transform(df))

    def confidence(self, df: pd.DataFrame) -> np.ndarray:
        """0-1 assignment confidence. Centroid models: (d2 - d1) / d2, the relative margin between the nearest and
        second-nearest centre (0 = on the boundary). Gaussian mixture: highest membership probability."""
        Xt = self.pre.transform(df)
        if self.kind == "gmm":
            return self.model.predict_proba(encode(Xt, self.pre, self.emphasis)).max(axis=1)
        if self.kind == "kmeans":
            d = self.model.transform(encode(Xt, self.pre, self.emphasis)) ** 2
        else:  # k-prototypes cost: squared Euclidean on numeric + gamma x categorical mismatches
            n_num = len(self.pre.num_cols_)
            num = Xt[self.pre.num_cols_].to_numpy(dtype=np.float64)
            cat = Xt[self.pre.cat_cols_].to_numpy()
            centres = self.model.cluster_centroids_
            c_num, c_cat = centres[:, :n_num].astype(np.float64), centres[:, n_num:]
            d = np.empty((len(Xt), len(centres)))
            for s in range(0, len(Xt), 10_000):   # chunked: the full 3-D difference array is ~0.5 GB
                e = slice(s, s + 10_000)
                d[e] = ((num[e, None, :] - c_num[None, :, :]) ** 2).sum(axis=2)
                d[e] += self.model.gamma * (cat[e, None, :] != c_cat[None, :, :]).sum(axis=2)
        d.sort(axis=1)
        return np.where(d[:, 1] > 0, (d[:, 1] - d[:, 0]) / d[:, 1], 0.0)


class SegmenterPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, segmenter: Segmenter):
        self.segmenter = segmenter

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.segmenter.predict(model_input)
