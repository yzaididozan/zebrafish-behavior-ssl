"""Small prediction-only model containers used by final evaluations."""

from __future__ import annotations

import numpy as np


class FrozenKMeansPredictor:
    """KMeans inference from immutable, previously fitted cluster centers."""

    def __init__(self, cluster_centers: np.ndarray) -> None:
        centers = np.asarray(cluster_centers, dtype=np.float64)
        if centers.ndim != 2 or not np.isfinite(centers).all():
            raise ValueError("Invalid frozen KMeans centers")
        self.cluster_centers_ = centers

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        squared = (
            np.sum(values * values, axis=1, keepdims=True)
            + np.sum(self.cluster_centers_ * self.cluster_centers_, axis=1)[None, :]
            - 2.0 * values @ self.cluster_centers_.T
        )
        return np.sqrt(np.maximum(squared, 0.0))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmin(self.transform(x), axis=1).astype(np.int64)
