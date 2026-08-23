"""Frozen speed-dependence controls.

These functions are intended for TRAIN/VALIDATION implementation tests and
later confirmatory execution under the frozen evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    r2_score,
)
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class SpeedRegressionResult:
    r2: float
    mae: float


@dataclass(frozen=True)
class SpeedOnlyComparison:
    ari: float
    nmi: float


@dataclass(frozen=True)
class ClusterSpeedSummary:
    cluster: int | str
    n: int
    median: float
    q25: float
    q75: float
    minimum: float
    maximum: float


def fit_speed_ridge(
    train_embeddings: np.ndarray,
    train_speed: np.ndarray,
    eval_embeddings: np.ndarray,
    eval_speed: np.ndarray,
    *,
    alpha: float = 1.0,
) -> tuple[Ridge, SpeedRegressionResult]:
    """Fit frozen Ridge(alpha=1.0) embedding -> mean speed control."""
    x_train = np.asarray(train_embeddings, dtype=float)
    y_train = np.asarray(train_speed, dtype=float)
    x_eval = np.asarray(eval_embeddings, dtype=float)
    y_eval = np.asarray(eval_speed, dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    pred = model.predict(x_eval)

    return model, SpeedRegressionResult(
        r2=float(r2_score(y_eval, pred)),
        mae=float(mean_absolute_error(y_eval, pred)),
    )


def speed_only_cluster_labels(
    train_speed: np.ndarray,
    eval_speed: np.ndarray,
    *,
    method: str,
    k: int,
    seed: int = 20260822,
) -> np.ndarray:
    """Fit one-dimensional speed-only clustering and predict eval labels."""
    train = np.asarray(train_speed, dtype=float).reshape(-1, 1)
    eval_ = np.asarray(eval_speed, dtype=float).reshape(-1, 1)

    method_key = method.lower()
    if method_key in {"gmm", "gaussianmixture", "gaussian_mixture"}:
        model = GaussianMixture(n_components=k, random_state=seed)
    elif method_key in {"kmeans", "k-means"}:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    else:
        raise ValueError("method must be 'gmm' or 'kmeans'.")

    model.fit(train)
    return model.predict(eval_)


def compare_speed_only_to_ssl(
    speed_labels: np.ndarray,
    ssl_labels: np.ndarray,
) -> SpeedOnlyComparison:
    """Compare speed-only and SSL partitions."""
    speed_labels = np.asarray(speed_labels)
    ssl_labels = np.asarray(ssl_labels)
    if speed_labels.shape != ssl_labels.shape:
        raise ValueError("speed_labels and ssl_labels must have the same shape.")

    return SpeedOnlyComparison(
        ari=float(adjusted_rand_score(speed_labels, ssl_labels)),
        nmi=float(normalized_mutual_info_score(speed_labels, ssl_labels)),
    )


def cluster_speed_summaries(
    cluster_labels: np.ndarray,
    mean_speed: np.ndarray,
) -> tuple[list[ClusterSpeedSummary], float, float]:
    """Return cluster speed summaries plus descriptive Kruskal-Wallis H/p."""
    labels = np.asarray(cluster_labels)
    speed = np.asarray(mean_speed, dtype=float)
    if labels.shape != speed.shape:
        raise ValueError("cluster_labels and mean_speed must have the same shape.")

    summaries: list[ClusterSpeedSummary] = []
    groups: list[np.ndarray] = []

    for cluster in np.unique(labels):
        x = speed[labels == cluster]
        groups.append(x)
        summaries.append(
            ClusterSpeedSummary(
                cluster=cluster.item() if hasattr(cluster, "item") else cluster,
                n=int(x.size),
                median=float(np.median(x)),
                q25=float(np.quantile(x, 0.25)),
                q75=float(np.quantile(x, 0.75)),
                minimum=float(np.min(x)),
                maximum=float(np.max(x)),
            )
        )

    if len(groups) < 2:
        return summaries, float("nan"), float("nan")

    h, p = kruskal(*groups)
    return summaries, float(h), float(p)


def interpret_speed_r2(r2: float) -> str:
    """Frozen descriptive R² interpretation bands."""
    if r2 >= 0.75:
        return "very_high"
    if r2 >= 0.50:
        return "high"
    if r2 >= 0.25:
        return "moderate"
    return "low"
