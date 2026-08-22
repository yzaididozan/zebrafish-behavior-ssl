"""Train-only normalization for DS-005 SSL inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.ds005 import DS005


@dataclass(frozen=True)
class SSLNormalizationStats:
    speed_mean: float
    speed_std: float


def fit_training_speed_stats(dataset: DS005) -> SSLNormalizationStats:
    """Estimate speed normalization using TRAIN fish only.

    Uses an online algorithm so the full dataset is never loaded into memory.
    """

    count = 0
    mean = 0.0
    m2 = 0.0

    for bout in dataset.iter_bouts(
        partition="train",
        primary_qc_only=True,
        include_optional=False,
    ):
        speed = np.asarray(bout.speed_head, dtype=np.float64)

        if not np.all(np.isfinite(speed)):
            raise ValueError(
                f"Non-finite speed in "
                f"{bout.key.fish_id}/{bout.key.bout_index}"
            )

        for value in speed:
            count += 1

            delta = value - mean
            mean += delta / count

            delta2 = value - mean
            m2 += delta * delta2

    if count < 2:
        raise RuntimeError("Insufficient training samples.")

    variance = m2 / count
    std = float(np.sqrt(variance))

    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(
            f"Invalid training speed std: {std}"
        )

    return SSLNormalizationStats(
        speed_mean=float(mean),
        speed_std=std,
    )


def normalize_ssl_input(
    X: np.ndarray,
    stats: SSLNormalizationStats,
) -> np.ndarray:
    """Apply frozen training-only normalization to one SSL input."""

    X = np.asarray(X, dtype=np.float32).copy()

    if X.shape != (175, 3):
        raise ValueError(
            f"Expected SSL input shape (175, 3), got {X.shape}"
        )

    # Channels 0 and 1 are sin/cos orientation and remain unchanged.
    X[:, 2] = (
        X[:, 2] - stats.speed_mean
    ) / stats.speed_std

    if not np.all(np.isfinite(X)):
        raise RuntimeError(
            "Normalization produced non-finite values."
        )

    return X