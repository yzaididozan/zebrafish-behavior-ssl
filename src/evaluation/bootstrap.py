"""Bootstrap utilities for confirmatory zebrafish evaluation.

This module is intentionally dataset-agnostic. It operates on arrays that have
already been restricted to permitted TRAIN/VALIDATION partitions.

Frozen governance implemented here:
- bootstrap unit = fish
- default bootstrap replicates = 500
- default seed = 20260822
- percentile 95% confidence intervals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


DEFAULT_BOOTSTRAP_REPLICATES = 500
DEFAULT_BOOTSTRAP_SEED = 20260822


@dataclass(frozen=True)
class BootstrapSummary:
    """Summary of a bootstrap distribution."""

    point_estimate: float
    median: float
    q25: float
    q75: float
    ci_low: float
    ci_high: float
    n_replicates: int


def percentile_interval(
    values: np.ndarray | Iterable[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a percentile confidence interval."""
    x = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")

    alpha = 1.0 - confidence
    return (
        float(np.quantile(x, alpha / 2.0)),
        float(np.quantile(x, 1.0 - alpha / 2.0)),
    )


def bootstrap_fish_ids(
    fish_ids: np.ndarray,
    *,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Sample unique fish IDs with replacement for each bootstrap replicate.

    The returned arrays contain fish IDs, not bout indices. This makes it
    explicit that the statistical resampling unit is the fish.
    """
    fish_ids = np.asarray(fish_ids)
    if fish_ids.ndim != 1:
        raise ValueError("fish_ids must be 1D.")
    if fish_ids.size == 0:
        raise ValueError("fish_ids cannot be empty.")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive.")

    unique_fish = np.unique(fish_ids)
    rng = np.random.default_rng(seed)

    return [
        rng.choice(unique_fish, size=unique_fish.size, replace=True)
        for _ in range(n_replicates)
    ]


def indices_for_bootstrap_fish_sample(
    fish_ids: np.ndarray,
    sampled_fish_ids: np.ndarray,
) -> np.ndarray:
    """Convert a fish-level bootstrap sample to bout indices.

    If a fish is sampled multiple times, all of that fish's bouts are repeated
    the same number of times in the returned index array.
    """
    fish_ids = np.asarray(fish_ids)
    sampled_fish_ids = np.asarray(sampled_fish_ids)

    pieces: list[np.ndarray] = []
    for fish in sampled_fish_ids:
        idx = np.flatnonzero(fish_ids == fish)
        if idx.size == 0:
            raise ValueError(f"Sampled fish {fish!r} was not found in fish_ids.")
        pieces.append(idx)

    return np.concatenate(pieces) if pieces else np.empty(0, dtype=int)


def fish_bootstrap_statistic(
    fish_ids: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[np.ndarray, BootstrapSummary]:
    """Bootstrap any statistic whose callable accepts bout indices.

    Parameters
    ----------
    fish_ids:
        One fish identifier per bout.
    statistic_fn:
        Called as ``statistic_fn(indices)``.
    """
    fish_ids = np.asarray(fish_ids)
    point = float(statistic_fn(np.arange(fish_ids.size)))

    samples = bootstrap_fish_ids(
        fish_ids,
        n_replicates=n_replicates,
        seed=seed,
    )

    values = np.empty(n_replicates, dtype=float)
    for i, sample in enumerate(samples):
        idx = indices_for_bootstrap_fish_sample(fish_ids, sample)
        values[i] = float(statistic_fn(idx))

    ci_low, ci_high = percentile_interval(values, confidence=confidence)
    summary = BootstrapSummary(
        point_estimate=point,
        median=float(np.median(values)),
        q25=float(np.quantile(values, 0.25)),
        q75=float(np.quantile(values, 0.75)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_replicates=n_replicates,
    )
    return values, summary
