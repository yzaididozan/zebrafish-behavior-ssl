"""Reproducibility metrics for frozen zebrafish discovery analyses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .bootstrap import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    BootstrapSummary,
)


@dataclass(frozen=True)
class CrossSeedSummary:
    pairwise_ari: dict[str, float]
    median: float
    q25: float
    q75: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ClusterOccupancy:
    cluster: int | str
    n_fish_with_cluster: int
    median_per_fish_occupancy: float
    q25: float
    q75: float
    minimum: float
    maximum: float


def align_labels_hungarian(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> np.ndarray:
    """Align candidate cluster IDs to reference IDs by maximum overlap."""
    ref = np.asarray(reference_labels)
    cand = np.asarray(candidate_labels)
    if ref.shape != cand.shape or ref.ndim != 1:
        raise ValueError("reference_labels and candidate_labels must be equal-length 1D arrays.")

    ref_values = np.unique(ref)
    cand_values = np.unique(cand)

    overlap = np.zeros((ref_values.size, cand_values.size), dtype=np.int64)
    for i, rv in enumerate(ref_values):
        for j, cv in enumerate(cand_values):
            overlap[i, j] = np.sum((ref == rv) & (cand == cv))

    row_ind, col_ind = linear_sum_assignment(-overlap)
    mapping = {cand_values[c]: ref_values[r] for r, c in zip(row_ind, col_ind)}

    # Candidate-only unmatched labels get deterministic fresh integer labels.
    aligned = cand.copy()
    unmatched = [v for v in cand_values if v not in mapping]
    next_label = int(np.max(ref_values)) + 1 if np.issubdtype(ref_values.dtype, np.integer) else None

    for old, new in mapping.items():
        aligned[cand == old] = new

    if unmatched:
        if next_label is None:
            # For non-integer labels, retain unmatched values unchanged.
            return aligned
        for old in unmatched:
            aligned[cand == old] = next_label
            next_label += 1

    return aligned


def pairwise_seed_ari(
    labels_by_seed: Mapping[int, np.ndarray],
) -> CrossSeedSummary:
    """Compute all pairwise ARIs across frozen SSL seeds."""
    if len(labels_by_seed) < 2:
        raise ValueError("At least two seeds are required.")

    pairwise: dict[str, float] = {}
    values: list[float] = []

    for seed_a, seed_b in combinations(sorted(labels_by_seed), 2):
        a = np.asarray(labels_by_seed[seed_a])
        b = np.asarray(labels_by_seed[seed_b])
        if a.shape != b.shape:
            raise ValueError(f"Seed {seed_a} and {seed_b} labels have different shapes.")
        score = float(adjusted_rand_score(a, b))
        pairwise[f"{seed_a}_vs_{seed_b}"] = score
        values.append(score)

    x = np.asarray(values, dtype=float)
    return CrossSeedSummary(
        pairwise_ari=pairwise,
        median=float(np.median(x)),
        q25=float(np.quantile(x, 0.25)),
        q75=float(np.quantile(x, 0.75)),
        minimum=float(np.min(x)),
        maximum=float(np.max(x)),
    )


def heldout_cluster_occupancy(
    labels: np.ndarray,
    fish_ids: np.ndarray,
) -> list[ClusterOccupancy]:
    """Summarize per-fish cluster occupancy on a held-out partition."""
    labels = np.asarray(labels)
    fish_ids = np.asarray(fish_ids)
    if labels.shape != fish_ids.shape or labels.ndim != 1:
        raise ValueError("labels and fish_ids must be equal-length 1D arrays.")

    unique_fish = np.unique(fish_ids)
    results: list[ClusterOccupancy] = []

    for cluster in np.unique(labels):
        per_fish = []
        for fish in unique_fish:
            mask = fish_ids == fish
            per_fish.append(float(np.mean(labels[mask] == cluster)))

        x = np.asarray(per_fish, dtype=float)
        results.append(
            ClusterOccupancy(
                cluster=cluster.item() if hasattr(cluster, "item") else cluster,
                n_fish_with_cluster=int(np.sum(x > 0)),
                median_per_fish_occupancy=float(np.median(x)),
                q25=float(np.quantile(x, 0.25)),
                q75=float(np.quantile(x, 0.75)),
                minimum=float(np.min(x)),
                maximum=float(np.max(x)),
            )
        )

    return results


def fish_bootstrap_ari_from_labels(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    fish_ids: np.ndarray,
    *,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[np.ndarray, BootstrapSummary]:
    """Fish-bootstrap ARI for two fixed labelings of the same bouts.

    This is appropriate for testing/bootstrap fixtures and fixed-partition
    comparisons. If the confirmatory procedure requires *refitting* a model on
    each fish bootstrap sample, use the higher-level model-refit orchestration
    in the eventual evaluation runner.
    """
    reference_labels = np.asarray(reference_labels)
    candidate_labels = np.asarray(candidate_labels)
    fish_ids = np.asarray(fish_ids)

    if not (
        reference_labels.shape == candidate_labels.shape == fish_ids.shape
        and reference_labels.ndim == 1
    ):
        raise ValueError("reference_labels, candidate_labels, and fish_ids must align.")

    unique_fish = np.unique(fish_ids)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_replicates, dtype=float)

    for i in range(n_replicates):
        sampled_fish = rng.choice(unique_fish, size=unique_fish.size, replace=True)
        idx_parts = [np.flatnonzero(fish_ids == fish) for fish in sampled_fish]
        idx = np.concatenate(idx_parts)
        scores[i] = adjusted_rand_score(reference_labels[idx], candidate_labels[idx])

    point = float(adjusted_rand_score(reference_labels, candidate_labels))
    summary = BootstrapSummary(
        point_estimate=point,
        median=float(np.median(scores)),
        q25=float(np.quantile(scores, 0.25)),
        q75=float(np.quantile(scores, 0.75)),
        ci_low=float(np.quantile(scores, 0.025)),
        ci_high=float(np.quantile(scores, 0.975)),
        n_replicates=n_replicates,
    )
    return scores, summary


def interpret_ari(score: float) -> str:
    """Frozen descriptive ARI interpretation bands."""
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "moderate"
    if score >= 0.25:
        return "weak"
    return "poor"
