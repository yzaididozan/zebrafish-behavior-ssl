"""Figure-generation functions for frozen zebrafish evaluation outputs.

These functions accept already-computed results. They do not fit models,
choose hyperparameters, or load TEST data.

No scientific figure is considered final merely because this module can draw
it; final figures should be generated only from the appropriate frozen result
artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _save_or_return(fig, output_path: str | Path | None):
    if output_path is None:
        return fig

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def plot_cluster_occupancy(
    clusters: Sequence[str | int],
    median_occupancy: Sequence[float],
    *,
    output_path: str | Path | None = None,
    title: str = "Held-out cluster occupancy",
):
    """Bar plot of median per-fish occupancy by cluster."""
    if len(clusters) != len(median_occupancy):
        raise ValueError("clusters and median_occupancy must align.")

    fig, ax = plt.subplots()
    x = np.arange(len(clusters))
    ax.bar(x, median_occupancy)
    ax.set_xticks(x, [str(c) for c in clusters])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Median per-fish occupancy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_seed_metric(
    metric_by_seed: Mapping[int | str, float],
    *,
    ylabel: str,
    output_path: str | Path | None = None,
    title: str = "Metric by seed",
    reference_lines: Sequence[float] = (),
):
    """Plot one frozen metric across seeds."""
    items = sorted(metric_by_seed.items(), key=lambda kv: str(kv[0]))
    seeds = [str(k) for k, _ in items]
    values = [float(v) for _, v in items]

    fig, ax = plt.subplots()
    ax.plot(seeds, values, marker="o")
    for value in reference_lines:
        ax.axhline(float(value), linestyle="--")
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_agreement_comparison(
    labels: Sequence[str],
    ari: Sequence[float],
    nmi: Sequence[float],
    *,
    output_path: str | Path | None = None,
    title: str = "Partition agreement",
):
    """Grouped bars for ARI and NMI across comparisons."""
    if not (len(labels) == len(ari) == len(nmi)):
        raise ValueError("labels, ari, and nmi must have equal length.")

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, ari, width, label="ARI")
    ax.bar(x + width / 2, nmi, width, label="NMI")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Agreement")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_probe_performance(
    labels: Sequence[str],
    balanced_accuracy: Sequence[float],
    chance: Sequence[float],
    *,
    output_path: str | Path | None = None,
    title: str = "Nuisance prediction",
):
    """Compare nuisance-probe balanced accuracy against chance."""
    if not (
        len(labels)
        == len(balanced_accuracy)
        == len(chance)
    ):
        raise ValueError(
            "labels, balanced_accuracy, and chance must have equal length."
        )

    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, balanced_accuracy, label="Observed")
    ax.scatter(x, chance, marker="_", s=250, label="Uniform chance")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_speed_regression_summary(
    seed_labels: Sequence[str],
    r2_values: Sequence[float],
    *,
    output_path: str | Path | None = None,
    title: str = "SSL embedding to mean-speed predictability",
):
    """Bar plot of embedding-to-speed R² by seed."""
    if len(seed_labels) != len(r2_values):
        raise ValueError("seed_labels and r2_values must align.")

    fig, ax = plt.subplots()
    x = np.arange(len(seed_labels))
    ax.bar(x, r2_values)
    ax.set_xticks(x, seed_labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("R²")
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_bootstrap_distribution(
    values: Sequence[float],
    *,
    point_estimate: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    output_path: str | Path | None = None,
    title: str = "Fish-bootstrap metric distribution",
    xlabel: str = "Metric",
):
    """Histogram of a fish-bootstrap metric distribution."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("values must be a non-empty 1D sequence.")

    fig, ax = plt.subplots()
    ax.hist(x, bins="auto")
    if point_estimate is not None:
        ax.axvline(float(point_estimate), linestyle="-", label="Point estimate")
    if ci_low is not None:
        ax.axvline(float(ci_low), linestyle="--", label="95% CI")
    if ci_high is not None:
        ax.axvline(float(ci_high), linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title(title)
    if point_estimate is not None or ci_low is not None or ci_high is not None:
        ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, output_path)
