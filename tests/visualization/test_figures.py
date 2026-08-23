from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from src.visualization.figures import (
    plot_agreement_comparison,
    plot_bootstrap_distribution,
    plot_cluster_occupancy,
    plot_probe_performance,
    plot_seed_metric,
    plot_speed_regression_summary,
)


def test_plot_cluster_occupancy_returns_figure():
    fig = plot_cluster_occupancy(
        [0, 1],
        [0.9, 0.1],
    )

    assert fig is not None


def test_plot_cluster_occupancy_rejects_length_mismatch():
    with pytest.raises(ValueError):
        plot_cluster_occupancy([0, 1], [0.9])


def test_plot_seed_metric_returns_figure():
    fig = plot_seed_metric(
        {11: 0.8, 23: 0.75, 37: 0.78},
        ylabel="ARI",
        reference_lines=(0.75,),
    )

    assert fig is not None


def test_plot_agreement_comparison_returns_figure():
    fig = plot_agreement_comparison(
        ["seed11", "seed23"],
        [0.4, 0.5],
        [0.3, 0.4],
    )

    assert fig is not None


def test_plot_probe_performance_returns_figure():
    fig = plot_probe_performance(
        ["Input A", "SSL"],
        [0.27, 0.20],
        [0.0714, 0.0714],
    )

    assert fig is not None


def test_plot_speed_regression_summary_returns_figure():
    fig = plot_speed_regression_summary(
        ["11", "23"],
        [0.2, 0.3],
    )

    assert fig is not None


def test_plot_bootstrap_distribution_returns_figure():
    fig = plot_bootstrap_distribution(
        [0.6, 0.7, 0.8, 0.75],
        point_estimate=0.73,
        ci_low=0.61,
        ci_high=0.79,
    )

    assert fig is not None


def test_figure_can_be_saved(tmp_path):
    path = tmp_path / "occupancy.png"

    written = plot_cluster_occupancy(
        [0, 1],
        [0.9, 0.1],
        output_path=path,
    )

    assert written == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_bootstrap_distribution_rejects_empty_values():
    with pytest.raises(ValueError):
        plot_bootstrap_distribution([])
