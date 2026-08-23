import numpy as np
import pytest

from src.evaluation.speed_controls import (
    cluster_speed_summaries,
    compare_speed_only_to_ssl,
    fit_speed_ridge,
    interpret_speed_r2,
    speed_only_cluster_labels,
)


def test_fit_speed_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)

    x_train = rng.normal(size=(200, 3))
    y_train = 2.0 * x_train[:, 0] - 0.5 * x_train[:, 1]

    x_eval = rng.normal(size=(80, 3))
    y_eval = 2.0 * x_eval[:, 0] - 0.5 * x_eval[:, 1]

    _, result = fit_speed_ridge(
        x_train,
        y_train,
        x_eval,
        y_eval,
        alpha=1.0,
    )

    assert result.r2 > 0.95
    assert result.mae < 0.2


def test_speed_only_gmm_returns_expected_shape():
    train_speed = np.array([0.1, 0.2, 0.3, 5.0, 5.1, 5.2])
    eval_speed = np.array([0.15, 5.15])

    labels = speed_only_cluster_labels(
        train_speed,
        eval_speed,
        method="gmm",
        k=2,
        seed=42,
    )

    assert labels.shape == (2,)
    assert len(np.unique(labels)) == 2


def test_speed_only_kmeans_returns_expected_shape():
    train_speed = np.array([0.1, 0.2, 0.3, 5.0, 5.1, 5.2])
    eval_speed = np.array([0.15, 5.15])

    labels = speed_only_cluster_labels(
        train_speed,
        eval_speed,
        method="kmeans",
        k=2,
        seed=42,
    )

    assert labels.shape == (2,)
    assert len(np.unique(labels)) == 2


def test_speed_only_cluster_labels_rejects_unknown_method():
    with pytest.raises(ValueError):
        speed_only_cluster_labels(
            np.array([1.0, 2.0]),
            np.array([1.5]),
            method="unknown",
            k=2,
        )


def test_compare_speed_only_to_ssl_perfect_match():
    speed_labels = np.array([0, 0, 1, 1])
    ssl_labels = np.array([1, 1, 0, 0])

    result = compare_speed_only_to_ssl(speed_labels, ssl_labels)

    assert result.ari == pytest.approx(1.0)
    assert result.nmi == pytest.approx(1.0)


def test_cluster_speed_summaries():
    labels = np.array([0, 0, 1, 1])
    speed = np.array([1.0, 2.0, 10.0, 12.0])

    summaries, h, p = cluster_speed_summaries(labels, speed)
    by_cluster = {s.cluster: s for s in summaries}

    assert by_cluster[0].median == pytest.approx(1.5)
    assert by_cluster[1].median == pytest.approx(11.0)
    assert np.isfinite(h)
    assert np.isfinite(p)


@pytest.mark.parametrize(
    ("r2", "expected"),
    [
        (0.80, "very_high"),
        (0.60, "high"),
        (0.30, "moderate"),
        (0.10, "low"),
    ],
)
def test_interpret_speed_r2(r2, expected):
    assert interpret_speed_r2(r2) == expected
