import numpy as np
import pytest

from src.evaluation.compare_representations import (
    compare_clusterings,
    fit_input_a_to_ssl_cluster_probe,
    summarize_seedwise_partition_comparisons,
)


def test_compare_clusterings_is_permutation_invariant():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])

    result = compare_clusterings(a, b)

    assert result.ari == pytest.approx(1.0)
    assert result.nmi == pytest.approx(1.0)


def test_compare_clusterings_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_clusterings(
            np.array([0, 1]),
            np.array([0, 1, 1]),
        )


def test_input_a_probe_predicts_ssl_clusters_when_signal_is_easy():
    rng = np.random.default_rng(0)

    x_train = np.vstack([
        rng.normal(loc=-2.0, scale=0.3, size=(100, 3)),
        rng.normal(loc=2.0, scale=0.3, size=(100, 3)),
    ])
    y_train = np.array([0] * 100 + [1] * 100)

    x_eval = np.vstack([
        rng.normal(loc=-2.0, scale=0.3, size=(40, 3)),
        rng.normal(loc=2.0, scale=0.3, size=(40, 3)),
    ])
    y_eval = np.array([0] * 40 + [1] * 40)

    _, result = fit_input_a_to_ssl_cluster_probe(
        x_train,
        y_train,
        x_eval,
        y_eval,
    )

    assert result.balanced_accuracy > 0.95
    assert result.macro_f1 > 0.95


def test_seedwise_partition_summary():
    input_a = np.array([0, 0, 1, 1, 2, 2])

    input_b_by_seed = {
        11: np.array([0, 0, 1, 1, 2, 2]),
        23: np.array([2, 2, 0, 0, 1, 1]),
        37: np.array([0, 1, 0, 1, 2, 2]),
    }

    summary = summarize_seedwise_partition_comparisons(
        input_a,
        input_b_by_seed,
    )

    assert set(summary["per_seed"]) == {11, 23, 37}
    assert summary["ari"]["maximum"] == pytest.approx(1.0)
    assert summary["nmi"]["maximum"] == pytest.approx(1.0)
    assert summary["ari"]["minimum"] <= summary["ari"]["median"]
    assert summary["nmi"]["minimum"] <= summary["nmi"]["median"]
