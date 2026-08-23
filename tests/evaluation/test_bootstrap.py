import numpy as np
import pytest

from src.evaluation.bootstrap import (
    bootstrap_fish_ids,
    fish_bootstrap_statistic,
    indices_for_bootstrap_fish_sample,
    percentile_interval,
)


def test_percentile_interval_basic():
    values = np.arange(100, dtype=float)
    low, high = percentile_interval(values, confidence=0.95)

    assert low == pytest.approx(np.quantile(values, 0.025))
    assert high == pytest.approx(np.quantile(values, 0.975))


def test_percentile_interval_rejects_invalid_input():
    with pytest.raises(ValueError):
        percentile_interval(np.array([]))

    with pytest.raises(ValueError):
        percentile_interval(np.ones((2, 2)))

    with pytest.raises(ValueError):
        percentile_interval(np.arange(5), confidence=1.0)


def test_bootstrap_fish_ids_is_deterministic():
    fish_ids = np.array(["f1", "f1", "f2", "f2", "f3"])

    a = bootstrap_fish_ids(fish_ids, n_replicates=5, seed=123)
    b = bootstrap_fish_ids(fish_ids, n_replicates=5, seed=123)

    assert len(a) == 5
    assert len(b) == 5

    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_bootstrap_fish_ids_samples_unique_fish_count():
    fish_ids = np.array(["f1", "f1", "f2", "f2", "f3", "f3"])
    samples = bootstrap_fish_ids(fish_ids, n_replicates=4, seed=5)

    for sample in samples:
        assert sample.shape == (3,)
        assert set(sample).issubset({"f1", "f2", "f3"})


def test_indices_for_bootstrap_fish_sample_repeats_entire_fish():
    fish_ids = np.array(["f1", "f1", "f2", "f3", "f3"])
    sampled = np.array(["f1", "f3", "f1"])

    idx = indices_for_bootstrap_fish_sample(fish_ids, sampled)

    np.testing.assert_array_equal(idx, np.array([0, 1, 3, 4, 0, 1]))


def test_fish_bootstrap_statistic_returns_summary():
    fish_ids = np.array(["f1", "f1", "f2", "f2", "f3", "f3"])
    values = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    dist, summary = fish_bootstrap_statistic(
        fish_ids,
        lambda idx: float(np.mean(values[idx])),
        n_replicates=20,
        seed=7,
    )

    assert dist.shape == (20,)
    assert summary.n_replicates == 20
    assert summary.point_estimate == pytest.approx(np.mean(values))
    assert summary.ci_low <= summary.median <= summary.ci_high
