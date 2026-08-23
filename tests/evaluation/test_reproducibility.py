import numpy as np
import pytest

from src.evaluation.reproducibility import (
    align_labels_hungarian,
    fish_bootstrap_ari_from_labels,
    heldout_cluster_occupancy,
    interpret_ari,
    pairwise_seed_ari,
)


def test_align_labels_hungarian_recovers_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2])
    candidate = np.array([2, 2, 0, 0, 1, 1])

    aligned = align_labels_hungarian(reference, candidate)

    np.testing.assert_array_equal(aligned, reference)


def test_pairwise_seed_ari_perfect_when_partitions_equivalent():
    labels = {
        11: np.array([0, 0, 1, 1, 2, 2]),
        23: np.array([1, 1, 2, 2, 0, 0]),
        37: np.array([2, 2, 0, 0, 1, 1]),
    }

    result = pairwise_seed_ari(labels)

    assert len(result.pairwise_ari) == 3
    assert result.median == pytest.approx(1.0)
    assert result.minimum == pytest.approx(1.0)
    assert result.maximum == pytest.approx(1.0)


def test_pairwise_seed_ari_requires_two_seeds():
    with pytest.raises(ValueError):
        pairwise_seed_ari({11: np.array([0, 1])})


def test_heldout_cluster_occupancy_summarizes_by_fish():
    labels = np.array([0, 0, 1, 1, 0, 1])
    fish_ids = np.array(["f1", "f1", "f2", "f2", "f3", "f3"])

    results = heldout_cluster_occupancy(labels, fish_ids)
    by_cluster = {r.cluster: r for r in results}

    assert by_cluster[0].n_fish_with_cluster == 2
    assert by_cluster[1].n_fish_with_cluster == 2

    assert by_cluster[0].median_per_fish_occupancy == pytest.approx(0.5)
    assert by_cluster[1].median_per_fish_occupancy == pytest.approx(0.5)


def test_fish_bootstrap_ari_is_one_for_identical_labels():
    labels = np.array([0, 0, 1, 1, 0, 1])
    fish_ids = np.array(["f1", "f1", "f2", "f2", "f3", "f3"])

    scores, summary = fish_bootstrap_ari_from_labels(
        labels,
        labels.copy(),
        fish_ids,
        n_replicates=25,
        seed=1,
    )

    np.testing.assert_allclose(scores, 1.0)
    assert summary.point_estimate == pytest.approx(1.0)
    assert summary.median == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.80, "strong"),
        (0.60, "moderate"),
        (0.30, "weak"),
        (0.10, "poor"),
    ],
)
def test_interpret_ari(score, expected):
    assert interpret_ari(score) == expected
