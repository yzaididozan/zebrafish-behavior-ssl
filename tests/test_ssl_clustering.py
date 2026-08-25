"""Structural tests for SSL clustering utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.discovery.ssl_clustering import (
    deterministic_indices,
    fit_preprocessing,
    selection_score,
)


def test_selection_score_matches_documented_weights() -> None:
    score = selection_score(
        validation_silhouette=0.5,
        stability_ari=1.0,
        validation_min_cluster_fraction=0.05,
        validation_empty_clusters=0,
    )
    assert score == pytest.approx(0.7)


def test_deterministic_indices_are_reproducible() -> None:
    a = deterministic_indices(
        100,
        max_rows=20,
        seed=11,
    )
    b = deterministic_indices(
        100,
        max_rows=20,
        seed=11,
    )

    np.testing.assert_array_equal(a, b)


def test_fit_preprocessing_uses_expected_shapes() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(size=(200, 64)).astype(np.float32)
    validation = rng.normal(size=(50, 64)).astype(np.float32)

    scaler, pca, train_pca, validation_pca = fit_preprocessing(
        train,
        validation,
        pca_variance=0.95,
        random_state=20260822,
    )

    assert train_pca.shape[0] == 200
    assert validation_pca.shape[0] == 50
    assert train_pca.shape[1] == validation_pca.shape[1]
    assert pca.n_components_ <= 64
    assert np.isfinite(train_pca).all()
    assert np.isfinite(validation_pca).all()
