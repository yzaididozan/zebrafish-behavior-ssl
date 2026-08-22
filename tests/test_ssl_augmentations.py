"""Tests for DS-005 SSL augmentations."""

from __future__ import annotations

import numpy as np

from src.ssl.augmentations import (
    AugmentationConfig,
    EXPECTED_SHAPE,
    feature_mask,
    make_two_views,
    temporal_mask,
    validate_ssl_input,
)


def make_dummy_input() -> np.ndarray:
    """Create a deterministic synthetic SSL input with shape (175, 3)."""
    t = np.linspace(0.0, 1.0, EXPECTED_SHAPE[0], dtype=np.float32)

    X = np.column_stack(
        [
            np.sin(2.0 * np.pi * t),
            np.cos(2.0 * np.pi * t),
            1.0 + 4.0 * t,
        ]
    ).astype(np.float32)

    return X


def test_validate_ssl_input_accepts_valid_input() -> None:
    X = make_dummy_input()

    validated = validate_ssl_input(X)

    assert validated.shape == (175, 3)
    assert validated.dtype == np.float32
    assert np.all(np.isfinite(validated))


def test_validate_ssl_input_rejects_wrong_shape() -> None:
    X = np.zeros((174, 3), dtype=np.float32)

    try:
        validate_ssl_input(X)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for incorrect SSL input shape.")


def test_validate_ssl_input_rejects_nonfinite_values() -> None:
    X = make_dummy_input()
    X[0, 0] = np.nan

    try:
        validate_ssl_input(X)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for NaN input.")


def test_temporal_mask_preserves_shape_and_finiteness() -> None:
    X = make_dummy_input()
    original = X.copy()

    rng = np.random.default_rng(123)

    masked = temporal_mask(
        X,
        rng=rng,
        max_fraction=0.10,
        mask_value=0.0,
    )

    assert masked.shape == X.shape
    assert masked.dtype == np.float32
    assert np.all(np.isfinite(masked))

    # Input must not be modified in place.
    assert np.array_equal(X, original)

    # At least one value should differ because max_fraction > 0.
    assert not np.array_equal(masked, original)


def test_temporal_mask_is_deterministic_with_same_seed() -> None:
    X = make_dummy_input()

    a = temporal_mask(
        X,
        rng=np.random.default_rng(20260822),
        max_fraction=0.10,
    )

    b = temporal_mask(
        X,
        rng=np.random.default_rng(20260822),
        max_fraction=0.10,
    )

    assert np.array_equal(a, b)


def test_feature_mask_preserves_orientation_pairing() -> None:
    X = make_dummy_input()

    masked = feature_mask(
        X,
        rng=np.random.default_rng(1),
        allow_orientation_mask=True,
        allow_speed_mask=False,
    )

    # Orientation channels must be masked together.
    assert np.all(masked[:, 0] == 0.0)
    assert np.all(masked[:, 1] == 0.0)

    # Speed must remain unchanged.
    assert np.array_equal(masked[:, 2], X[:, 2])


def test_feature_mask_can_mask_speed_only() -> None:
    X = make_dummy_input()

    masked = feature_mask(
        X,
        rng=np.random.default_rng(2),
        allow_orientation_mask=False,
        allow_speed_mask=True,
    )

    assert np.array_equal(masked[:, 0], X[:, 0])
    assert np.array_equal(masked[:, 1], X[:, 1])
    assert np.all(masked[:, 2] == 0.0)


def test_make_two_views_preserves_shape_and_finiteness() -> None:
    X = make_dummy_input()
    original = X.copy()

    config = AugmentationConfig(
        temporal_mask_probability=1.0,
        temporal_mask_max_fraction=0.10,
        feature_mask_probability=1.0,
    )

    view_a, view_b = make_two_views(
        X,
        seed=20260822,
        config=config,
    )

    assert view_a.shape == (175, 3)
    assert view_b.shape == (175, 3)

    assert view_a.dtype == np.float32
    assert view_b.dtype == np.float32

    assert np.all(np.isfinite(view_a))
    assert np.all(np.isfinite(view_b))

    # Source array must remain untouched.
    assert np.array_equal(X, original)


def test_make_two_views_is_reproducible_with_same_seed() -> None:
    X = make_dummy_input()

    config = AugmentationConfig(
        temporal_mask_probability=1.0,
        temporal_mask_max_fraction=0.10,
        feature_mask_probability=1.0,
    )

    a1, b1 = make_two_views(
        X,
        seed=777,
        config=config,
    )

    a2, b2 = make_two_views(
        X,
        seed=777,
        config=config,
    )

    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


def test_different_seeds_can_produce_different_views() -> None:
    X = make_dummy_input()

    config = AugmentationConfig(
        temporal_mask_probability=1.0,
        temporal_mask_max_fraction=0.10,
        feature_mask_probability=1.0,
    )

    a1, b1 = make_two_views(
        X,
        seed=100,
        config=config,
    )

    a2, b2 = make_two_views(
        X,
        seed=200,
        config=config,
    )

    # It is sufficient that at least one paired view differs.
    assert (
        not np.array_equal(a1, a2)
        or not np.array_equal(b1, b2)
    )


def test_zero_probability_config_returns_original_views() -> None:
    X = make_dummy_input()

    config = AugmentationConfig(
        temporal_mask_probability=0.0,
        temporal_mask_max_fraction=0.10,
        feature_mask_probability=0.0,
    )

    view_a, view_b = make_two_views(
        X,
        seed=42,
        config=config,
    )

    assert np.array_equal(view_a, X)
    assert np.array_equal(view_b, X)


def test_seed_and_rng_cannot_both_be_supplied() -> None:
    X = make_dummy_input()

    try:
        make_two_views(
            X,
            seed=1,
            rng=np.random.default_rng(1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError when both seed and rng are supplied."
        )
