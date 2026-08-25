"""Tests for DS-005 hand-engineered baseline features."""

from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.requires_ds005


REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_DATA = REPO_ROOT / "src" / "data"
SRC_FEATURES = REPO_ROOT / "src" / "features"

for path in (SRC_DATA, SRC_FEATURES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ds005 import DS005  # noqa: E402
from baseline import (  # noqa: E402
    CORE_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    TrainOnlyStandardScaler,
    audit_feature_matrix,
    extract_features,
    extract_fish_features,
    fit_train_scaler,
    transform_feature_matrix,
)


@pytest.fixture(scope="module")
def ds():
    dataset = DS005(repo_root=REPO_ROOT)
    yield dataset
    dataset.close()


@pytest.fixture(scope="module")
def train_fish(ds):
    return ds.fish_in_partition("train")[0]


@pytest.fixture(scope="module")
def train_fish_features(ds, train_fish):
    return extract_fish_features(
        ds,
        train_fish,
        profile="core",
    )


def test_core_feature_schema():
    assert len(CORE_FEATURE_NAMES) == 18

    assert CORE_FEATURE_NAMES == (
        "bout_duration_s",
        "inter_bout_interval_s",
        "speed_mean",
        "speed_std",
        "speed_median",
        "speed_max",
        "speed_p95",
        "speed_rms",
        "accel_abs_mean",
        "accel_abs_std",
        "accel_abs_max",
        "accel_rms",
        "turn_abs_total_rad",
        "turn_net_rad",
        "turn_abs_mean_rad",
        "turn_abs_std_rad",
        "turn_abs_max_rad",
        "turn_rms_rad",
    )


def test_train_fish_is_expected_fish(train_fish):
    assert train_fish.canonical_fish_id == "DS005-JM-F001"
    assert train_fish.partition == "train"
    assert train_fish.valid_bout_count == 665


def test_one_feature_row_per_valid_bout(
    train_fish_features,
    train_fish,
):
    assert train_fish_features.n_rows == train_fish.valid_bout_count
    assert train_fish_features.n_rows == 665
    assert train_fish_features.n_features == 18


def test_core_feature_matrix_is_finite(train_fish_features):
    assert np.all(np.isfinite(train_fish_features.X))


def test_metadata_matches_fish(train_fish_features, train_fish):
    assert len(train_fish_features.metadata) == 665

    for row in train_fish_features.metadata:
        assert row.dataset_id == "DS-005"
        assert row.fish_id == train_fish.canonical_fish_id
        assert row.session_id == train_fish.canonical_session_id
        assert row.fish_index == train_fish.source_fish_index
        assert row.partition == "train"
        assert row.context_id == train_fish.context_id
        assert row.context_name == train_fish.context_name


def test_bout_indices_are_contiguous(train_fish_features):
    observed = [
        row.bout_index
        for row in train_fish_features.metadata
    ]

    assert observed == list(range(665))


def test_first_bout_duration_matches_source(
    train_fish_features,
):
    feature_index = CORE_FEATURE_NAMES.index(
        "bout_duration_s"
    )

    value = float(
        train_fish_features.X[0, feature_index]
    )

    # Fish 1 first bout:
    # [334, 444] -> 110 frames / 700 Hz.
    assert value == pytest.approx(
        110.0 / 700.0,
        rel=1e-6,
    )


def test_first_inter_bout_interval_is_zero(
    train_fish_features,
):
    feature_index = CORE_FEATURE_NAMES.index(
        "inter_bout_interval_s"
    )

    assert float(
        train_fish_features.X[0, feature_index]
    ) == pytest.approx(0.0)


def test_second_inter_bout_interval_matches_source(
    train_fish_features,
):
    feature_index = CORE_FEATURE_NAMES.index(
        "inter_bout_interval_s"
    )

    value = float(
        train_fish_features.X[1, feature_index]
    )

    # Fish 1:
    # first bout ends at 444
    # second bout begins at 2030
    expected = (2030.0 - 444.0) / 700.0

    assert value == pytest.approx(
        expected,
        rel=1e-6,
    )


def test_speed_summary_relationships(
    train_fish_features,
):
    names = train_fish_features.feature_names

    idx_mean = names.index("speed_mean")
    idx_median = names.index("speed_median")
    idx_max = names.index("speed_max")
    idx_p95 = names.index("speed_p95")
    idx_rms = names.index("speed_rms")

    X = train_fish_features.X

    # These relationships should hold for non-negative speed magnitudes.
    assert np.all(X[:, idx_max] >= X[:, idx_p95])
    assert np.all(X[:, idx_max] >= X[:, idx_mean])
    assert np.all(X[:, idx_max] >= X[:, idx_median])

    # RMS of non-negative values is >= arithmetic mean.
    assert np.all(
        X[:, idx_rms] + 1e-6 >= X[:, idx_mean]
    )


def test_turn_features_are_bounded_where_expected(
    train_fish_features,
):
    names = train_fish_features.feature_names

    idx_mean = names.index("turn_abs_mean_rad")
    idx_max = names.index("turn_abs_max_rad")

    assert np.all(
        train_fish_features.X[:, idx_mean] >= 0
    )
    assert np.all(
        train_fish_features.X[:, idx_max] >= 0
    )

    # Wrapped orientation step is constrained to pi.
    assert np.all(
        train_fish_features.X[:, idx_max]
        <= np.pi + 1e-5
    )


def test_qc_metadata_for_smoke_test_fish(
    train_fish_features,
):
    assert sum(
        row.all_zero_speed
        for row in train_fish_features.metadata
    ) == 0

    assert sum(
        row.extreme_speed_gt_100
        for row in train_fish_features.metadata
    ) == 0


def test_audit_for_single_train_fish(
    train_fish_features,
):
    audit = audit_feature_matrix(
        train_fish_features
    )

    assert audit["profile"] == "core"
    assert audit["rows"] == 665
    assert audit["features"] == 18
    assert audit["finite"] is True
    assert audit["fish_count"] == 1

    assert audit["partition_row_counts"] == {
        "train": 665
    }

    assert (
        audit["fish_overlap_across_partitions"]
        is False
    )


def test_extract_features_partition_filter(ds):
    features = extract_features(
        ds,
        partition="validation",
        max_fish=1,
        profile="core",
    )

    assert features.n_rows > 0
    assert {
        row.partition
        for row in features.metadata
    } == {"validation"}


def test_extract_features_fish_filter(ds):
    features = extract_features(
        ds,
        fish_ids=["DS005-JM-F001"],
        profile="core",
    )

    assert features.n_rows == 665
    assert {
        row.fish_id
        for row in features.metadata
    } == {"DS005-JM-F001"}


def test_extract_features_missing_fish_raises(ds):
    with pytest.raises(KeyError):
        extract_features(
            ds,
            fish_ids=["DS005-JM-F999"],
            profile="core",
        )


def test_invalid_profile_raises(ds, train_fish):
    with pytest.raises(ValueError):
        extract_fish_features(
            ds,
            train_fish,
            profile="invalid",  # type: ignore[arg-type]
        )


def test_extended_profile_has_expected_schema(
    ds,
    train_fish,
):
    features = extract_fish_features(
        ds,
        train_fish,
        profile="extended",
    )

    assert features.n_features == len(
        EXTENDED_FEATURE_NAMES
    )
    assert features.n_features == 22
    assert features.X.shape == (665, 22)
    assert np.all(np.isfinite(features.X))


def test_core_profile_excludes_head_position_features():
    assert "head_net_displacement" not in CORE_FEATURE_NAMES
    assert "head_path_length" not in CORE_FEATURE_NAMES
    assert "head_mean_step" not in CORE_FEATURE_NAMES
    assert "head_max_step" not in CORE_FEATURE_NAMES


def test_scaler_refuses_non_train_partition(
    train_fish_features,
):
    scaler = TrainOnlyStandardScaler(
        feature_names=train_fish_features.feature_names
    )

    with pytest.raises(ValueError):
        scaler.fit(
            train_fish_features.X,
            partition="validation",
        )


def test_fit_train_scaler_requires_train_only(
    ds,
):
    validation = extract_features(
        ds,
        partition="validation",
        max_fish=1,
        profile="core",
    )

    with pytest.raises(ValueError):
        fit_train_scaler(validation)


def test_train_scaler_fits_train_features(
    train_fish_features,
):
    scaler = fit_train_scaler(
        train_fish_features
    )

    assert scaler.fitted_on_partition == "train"
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None

    assert scaler.mean_.shape == (18,)
    assert scaler.scale_.shape == (18,)

    assert np.all(np.isfinite(scaler.mean_))
    assert np.all(np.isfinite(scaler.scale_))
    assert np.all(scaler.scale_ > 0)


def test_train_scaler_centers_training_data(
    train_fish_features,
):
    scaler = fit_train_scaler(
        train_fish_features
    )

    transformed = scaler.transform(
        train_fish_features.X
    )

    means = np.mean(transformed, axis=0)

    assert np.allclose(
        means,
        np.zeros(18),
        atol=1e-5,
    )


def test_train_scaler_scales_training_data(
    train_fish_features,
):
    scaler = fit_train_scaler(
        train_fish_features
    )

    transformed = scaler.transform(
        train_fish_features.X
    )

    stds = np.std(
        transformed,
        axis=0,
        ddof=0,
    )

    # All current core features vary within fish 1.
    assert np.allclose(
        stds,
        np.ones(18),
        atol=1e-5,
    )


def test_transform_preserves_metadata(
    train_fish_features,
):
    scaler = fit_train_scaler(
        train_fish_features
    )

    transformed = transform_feature_matrix(
        train_fish_features,
        scaler,
    )

    assert transformed.X.shape == (
        train_fish_features.X.shape
    )
    assert (
        transformed.feature_names
        == train_fish_features.feature_names
    )
    assert transformed.profile == "core"
    assert (
        transformed.metadata
        == train_fish_features.metadata
    )


def test_validation_uses_train_fitted_scaler(
    ds,
    train_fish_features,
):
    scaler = fit_train_scaler(
        train_fish_features
    )

    validation = extract_features(
        ds,
        partition="validation",
        max_fish=1,
        profile="core",
    )

    transformed = transform_feature_matrix(
        validation,
        scaler,
    )

    assert transformed.n_rows == validation.n_rows
    assert transformed.n_features == 18

    assert {
        row.partition
        for row in transformed.metadata
    } == {"validation"}

    assert np.all(np.isfinite(transformed.X))


def test_scaler_schema_mismatch_raises(
    train_fish_features,
):
    scaler = TrainOnlyStandardScaler(
        feature_names=("wrong_feature",)
    )

    X = np.ones((10, 1))

    scaler.fit(
        X,
        partition="train",
    )

    with pytest.raises(ValueError):
        scaler.transform(
            train_fish_features.X
        )


def test_feature_context_not_used_as_numeric_input(
    train_fish_features,
):
    assert train_fish_features.X.dtype.kind in {
        "f",
        "i",
        "u",
    }

    assert "context_id" not in train_fish_features.feature_names
    assert "context_name" not in train_fish_features.feature_names
    assert "stimulus_code" not in train_fish_features.feature_names
    assert "bout_type" not in train_fish_features.feature_names


def test_bout_type_and_stimulus_are_metadata_only(
    train_fish_features,
):
    first = train_fish_features.metadata[0]

    assert isinstance(first.stimulus_code, float)
    assert isinstance(first.bout_type, float)

    assert "stimulus_code" not in CORE_FEATURE_NAMES
    assert "bout_type" not in CORE_FEATURE_NAMES


def test_feature_order_is_stable(
    ds,
    train_fish,
):
    first = extract_fish_features(
        ds,
        train_fish,
        profile="core",
    )

    second = extract_fish_features(
        ds,
        train_fish,
        profile="core",
    )

    assert first.feature_names == second.feature_names
    assert np.array_equal(first.X, second.X)


def test_no_partition_leakage_when_extracting_two_train_fish(
    ds,
):
    features = extract_features(
        ds,
        partition="train",
        max_fish=2,
        profile="core",
    )

    assert {
        row.partition
        for row in features.metadata
    } == {"train"}

    fish_ids = {
        row.fish_id
        for row in features.metadata
    }

    assert len(fish_ids) == 2


def test_feature_row_count_equals_sum_of_selected_fish_bouts(
    ds,
):
    selected = ds.fish_in_partition("train")[:2]

    expected = sum(
        fish.valid_bout_count
        for fish in selected
    )

    features = extract_features(
        ds,
        partition="train",
        max_fish=2,
        profile="core",
    )

    assert features.n_rows == expected
