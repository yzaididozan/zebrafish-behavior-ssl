"""Tests for the canonical frozen DS-005 loader."""

from pathlib import Path
import sys

import pytest


# Allow importing src/data/ds005.py without requiring package installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DATA = REPO_ROOT / "src" / "data"

if str(SRC_DATA) not in sys.path:
    sys.path.insert(0, str(SRC_DATA))

from ds005 import (  # noqa: E402
    DS005,
    EXPECTED_CONTEXTS,
    EXPECTED_FISH,
    EXPECTED_VALID_BOUTS,
    FROZEN_SPLIT_SHA256,
    sha256_file,
)


@pytest.fixture(scope="module")
def ds():
    dataset = DS005(repo_root=REPO_ROOT)
    yield dataset
    dataset.close()


def test_dataset_counts(ds):
    assert ds.n_fish == EXPECTED_FISH == 463
    assert ds.n_valid_bouts == EXPECTED_VALID_BOUTS == 1_203_409
    assert len(ds.contexts) == EXPECTED_CONTEXTS == 14


def test_partition_counts(ds):
    assert ds.partition_counts() == {
        "train": 323,
        "validation": 70,
        "test": 70,
    }


def test_no_fish_overlap(ds):
    ds.assert_no_fish_overlap()

    train_ids = {
        fish.canonical_fish_id
        for fish in ds.fish_in_partition("train")
    }
    validation_ids = {
        fish.canonical_fish_id
        for fish in ds.fish_in_partition("validation")
    }
    test_ids = {
        fish.canonical_fish_id
        for fish in ds.fish_in_partition("test")
    }

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    assert len(train_ids | validation_ids | test_ids) == 463


def test_every_context_in_every_partition(ds):
    for context in ds.contexts:
        assert len(ds.fish_in_context(context, "train")) > 0
        assert len(ds.fish_in_context(context, "validation")) > 0
        assert len(ds.fish_in_context(context, "test")) > 0


def test_fish_zero_metadata(ds):
    fish = ds.get_fish(0)

    assert fish.dataset_id == "DS-005"
    assert fish.cohort == "JM_data"
    assert fish.canonical_fish_id == "DS005-JM-F000"
    assert fish.canonical_session_id == "DS005-JM-S000"
    assert fish.source_fish_index == 0
    assert fish.valid_bout_count == 1381
    assert fish.context_id == "CTX-09"
    assert fish.context_name == "3 min Light<->Dark(5x5cm)"
    assert fish.partition == "validation"


def test_fish_lookup_by_id_matches_index(ds):
    by_index = ds.get_fish(0)
    by_id = ds.get_fish("DS005-JM-F000")

    assert by_index == by_id


def test_first_bout_shapes_and_metadata(ds):
    bout = ds.load_bout(0, 0)

    assert bout.key.fish_id == "DS005-JM-F000"
    assert bout.key.fish_index == 0
    assert bout.key.bout_index == 0
    assert bout.key.partition == "validation"
    assert bout.key.context_id == "CTX-09"
    assert bout.key.context_name == "3 min Light<->Dark(5x5cm)"

    assert bout.head_pos.shape == (175, 2)
    assert bout.orientation_smooth.shape == (175,)
    assert bout.speed_head.shape == (175,)
    assert bout.times_bouts.shape == (2,)

    assert bout.times_bouts.tolist() == [351.0, 455.0]

    assert bout.qc.contains_nonfinite is False
    assert bout.qc.all_zero_speed is False
    assert bout.qc.extreme_speed_gt_100 is False
    assert bout.qc.primary_exclude is False


def test_invalid_padded_bout_access_raises(ds):
    fish = ds.get_fish(0)

    with pytest.raises(IndexError):
        ds.load_bout(
            fish.source_fish_index,
            fish.valid_bout_count,
        )


def test_negative_bout_index_raises(ds):
    with pytest.raises(IndexError):
        ds.load_bout(0, -1)


def test_frozen_split_hash(ds):
    assert sha256_file(ds.split_path) == FROZEN_SPLIT_SHA256
    assert (
        FROZEN_SPLIT_SHA256
        == "19c1c7589e046337ec51b66b8fec7632029084d59905ca45b2ce751b3268c935"
    )


def test_frame_rate(ds):
    assert ds.frame_rate_hz == pytest.approx(700.0)


def test_fish_field_valid_only_uses_lengths_data(ds):
    fish = ds.get_fish(0)

    speed = ds.load_fish_field(
        fish.source_fish_index,
        "speed_head",
        valid_only=True,
    )

    assert speed.shape == (fish.valid_bout_count, 175)
    assert speed.shape[0] == 1381


def test_bout_keys_inherit_fish_partition(ds):
    fish = ds.get_fish(0)

    keys = list(
        ds.iter_bout_keys(
            fish_ids=[fish.canonical_fish_id]
        )
    )

    assert len(keys) == fish.valid_bout_count
    assert all(key.partition == fish.partition for key in keys)
    assert all(key.fish_id == fish.canonical_fish_id for key in keys)


def test_partition_assertion_passes_for_correct_partition(ds):
    ds.assert_fish_partition(
        "DS005-JM-F000",
        "validation",
    )


def test_partition_assertion_fails_for_wrong_partition(ds):
    with pytest.raises(RuntimeError):
        ds.assert_fish_partition(
            "DS005-JM-F000",
            "train",
        )


def test_summary_matches_frozen_dataset(ds):
    summary = ds.summary()

    assert summary["dataset_id"] == "DS-005"
    assert summary["n_fish"] == 463
    assert summary["n_valid_bouts"] == 1_203_409
    assert summary["n_contexts"] == 14
    assert summary["frame_rate_hz"] == pytest.approx(700.0)

    assert summary["partition_counts"] == {
        "train": 323,
        "validation": 70,
        "test": 70,
    }

    assert summary["split_sha256"] == FROZEN_SPLIT_SHA256
