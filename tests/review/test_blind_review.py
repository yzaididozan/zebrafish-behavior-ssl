import json

import pytest

from src.review.blind_review import (
    blind_sequence,
    invert_blind_map,
    make_blind_map,
    make_cluster_blind_map,
    make_representation_blind_map,
    write_blind_key,
    write_blinded_manifest,
)


def test_make_blind_map_is_deterministic():
    labels = ["INPUT_A", "SSL_SEED_11", "SSL_SEED_23"]

    a = make_blind_map(labels, seed=123)
    b = make_blind_map(labels, seed=123)

    assert a == b
    assert set(a.values()) == {"R01", "R02", "R03"}


def test_invert_blind_map():
    mapping = {"A": "R01", "B": "R02"}
    inverse = invert_blind_map(mapping)

    assert inverse == {"R01": "A", "R02": "B"}


def test_blind_sequence():
    mapping = {"A": "R02", "B": "R01"}

    result = blind_sequence(["A", "B", "A"], mapping)

    assert result == ["R02", "R01", "R02"]


def test_blind_sequence_rejects_unknown_label():
    with pytest.raises(KeyError):
        blind_sequence(["missing"], {"A": "R01"})


def test_representation_blind_map_contains_all_frozen_inputs():
    mapping = make_representation_blind_map()

    expected = {
        "INPUT_A",
        "SSL_SEED_11",
        "SSL_SEED_23",
        "SSL_SEED_37",
        "SSL_SEED_51",
        "SSL_SEED_79",
    }

    assert set(mapping) == expected
    assert len(set(mapping.values())) == len(expected)


def test_cluster_blind_map():
    mapping = make_cluster_blind_map([0, 1])

    assert set(mapping) == {"0", "1"}
    assert set(mapping.values()) == {"C01", "C02"}


def test_write_blind_key_refuses_overwrite(tmp_path):
    path = tmp_path / "blind_key.json"
    mapping = {"INPUT_A": "R01"}

    write_blind_key(mapping, path)

    with pytest.raises(FileExistsError):
        write_blind_key(mapping, path)


def test_write_blinded_manifest_contains_no_unblinded_ids(tmp_path):
    items = ["INPUT_A", "SSL_SEED_11"]
    mapping = make_blind_map(items)

    path = tmp_path / "manifest.json"
    write_blinded_manifest(items, mapping, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)

    assert payload["contains_unblinded_identity"] is False
    assert "INPUT_A" not in serialized
    assert "SSL_SEED_11" not in serialized
