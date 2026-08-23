from pathlib import Path

import pytest

from src.registry.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
    hash_existing_files,
    make_experiment_id,
    sha256_file,
    ssl_run_template,
)


def test_make_experiment_id():
    exp_id = make_experiment_id(
        "ssl_training",
        "DS-005",
        seed=11,
    )
    assert exp_id == "ds005__ssl_training__seed11"


def test_ssl_run_template_is_train_validation_only():
    record = ssl_run_template(seed=23)

    assert record.dataset_id == "DS-005"
    assert record.seed == 23
    assert record.partition_scope == ["train", "validation"]
    assert record.test_accessed is False

    record.validate()


def test_experiment_record_rejects_test_scope_without_flag():
    record = ExperimentRecord(
        experiment_id="x",
        experiment_type="evaluation",
        dataset_id="DS-005",
        partition_scope=["test"],
        test_accessed=False,
    )

    with pytest.raises(ValueError):
        record.validate()


def test_experiment_record_rejects_test_flag_without_scope():
    record = ExperimentRecord(
        experiment_id="x",
        experiment_type="evaluation",
        dataset_id="DS-005",
        partition_scope=["validation"],
        test_accessed=True,
    )

    with pytest.raises(ValueError):
        record.validate()


def test_registry_append_and_get(tmp_path):
    path = tmp_path / "registry.jsonl"
    registry = ExperimentRegistry(path)

    record = ssl_run_template(seed=11)
    registry.append(record)

    loaded = registry.get(record.experiment_id)

    assert loaded["experiment_id"] == record.experiment_id
    assert loaded["seed"] == 11
    assert loaded["test_accessed"] is False


def test_registry_rejects_duplicate_id(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    record = ssl_run_template(seed=11)

    registry.append(record)

    with pytest.raises(ValueError):
        registry.append(record)


def test_sha256_helpers(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("zebrafish\n", encoding="utf-8")

    digest = sha256_file(path)
    hashes = hash_existing_files([path])

    assert len(digest) == 64
    assert hashes[str(path)] == digest
