import json

import pytest

from src.evaluation.result_schema import (
    EvaluationResult,
    TrackingQCResults,
    empty_validation_result,
    to_dict,
    write_result_json,
)


def test_empty_validation_result_is_safe():
    result = empty_validation_result()

    assert result.dataset_id == "DS-005"
    assert result.partition == "validation"
    assert result.test_accessed is False

    result.validate()


def test_validation_result_rejects_test_access():
    result = EvaluationResult(
        dataset_id="DS-005",
        partition="validation",
        analysis_status="dry_run",
        test_accessed=True,
    )

    with pytest.raises(ValueError):
        result.validate()


def test_test_result_requires_test_access_flag():
    result = EvaluationResult(
        dataset_id="DS-005",
        partition="test",
        analysis_status="final",
        test_accessed=False,
    )

    with pytest.raises(ValueError):
        result.validate()


def test_rejects_post_clustering_exclusions():
    result = EvaluationResult(
        dataset_id="DS-005",
        partition="validation",
        analysis_status="dry_run",
        test_accessed=False,
        tracking_qc=TrackingQCResults(
            new_post_clustering_exclusions_applied=True
        ),
    )

    with pytest.raises(ValueError):
        result.validate()


def test_rejects_unknown_outcome_category():
    result = empty_validation_result()
    result.outcome_category = "SSL_WON"

    with pytest.raises(ValueError):
        result.validate()


def test_to_dict_and_write_json(tmp_path):
    result = empty_validation_result()

    payload = to_dict(result)
    assert payload["dataset_id"] == "DS-005"
    assert payload["partition"] == "validation"

    path = tmp_path / "result.json"
    written = write_result_json(result, path)

    assert written == path
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["test_accessed"] is False
    assert loaded["analysis_status"] == "template"
