"""Experiment registry utilities for Zebrafish Behavior SSL.

The registry is designed to record experiment provenance without requiring
access to held-out TEST data.

Typical uses:
- register SSL runs and frozen baseline runs
- record seed/checkpoint/config/artifact hashes
- record warnings and completion status
- make explicit whether TEST was accessed

The registry is append-oriented. Existing records are not silently replaced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REGISTRY_SCHEMA_VERSION = "1.0"


@dataclass
class ExperimentRecord:
    experiment_id: str
    experiment_type: str
    dataset_id: str
    partition_scope: list[str]
    status: str = "planned"
    seed: int | None = None
    config_path: str | None = None
    config_sha256: str | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    test_accessed: bool = False
    notes: str | None = None
    git_commit: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = REGISTRY_SCHEMA_VERSION

    def validate(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id cannot be empty.")
        if not self.experiment_type.strip():
            raise ValueError("experiment_type cannot be empty.")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty.")
        if not self.partition_scope:
            raise ValueError("partition_scope cannot be empty.")

        allowed = {"train", "validation", "test"}
        normalized = {str(x).lower() for x in self.partition_scope}
        unknown = normalized - allowed
        if unknown:
            raise ValueError(f"Unknown partition(s): {sorted(unknown)}")

        if "test" in normalized and not self.test_accessed:
            raise ValueError(
                "partition_scope includes TEST but test_accessed=False."
            )

        if self.test_accessed and "test" not in normalized:
            raise ValueError(
                "test_accessed=True requires 'test' in partition_scope."
            )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 for a local file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing_files(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash each supplied file and return {path: sha256}."""
    result: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = sha256_file(path)
    return result


def make_experiment_id(
    experiment_type: str,
    dataset_id: str,
    *,
    seed: int | None = None,
    suffix: str | None = None,
) -> str:
    """Construct a readable deterministic-style experiment identifier."""
    parts = [
        dataset_id.lower().replace("-", ""),
        experiment_type.lower().replace(" ", "_"),
    ]
    if seed is not None:
        parts.append(f"seed{seed}")
    if suffix:
        parts.append(suffix)
    return "__".join(parts)


class ExperimentRegistry:
    """JSONL-backed append-only experiment registry."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on registry line {line_number}."
                    ) from exc
        return rows

    def experiment_ids(self) -> set[str]:
        return {str(row["experiment_id"]) for row in self.records()}

    def append(
        self,
        record: ExperimentRecord,
        *,
        allow_duplicate_id: bool = False,
    ) -> None:
        record.validate()

        if not allow_duplicate_id and record.experiment_id in self.experiment_ids():
            raise ValueError(
                f"Experiment ID already exists: {record.experiment_id}"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), sort_keys=True))
            f.write("\n")

    def get(self, experiment_id: str) -> dict[str, Any]:
        matches = [
            row
            for row in self.records()
            if row.get("experiment_id") == experiment_id
        ]
        if not matches:
            raise KeyError(experiment_id)
        if len(matches) > 1:
            raise ValueError(
                f"Registry contains duplicate experiment ID: {experiment_id}"
            )
        return matches[0]


def ssl_run_template(
    *,
    seed: int,
    dataset_id: str = "DS-005",
) -> ExperimentRecord:
    """Return a TRAIN/VALIDATION-only template for a frozen SSL run."""
    return ExperimentRecord(
        experiment_id=make_experiment_id(
            "ssl_training",
            dataset_id,
            seed=seed,
        ),
        experiment_type="ssl_training",
        dataset_id=dataset_id,
        partition_scope=["train", "validation"],
        seed=seed,
        status="planned",
        test_accessed=False,
    )
