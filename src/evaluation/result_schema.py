"""Result schemas for preregistered zebrafish evaluation.

These dataclasses define stable result shapes before final TEST evaluation.
They do not perform analysis and do not load any dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class MetricSummary:
    values: list[float]
    median: float
    q25: float
    q75: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class BootstrapMetric:
    point_estimate: float
    median: float
    q25: float
    q75: float
    ci_low: float
    ci_high: float
    n_replicates: int


@dataclass(frozen=True)
class ClassificationMetric:
    balanced_accuracy: float
    macro_f1: float
    n_classes: int | None = None
    uniform_chance: float | None = None
    chance_ratio: float | None = None
    converged: bool | None = None


@dataclass(frozen=True)
class RegressionMetric:
    r2: float
    mae: float


@dataclass(frozen=True)
class PartitionAgreement:
    ari: float
    nmi: float


@dataclass
class ReproducibilityResults:
    fish_bootstrap_ari: BootstrapMetric | None = None
    cross_seed_ari: MetricSummary | None = None
    heldout_cluster_occupancy: list[dict[str, Any]] = field(
        default_factory=list
    )


@dataclass
class BaselineVsSSLResults:
    agreement_by_seed: dict[str, PartitionAgreement] = field(
        default_factory=dict
    )
    input_a_predicts_ssl_by_seed: dict[str, ClassificationMetric] = field(
        default_factory=dict
    )


@dataclass
class SpeedControlResults:
    speed_only_vs_ssl_by_seed: dict[str, PartitionAgreement] = field(
        default_factory=dict
    )
    embedding_to_speed_by_seed: dict[str, RegressionMetric] = field(
        default_factory=dict
    )
    cluster_speed_descriptives: dict[str, Any] = field(default_factory=dict)


@dataclass
class NuisanceResults:
    fish_identity_input_a: ClassificationMetric | None = None
    fish_identity_ssl_by_seed: dict[str, ClassificationMetric] = field(
        default_factory=dict
    )
    context_input_a: ClassificationMetric | None = None
    context_ssl_by_seed: dict[str, ClassificationMetric] = field(
        default_factory=dict
    )
    session_estimable_independently: bool | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class TrackingQCResults:
    proxy_rates: dict[str, float] = field(default_factory=dict)
    cluster_enrichment: dict[str, Any] = field(default_factory=dict)
    new_post_clustering_exclusions_applied: bool = False


@dataclass
class EvaluationResult:
    dataset_id: str
    partition: str
    analysis_status: str
    test_accessed: bool
    reproducibility: ReproducibilityResults = field(
        default_factory=ReproducibilityResults
    )
    baseline_vs_ssl: BaselineVsSSLResults = field(
        default_factory=BaselineVsSSLResults
    )
    speed_controls: SpeedControlResults = field(
        default_factory=SpeedControlResults
    )
    nuisance: NuisanceResults = field(default_factory=NuisanceResults)
    tracking_qc: TrackingQCResults = field(default_factory=TrackingQCResults)
    claim_gate_results: dict[str, bool | None] = field(default_factory=dict)
    outcome_category: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION

    def validate(self) -> None:
        partition = self.partition.lower()
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid partition: {self.partition}")

        if partition == "test" and not self.test_accessed:
            raise ValueError("TEST result must set test_accessed=True.")

        if partition != "test" and self.test_accessed:
            raise ValueError(
                "test_accessed=True is inconsistent with a non-TEST result."
            )

        if self.tracking_qc.new_post_clustering_exclusions_applied:
            raise ValueError(
                "Frozen protocol prohibits new post-clustering exclusions."
            )

        valid_outcomes = {
            None,
            "SUPPORTED",
            "PARTIALLY_SUPPORTED",
            "NOT_SUPPORTED_EQUIVALENT",
            "NOT_SUPPORTED_NUISANCE",
            "NOT_SUPPORTED_UNSTABLE",
            "NOT_SUPPORTED_REPLICATION_FAILURE",
        }
        if self.outcome_category not in valid_outcomes:
            raise ValueError(
                f"Unknown outcome category: {self.outcome_category}"
            )


def to_dict(result: EvaluationResult) -> dict[str, Any]:
    result.validate()
    return asdict(result)


def write_result_json(
    result: EvaluationResult,
    path: str | Path,
) -> Path:
    result.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def empty_validation_result(
    dataset_id: str = "DS-005",
) -> EvaluationResult:
    """Create an empty safe template for pre-TEST implementation work."""
    return EvaluationResult(
        dataset_id=dataset_id,
        partition="validation",
        analysis_status="template",
        test_accessed=False,
    )
