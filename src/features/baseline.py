"""Hand-engineered baseline features for frozen DS-005.

This module implements Input A for the zebrafish-behavior-ssl project.

The confirmatory baseline is intentionally conservative:
- one feature vector per valid behavioral bout;
- the same frozen fish-level train/validation/test split as DS-005;
- normalization statistics are fit on TRAIN fish only;
- no labels or context identities are used to construct features;
- raw head-position path/jump features are excluded from the default
  confirmatory feature set because DS-005 head_pos coordinate semantics
  showed discontinuities during preregistered QC review.

The default feature vector uses only:
- bout timing,
- speed magnitude,
- speed-change/acceleration summaries,
- wrapped orientation-change summaries,
- inter-bout interval.

An optional ``extended`` profile adds head-position-derived displacement
and path features for exploratory/sensitivity analysis. It should not
replace the frozen core baseline without a recorded protocol amendment.

Expected companion module:
    src/data/ds005.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, Union
import csv
import json

import numpy as np

try:
    # Preferred when src is configured as a package.
    from src.data.ds005 import DS005, FishRecord
except ImportError:
    # Supports direct execution from the repository.
    import sys

    _THIS_FILE = Path(__file__).resolve()
    _SRC_DATA = _THIS_FILE.parents[1] / "data"
    if str(_SRC_DATA) not in sys.path:
        sys.path.insert(0, str(_SRC_DATA))

    from ds005 import DS005, FishRecord  # type: ignore


Partition = Literal["train", "validation", "test"]
FeatureProfile = Literal["core", "extended"]


# ---------------------------------------------------------------------------
# Frozen baseline feature definitions
# ---------------------------------------------------------------------------

CORE_FEATURE_NAMES: Tuple[str, ...] = (
    # Timing
    "bout_duration_s",
    "inter_bout_interval_s",

    # Speed
    "speed_mean",
    "speed_std",
    "speed_median",
    "speed_max",
    "speed_p95",
    "speed_rms",

    # Speed change / acceleration
    "accel_abs_mean",
    "accel_abs_std",
    "accel_abs_max",
    "accel_rms",

    # Orientation / turning
    "turn_abs_total_rad",
    "turn_net_rad",
    "turn_abs_mean_rad",
    "turn_abs_std_rad",
    "turn_abs_max_rad",
    "turn_rms_rad",
)

EXTENDED_HEAD_POSITION_FEATURE_NAMES: Tuple[str, ...] = (
    "head_net_displacement",
    "head_path_length",
    "head_mean_step",
    "head_max_step",
)

EXTENDED_FEATURE_NAMES: Tuple[str, ...] = (
    CORE_FEATURE_NAMES + EXTENDED_HEAD_POSITION_FEATURE_NAMES
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureRowMeta:
    """Metadata attached to one baseline feature row."""

    dataset_id: str
    fish_id: str
    session_id: str
    fish_index: int
    bout_index: int
    partition: Partition
    context_id: str
    context_name: str
    stimulus_code: float
    bout_type: float
    all_zero_speed: bool
    extreme_speed_gt_100: bool


@dataclass
class FeatureMatrix:
    """Baseline feature matrix plus stable row metadata."""

    X: np.ndarray
    feature_names: Tuple[str, ...]
    metadata: List[FeatureRowMeta]
    profile: FeatureProfile

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                "X column count does not match feature_names."
            )
        if self.X.shape[0] != len(self.metadata):
            raise ValueError(
                "X row count does not match metadata."
            )

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def partitions(self) -> np.ndarray:
        return np.asarray(
            [m.partition for m in self.metadata],
            dtype=object,
        )

    def fish_ids(self) -> np.ndarray:
        return np.asarray(
            [m.fish_id for m in self.metadata],
            dtype=object,
        )

    def contexts(self) -> np.ndarray:
        return np.asarray(
            [m.context_name for m in self.metadata],
            dtype=object,
        )

    def subset_partition(self, partition: Partition) -> "FeatureMatrix":
        mask = self.partitions() == partition
        idx = np.flatnonzero(mask)
        return FeatureMatrix(
            X=self.X[idx],
            feature_names=self.feature_names,
            metadata=[self.metadata[i] for i in idx],
            profile=self.profile,
        )


@dataclass
class TrainOnlyStandardScaler:
    """Simple z-score scaler fit only on training rows.

    Features with zero or non-finite training standard deviation are given
    scale 1.0, preventing division by zero while leaving them centered.
    """

    feature_names: Tuple[str, ...]
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    fitted_on_partition: Optional[str] = None

    def fit(
        self,
        X: np.ndarray,
        *,
        partition: str = "train",
    ) -> "TrainOnlyStandardScaler":
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                "Feature count does not match scaler feature_names."
            )
        if X.shape[0] == 0:
            raise ValueError("Cannot fit scaler on zero rows.")
        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Training matrix contains non-finite values."
            )
        if partition != "train":
            raise ValueError(
                "Frozen protocol requires scaler fitting on train only."
            )

        self.mean_ = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=0)

        bad = (~np.isfinite(scale)) | (scale == 0.0)
        scale[bad] = 1.0

        self.scale_ = scale
        self.fitted_on_partition = "train"
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                "Feature count does not match fitted scaler."
            )

        return (X - self.mean_) / self.scale_

    def fit_transform(
        self,
        X: np.ndarray,
        *,
        partition: str = "train",
    ) -> np.ndarray:
        return self.fit(X, partition=partition).transform(X)

    def to_dict(self) -> Dict[str, object]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit.")

        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "fitted_on_partition": self.fitted_on_partition,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "TrainOnlyStandardScaler":
        scaler = cls(
            feature_names=tuple(payload["feature_names"])  # type: ignore[arg-type]
        )
        scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        scaler.fitted_on_partition = str(
            payload["fitted_on_partition"]
        )
        return scaler

    @classmethod
    def load_json(
        cls,
        path: Union[str, Path],
    ) -> "TrainOnlyStandardScaler":
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _wrapped_angle_diff(x: np.ndarray) -> np.ndarray:
    """Wrapped consecutive angular differences in [-pi, pi]."""
    d = np.diff(x, axis=-1)
    return np.arctan2(np.sin(d), np.cos(d))


def _safe_percentile_rows(
    x: np.ndarray,
    q: float,
) -> np.ndarray:
    """Row-wise percentile with compatibility across NumPy versions."""
    return np.percentile(x, q, axis=1)


def _validate_finite_feature_block(
    X: np.ndarray,
    *,
    fish_id: str,
) -> None:
    if not np.all(np.isfinite(X)):
        bad_rows, bad_cols = np.where(~np.isfinite(X))
        sample = list(
            zip(
                bad_rows[:10].tolist(),
                bad_cols[:10].tolist(),
            )
        )
        raise RuntimeError(
            f"Non-finite engineered features for {fish_id}. "
            f"First bad row/column pairs: {sample}"
        )


# ---------------------------------------------------------------------------
# Per-fish feature extraction
# ---------------------------------------------------------------------------

def extract_fish_features(
    ds: DS005,
    fish: Union[str, int, FishRecord],
    *,
    profile: FeatureProfile = "core",
) -> FeatureMatrix:
    """Extract one feature vector per valid bout for one fish.

    Parameters
    ----------
    ds:
        Open DS005 loader.
    fish:
        FishRecord, canonical fish ID, or source fish index.
    profile:
        ``"core"`` for the confirmatory baseline.
        ``"extended"`` additionally includes head_pos-derived features.

    Returns
    -------
    FeatureMatrix
        One row per valid bout for the selected fish.
    """
    if profile not in {"core", "extended"}:
        raise ValueError(
            f"Unknown feature profile: {profile!r}"
        )

    # Accept canonical FishRecord objects regardless of whether they were
    # imported as ``ds005.FishRecord`` or ``src.data.ds005.FishRecord``.
    # Python treats those import paths as distinct class identities even
    # when they refer to the same source file, so strict isinstance checks
    # are brittle in tests and direct-script execution.
    if (
        hasattr(fish, "canonical_fish_id")
        and hasattr(fish, "source_fish_index")
        and hasattr(fish, "valid_bout_count")
        and hasattr(fish, "partition")
        and hasattr(fish, "context_id")
        and hasattr(fish, "context_name")
        and hasattr(fish, "canonical_session_id")
    ):
        rec = fish
    else:
        rec = ds.get_fish(fish)

    i = rec.source_fish_index
    n = rec.valid_bout_count
    hz = float(ds.frame_rate_hz)

    # Load one fish at a time to avoid millions of small HDF5 reads while
    # keeping memory use manageable.
    speed = np.asarray(
        ds.h5["speed_head"][i, :n],
        dtype=np.float64,
    )
    orient = np.asarray(
        ds.h5["orientation_smooth"][i, :n],
        dtype=np.float64,
    )
    times = np.asarray(
        ds.h5["times_bouts"][i, :n],
        dtype=np.float64,
    )
    stimulus = np.asarray(
        ds.h5["stims"][i, :n],
        dtype=np.float64,
    )
    bout_types = np.asarray(
        ds.h5["bout_types"][i, :n],
        dtype=np.float64,
    )

    if speed.shape != (n, 175):
        raise RuntimeError(
            f"Unexpected speed shape for {rec.canonical_fish_id}: "
            f"{speed.shape}"
        )
    if orient.shape != (n, 175):
        raise RuntimeError(
            f"Unexpected orientation shape for "
            f"{rec.canonical_fish_id}: {orient.shape}"
        )
    if times.shape != (n, 2):
        raise RuntimeError(
            f"Unexpected times shape for {rec.canonical_fish_id}: "
            f"{times.shape}"
        )

    # Frozen primary structural QC excludes non-finite valid bouts. The
    # dataset audit found none, but keep the invariant explicit.
    if not (
        np.all(np.isfinite(speed))
        and np.all(np.isfinite(orient))
        and np.all(np.isfinite(times))
    ):
        raise RuntimeError(
            f"Non-finite valid source data detected for "
            f"{rec.canonical_fish_id}."
        )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    bout_duration_s = (times[:, 1] - times[:, 0]) / hz

    inter_bout_interval_s = np.zeros(n, dtype=np.float64)
    if n > 1:
        intervals = (times[1:, 0] - times[:-1, 1]) / hz

        # An interval below zero would imply overlapping bouts. Retain the
        # actual value rather than silently clipping it.
        inter_bout_interval_s[1:] = intervals

    # First bout has no preceding bout. Use 0 by frozen representation
    # convention and expose this decision in FEATURE_NOTES below.

    # ------------------------------------------------------------------
    # Speed summaries
    # ------------------------------------------------------------------

    abs_speed = np.abs(speed)

    speed_mean = np.mean(abs_speed, axis=1)
    speed_std = np.std(abs_speed, axis=1)
    speed_median = np.median(abs_speed, axis=1)
    speed_max = np.max(abs_speed, axis=1)
    speed_p95 = _safe_percentile_rows(abs_speed, 95.0)
    speed_rms = np.sqrt(np.mean(np.square(abs_speed), axis=1))

    # First difference in speed magnitude. Multiply by sampling rate so the
    # feature represents change per second in source units.
    accel = np.diff(abs_speed, axis=1) * hz
    abs_accel = np.abs(accel)

    accel_abs_mean = np.mean(abs_accel, axis=1)
    accel_abs_std = np.std(abs_accel, axis=1)
    accel_abs_max = np.max(abs_accel, axis=1)
    accel_rms = np.sqrt(np.mean(np.square(accel), axis=1))

    # ------------------------------------------------------------------
    # Orientation / turning summaries
    # ------------------------------------------------------------------

    dtheta = _wrapped_angle_diff(orient)
    abs_dtheta = np.abs(dtheta)

    turn_abs_total_rad = np.sum(abs_dtheta, axis=1)
    turn_net_rad = np.sum(dtheta, axis=1)
    turn_abs_mean_rad = np.mean(abs_dtheta, axis=1)
    turn_abs_std_rad = np.std(abs_dtheta, axis=1)
    turn_abs_max_rad = np.max(abs_dtheta, axis=1)
    turn_rms_rad = np.sqrt(
        np.mean(np.square(dtheta), axis=1)
    )

    columns: List[np.ndarray] = [
        bout_duration_s,
        inter_bout_interval_s,
        speed_mean,
        speed_std,
        speed_median,
        speed_max,
        speed_p95,
        speed_rms,
        accel_abs_mean,
        accel_abs_std,
        accel_abs_max,
        accel_rms,
        turn_abs_total_rad,
        turn_net_rad,
        turn_abs_mean_rad,
        turn_abs_std_rad,
        turn_abs_max_rad,
        turn_rms_rad,
    ]

    feature_names = CORE_FEATURE_NAMES

    # ------------------------------------------------------------------
    # Optional exploratory head-position features
    # ------------------------------------------------------------------

    if profile == "extended":
        head = np.asarray(
            ds.h5["head_pos"][i, :n],
            dtype=np.float64,
        )

        if head.shape != (n, 175, 2):
            raise RuntimeError(
                f"Unexpected head_pos shape for "
                f"{rec.canonical_fish_id}: {head.shape}"
            )
        if not np.all(np.isfinite(head)):
            raise RuntimeError(
                f"Non-finite head_pos for {rec.canonical_fish_id}."
            )

        net = head[:, -1, :] - head[:, 0, :]
        head_net_displacement = np.sqrt(
            np.sum(np.square(net), axis=1)
        )

        steps = np.diff(head, axis=1)
        step_norm = np.sqrt(
            np.sum(np.square(steps), axis=2)
        )

        head_path_length = np.sum(step_norm, axis=1)
        head_mean_step = np.mean(step_norm, axis=1)
        head_max_step = np.max(step_norm, axis=1)

        columns.extend(
            [
                head_net_displacement,
                head_path_length,
                head_mean_step,
                head_max_step,
            ]
        )
        feature_names = EXTENDED_FEATURE_NAMES

    X = np.column_stack(columns).astype(
        np.float32,
        copy=False,
    )

    _validate_finite_feature_block(
        X,
        fish_id=rec.canonical_fish_id,
    )

    all_zero_speed = np.all(speed == 0.0, axis=1)
    extreme_speed = speed_max > 100.0

    metadata = [
        FeatureRowMeta(
            dataset_id=rec.dataset_id,
            fish_id=rec.canonical_fish_id,
            session_id=rec.canonical_session_id,
            fish_index=i,
            bout_index=j,
            partition=rec.partition,
            context_id=rec.context_id,
            context_name=rec.context_name,
            stimulus_code=float(stimulus[j]),
            bout_type=float(bout_types[j]),
            all_zero_speed=bool(all_zero_speed[j]),
            extreme_speed_gt_100=bool(extreme_speed[j]),
        )
        for j in range(n)
    ]

    return FeatureMatrix(
        X=X,
        feature_names=feature_names,
        metadata=metadata,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Multi-fish / partition extraction
# ---------------------------------------------------------------------------

def extract_features(
    ds: DS005,
    *,
    partition: Optional[Partition] = None,
    fish_ids: Optional[Sequence[str]] = None,
    profile: FeatureProfile = "core",
    max_fish: Optional[int] = None,
) -> FeatureMatrix:
    """Extract features for a collection of fish.

    This function processes one fish at a time and concatenates results.

    Parameters
    ----------
    partition:
        Optional frozen partition filter.
    fish_ids:
        Optional explicit list of canonical fish IDs.
    profile:
        ``core`` or ``extended``.
    max_fish:
        Optional debug/pilot limit. Do not use for final confirmatory
        analyses unless explicitly recorded.

    Notes
    -----
    Context labels are attached only as metadata. They are never used to
    construct the feature vectors.
    """
    if partition is not None:
        fish_records = list(
            ds.fish_in_partition(partition)
        )
    else:
        fish_records = list(ds.fish_records)

    if fish_ids is not None:
        allowed = set(fish_ids)
        fish_records = [
            rec for rec in fish_records
            if rec.canonical_fish_id in allowed
        ]

        missing = allowed - {
            rec.canonical_fish_id for rec in fish_records
        }
        if missing:
            raise KeyError(
                f"Requested fish IDs not available under current "
                f"filters: {sorted(missing)}"
            )

    if max_fish is not None:
        if max_fish <= 0:
            raise ValueError("max_fish must be positive.")
        fish_records = fish_records[:max_fish]

    matrices: List[np.ndarray] = []
    metadata: List[FeatureRowMeta] = []
    names: Optional[Tuple[str, ...]] = None

    for rec in fish_records:
        block = extract_fish_features(
            ds,
            rec,
            profile=profile,
        )

        if names is None:
            names = block.feature_names
        elif names != block.feature_names:
            raise RuntimeError(
                "Feature schema changed across fish."
            )

        matrices.append(block.X)
        metadata.extend(block.metadata)

    if names is None:
        names = (
            CORE_FEATURE_NAMES
            if profile == "core"
            else EXTENDED_FEATURE_NAMES
        )

    if matrices:
        X = np.concatenate(matrices, axis=0)
    else:
        X = np.empty(
            (0, len(names)),
            dtype=np.float32,
        )

    return FeatureMatrix(
        X=X,
        feature_names=names,
        metadata=metadata,
        profile=profile,
    )


def extract_partition_features(
    ds: DS005,
    partition: Partition,
    *,
    profile: FeatureProfile = "core",
) -> FeatureMatrix:
    """Convenience wrapper for one frozen partition."""
    return extract_features(
        ds,
        partition=partition,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Train-only normalization
# ---------------------------------------------------------------------------

def fit_train_scaler(
    train_features: FeatureMatrix,
) -> TrainOnlyStandardScaler:
    """Fit z-score normalization using TRAIN rows only."""
    partitions = set(train_features.partitions().tolist())

    if partitions != {"train"}:
        raise ValueError(
            "fit_train_scaler requires a FeatureMatrix containing "
            f"only train rows; observed partitions={partitions}"
        )

    scaler = TrainOnlyStandardScaler(
        feature_names=train_features.feature_names
    )
    scaler.fit(
        train_features.X,
        partition="train",
    )
    return scaler


def transform_feature_matrix(
    features: FeatureMatrix,
    scaler: TrainOnlyStandardScaler,
) -> FeatureMatrix:
    """Apply a previously train-fitted scaler."""
    if tuple(features.feature_names) != tuple(
        scaler.feature_names
    ):
        raise ValueError(
            "Feature schema does not match fitted scaler."
        )

    X_scaled = scaler.transform(features.X).astype(
        np.float32,
        copy=False,
    )

    return FeatureMatrix(
        X=X_scaled,
        feature_names=features.feature_names,
        metadata=list(features.metadata),
        profile=features.profile,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_feature_matrix_npz(
    features: FeatureMatrix,
    path: Union[str, Path],
) -> None:
    """Save numeric features and row metadata to compressed NPZ.

    Metadata are stored as ordinary string/numeric arrays, avoiding Python
    pickle objects so the file can be loaded with ``allow_pickle=False``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = features.metadata

    np.savez_compressed(
        path,
        X=features.X,
        feature_names=np.asarray(
            features.feature_names,
            dtype="U64",
        ),
        profile=np.asarray(features.profile),
        dataset_id=np.asarray(
            [m.dataset_id for m in meta],
            dtype="U16",
        ),
        fish_id=np.asarray(
            [m.fish_id for m in meta],
            dtype="U32",
        ),
        session_id=np.asarray(
            [m.session_id for m in meta],
            dtype="U32",
        ),
        fish_index=np.asarray(
            [m.fish_index for m in meta],
            dtype=np.int32,
        ),
        bout_index=np.asarray(
            [m.bout_index for m in meta],
            dtype=np.int32,
        ),
        partition=np.asarray(
            [m.partition for m in meta],
            dtype="U16",
        ),
        context_id=np.asarray(
            [m.context_id for m in meta],
            dtype="U16",
        ),
        context_name=np.asarray(
            [m.context_name for m in meta],
            dtype="U64",
        ),
        stimulus_code=np.asarray(
            [m.stimulus_code for m in meta],
            dtype=np.float32,
        ),
        bout_type=np.asarray(
            [m.bout_type for m in meta],
            dtype=np.float32,
        ),
        all_zero_speed=np.asarray(
            [m.all_zero_speed for m in meta],
            dtype=bool,
        ),
        extreme_speed_gt_100=np.asarray(
            [m.extreme_speed_gt_100 for m in meta],
            dtype=bool,
        ),
    )


def save_feature_schema(
    path: Union[str, Path],
    *,
    profile: FeatureProfile = "core",
) -> None:
    """Write a human-readable JSON schema for the baseline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = (
        CORE_FEATURE_NAMES
        if profile == "core"
        else EXTENDED_FEATURE_NAMES
    )

    payload = {
        "dataset_id": "DS-005",
        "input": "Input A — hand-engineered baseline",
        "profile": profile,
        "feature_names": list(names),
        "split_unit": "fish",
        "normalization": (
            "z-score using training-partition bouts only; "
            "validation/test use frozen training statistics"
        ),
        "context_labels_used_as_features": False,
        "stimulus_codes_used_as_features": False,
        "bout_type_used_as_feature": False,
        "qc": {
            "primary_nonfinite_exclusion": True,
            "all_zero_speed_primary_exclusion": False,
            "speed_gt_100_primary_exclusion": False,
            "all_zero_speed_sensitivity_flag": True,
            "speed_gt_100_sensitivity_flag": True,
        },
        "head_position_note": (
            "Raw head_pos-derived path features are excluded from the "
            "core confirmatory profile because coordinate discontinuities "
            "were observed during QC and source coordinate semantics are "
            "not fully resolved."
        ),
        "first_inter_bout_interval_rule": (
            "First valid bout for each fish receives "
            "inter_bout_interval_s = 0 because no preceding bout exists."
        ),
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Auditing helpers
# ---------------------------------------------------------------------------

def audit_feature_matrix(
    features: FeatureMatrix,
) -> Dict[str, object]:
    """Return a compact quality/integrity audit."""
    X = features.X

    partition_counts: Dict[str, int] = {}
    fish_by_partition: Dict[str, set] = {}

    for m in features.metadata:
        partition_counts[m.partition] = (
            partition_counts.get(m.partition, 0) + 1
        )
        fish_by_partition.setdefault(
            m.partition,
            set(),
        ).add(m.fish_id)

    fish_overlap = False
    parts = list(fish_by_partition)
    for a_idx in range(len(parts)):
        for b_idx in range(a_idx + 1, len(parts)):
            if (
                fish_by_partition[parts[a_idx]]
                & fish_by_partition[parts[b_idx]]
            ):
                fish_overlap = True

    return {
        "profile": features.profile,
        "rows": features.n_rows,
        "features": features.n_features,
        "feature_names": list(features.feature_names),
        "finite": bool(np.all(np.isfinite(X))),
        "fish_count": len(
            {m.fish_id for m in features.metadata}
        ),
        "partition_row_counts": partition_counts,
        "fish_overlap_across_partitions": fish_overlap,
        "all_zero_speed_rows": sum(
            m.all_zero_speed for m in features.metadata
        ),
        "extreme_speed_gt_100_rows": sum(
            m.extreme_speed_gt_100
            for m in features.metadata
        ),
    }


# ---------------------------------------------------------------------------
# Command-line smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a lightweight baseline smoke test on the first train fish."""
    with DS005() as ds:
        train_fish = ds.fish_in_partition("train")[0]

        features = extract_fish_features(
            ds,
            train_fish,
            profile="core",
        )

        print("DS-005 BASELINE FEATURE SMOKE TEST")
        print("==================================")
        print(f"Fish: {train_fish.canonical_fish_id}")
        print(f"Partition: {train_fish.partition}")
        print(f"Context: {train_fish.context_name}")
        print(f"Rows: {features.n_rows}")
        print(f"Features: {features.n_features}")
        print()
        print("Feature names:")
        for name in features.feature_names:
            print(f"  - {name}")

        print()
        print("First feature vector:")
        for name, value in zip(
            features.feature_names,
            features.X[0],
        ):
            print(f"  {name:28s} {float(value):.6g}")

        print()
        print("Audit:")
        audit = audit_feature_matrix(features)
        for key, value in audit.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
