#!/usr/bin/env python3
"""
Prepare DS-006 for external replication.

This script implements the frozen DS-006 ingestion/QC contract and writes
leakage-safe baseline and SSL artifacts without using DS-006 to alter the
primary DS-005 method.

Key replication constraints
---------------------------
- DS-006 is EXTERNAL_REPLICATION only.
- It cannot change the primary SSL method or architecture.
- It cannot select primary hyperparameters.
- It cannot select primary cluster k.
- All preprocessing statistics are fit on DS-006 TRAIN only.
- The TEST partition is written but never used to fit preprocessing.

Frozen DS-006 QC
----------------
fps_hz                  = 160
px_to_mm                = 0.071
author interpolation    = splprep(..., s=10)
interpolation noise SD  = 0.1
QC random seed          = 20260822

Frozen accepted bouts   = 163,065
Frozen rejected bouts   = 958

Replication split
-----------------
A conservative recording-level split is used so no recording appears in more
than one partition. Splitting is performed independently within each experiment
family using seed 20260822, with approximately 70/15/15 train/validation/test.

SSL temporal adaptation
-----------------------
DS-006 bouts are variable length and sampled at 160 Hz, while the frozen DS-005
encoder expects a fixed temporal length of 175.

To avoid changing the encoder architecture, each accepted DS-006 bout is
deterministically resampled over normalized bout phase to 175 samples:

    Heading -> unwrap -> linear interpolation -> sin/cos
    derived head speed -> linear interpolation

Only the speed channel is standardized. Its mean/std are fit on TRAIN
resampled temporal samples only. sin/cos remain unstandardized.

Baseline preprocessing
----------------------
The 18 frozen feature families are generated from each accepted bout.
The first accepted bout in each fish-well has undefined IBI and is stored as
NaN in *_core_raw.npz. A TRAIN-only median imputation vector and TRAIN-only
z-score parameters are then used to produce *_core_scaled.npz.

Outputs
-------
data/processed/DS-006/
    baseline/
        train_core_raw.npz
        validation_core_raw.npz
        test_core_raw.npz
        train_core_scaled.npz
        validation_core_scaled.npz
        test_core_scaled.npz
        feature_manifest.json
        normalization.json
    ssl/
        train.npz
        validation.npz
        test.npz
        normalization.json
        input_manifest.json
    metadata/
        split_assignments.csv
        bout_metadata.csv
        qc_summary.json
    feasibility/
        ds006_feasibility.json        # produced by prior audit, not overwritten

Usage
-----
Run from repository root:

    python3 src/data/prepare_ds006.py

This is intentionally a full preprocessing pass and may take several minutes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

DATASET_ID = "DS-006"

FPS = 160.0
PX_TO_MM = 0.071

QC_SEED = 20260822
SPLIT_SEED = 20260822

INTERPOLATION_NOISE_SD = 0.1
INTERPOLATION_S = 10.0

SSL_TARGET_LENGTH = 175

EXPECTED_RAW_BOUTS = 165_579
EXPECTED_WELL_EXCLUDED_BOUTS = 1_556
EXPECTED_QC_CANDIDATES = 164_023
EXPECTED_ACCEPTED_BOUTS = 163_065
EXPECTED_REJECTED_BOUTS = 958

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

FEATURE_NAMES = [
    "bout_duration",
    "inter_bout_interval",
    "speed_mean",
    "speed_std",
    "speed_median",
    "speed_max",
    "speed_p95",
    "speed_rms",
    "speed_change_abs_mean",
    "speed_change_std",
    "speed_change_max",
    "speed_change_rms",
    "turn_total_abs",
    "turn_net",
    "turn_abs_mean",
    "turn_std",
    "turn_max",
    "turn_rms",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AcceptedBout:
    recording_id: str
    family: str
    well: int
    condition_code: int
    condition_label: str
    bout_index: int
    bout_id: str
    fish_id: str
    bout: Any
    baseline: np.ndarray
    ssl_raw: np.ndarray


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare DS-006 external replication artifacts."
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/DS-006/extracted/Data_all"),
    )
    p.add_argument(
        "--condition-map",
        type=Path,
        default=Path("data/manifests/DS-006/well-condition-map.csv"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/DS-006"),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def scalar_text(x: Any) -> str:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def scalar_float(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def family_from_recording(recording_id: str) -> str:
    for fam in ("pH_1a", "pH_2a", "pH_2b", "pH_2c"):
        if fam in recording_id:
            return fam
    return "UNKNOWN"


def canonical_fish_id(recording_id: str, well: int) -> str:
    return f"DS006::{recording_id}::well{well:02d}"


def canonical_bout_id(recording_id: str, well: int, bout_index: int) -> str:
    return (
        f"DS006::{recording_id}::well{well:02d}"
        f"::bout{bout_index:06d}"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Condition map
# ---------------------------------------------------------------------------

def load_condition_map(
    path: Path,
) -> dict[tuple[str, int], tuple[int, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Condition map not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Condition map is empty.")

    required = {"recording_id", "well", "condition_code", "condition_label"}
    missing = required - set(rows[0].keys())
    if missing:
        raise RuntimeError(
            f"Condition map missing columns: {sorted(missing)}"
        )

    mapping: dict[tuple[str, int], tuple[int, str]] = {}

    for r in rows:
        key = (r["recording_id"], int(r["well"]))
        mapping[key] = (int(r["condition_code"]), r["condition_label"])

    return mapping


# ---------------------------------------------------------------------------
# MAT loading
# ---------------------------------------------------------------------------

def list_mat_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("results_*.mat"))
    if len(files) != 32:
        raise RuntimeError(
            f"Expected 32 MAT files, found {len(files)} under {root}"
        )
    return files


def load_recording(path: Path) -> tuple[str, list[np.ndarray]]:
    obj = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )["videoDataResults"]

    recording_id = scalar_text(obj.organization.videoName)
    wells = [
        np.atleast_1d(w)
        for w in np.atleast_1d(obj.wellPoissMouv)
    ]

    if len(wells) != 12:
        raise RuntimeError(
            f"{recording_id}: expected 12 wells, found {len(wells)}"
        )

    return recording_id, wells


# ---------------------------------------------------------------------------
# Frozen author-style deterministic QC
# ---------------------------------------------------------------------------

def author_interp_head(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    noisy_x = x + INTERPOLATION_NOISE_SD * rng.randn(len(x))
    noisy_y = y + INTERPOLATION_NOISE_SD * rng.randn(len(y))

    tck, u = splprep(
        [noisy_x, noisy_y],
        s=INTERPOLATION_S,
    )
    points = splev(u, tck)

    return np.asarray(points[0]), np.asarray(points[1])


def author_distance_mm(
    bout: Any,
    rng: np.random.RandomState,
) -> float:
    x = as_1d(bout.HeadX)
    y = as_1d(bout.HeadY)

    xi, yi = author_interp_head(x, y, rng)

    if len(xi) < 2:
        return 0.0

    dx = np.diff(xi)
    dy = np.diff(yi)

    return float(
        np.sum(np.sqrt(dx * dx + dy * dy)) * PX_TO_MM
    )


def author_deltahead_deg(bout: Any) -> float:
    numps = 6

    tx = np.asarray(bout.TailX_VideoReferential, dtype=float)
    ty = np.asarray(bout.TailY_VideoReferential, dtype=float)

    hx = as_1d(bout.HeadX)
    hy = as_1d(bout.HeadY)

    bxs = np.concatenate(
        (np.asarray(tx[0][-numps:]).reshape(-1), [hx[0]])
    )
    bys = np.concatenate(
        (np.asarray(ty[0][-numps:]).reshape(-1), [hy[0]])
    )

    slope0 = math.degrees(
        math.atan2(
            bys[-1] - bys[0],
            bxs[-1] - bxs[0],
        )
    )

    bxs = np.concatenate(
        (np.asarray(tx[-1][-numps:]).reshape(-1), [hx[-1]])
    )
    bys = np.concatenate(
        (np.asarray(ty[-1][-numps:]).reshape(-1), [hy[-1]])
    )

    slope1 = math.degrees(
        math.atan2(
            bys[-1] - bys[0],
            bxs[-1] - bxs[0],
        )
    )

    delt = -(slope1 - slope0)

    if delt > 180:
        return float(360 - delt)
    if delt < -180:
        return float(-(360 + delt))
    return float(delt)


def qc_bout(
    bout: Any,
    rng: np.random.RandomState,
) -> tuple[bool, list[str]]:
    x = as_1d(bout.HeadX)

    duration = len(x) / FPS

    try:
        distance = author_distance_mm(bout, rng)
    except Exception:
        return False, ["interpolation_failure"]

    speed = distance / duration if duration > 0 else float("inf")

    try:
        deltahead = author_deltahead_deg(bout)
    except Exception:
        return False, ["deltahead_failure"]

    reasons: list[str] = []

    if duration < 0.04:
        reasons.append("time_too_short")
    if duration > 1.2:
        reasons.append("time_too_long")
    if distance > 25:
        reasons.append("distance_too_large")
    if distance < 0:
        reasons.append("distance_negative")
    if speed > 50:
        reasons.append("speed_too_high")
    if speed < 1:
        reasons.append("speed_too_low")
    if abs(deltahead) > 180:
        reasons.append("heading_change_too_large")

    return not reasons, reasons


# ---------------------------------------------------------------------------
# Input A
# ---------------------------------------------------------------------------

def pointwise_speed_mm_s(bout: Any) -> np.ndarray:
    x = as_1d(bout.HeadX)
    y = as_1d(bout.HeadY)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    if n == 0:
        return np.empty(0, dtype=float)

    if n == 1:
        return np.zeros(1, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)

    step_speed = (
        np.sqrt(dx * dx + dy * dy)
        * PX_TO_MM
        * FPS
    )

    # Repeat first step so temporal length equals position series length.
    return np.concatenate(([step_speed[0]], step_speed))


def baseline_features(
    bout: Any,
    previous_accepted_bout: Any | None,
) -> np.ndarray:
    x = as_1d(bout.HeadX)
    y = as_1d(bout.HeadY)
    heading = as_1d(bout.Heading)

    n = min(len(x), len(y), len(heading))

    if n < 2:
        raise ValueError("Accepted bout has fewer than two valid samples.")

    x = x[:n]
    y = y[:n]
    heading = heading[:n]

    speed = pointwise_speed_mm_s(bout)[:n]

    if len(speed) != n:
        raise RuntimeError("Speed temporal length mismatch.")

    duration = n / FPS

    ibi = float("nan")

    if previous_accepted_bout is not None:
        try:
            start = scalar_float(bout.BoutStart)
            prev_end = scalar_float(previous_accepted_bout.BoutEnd)
            ibi = (start - prev_end) / FPS
        except Exception:
            ibi = float("nan")

    ds = np.diff(speed)

    # Heading is radians according to the verified DS-006 audit.
    unwrapped = np.unwrap(heading)
    turns = np.diff(unwrapped)

    if ds.size == 0 or turns.size == 0:
        raise ValueError("Insufficient accepted bout length for features.")

    features = np.asarray(
        [
            duration,
            ibi,
            np.mean(speed),
            np.std(speed),
            np.median(speed),
            np.max(speed),
            np.percentile(speed, 95),
            np.sqrt(np.mean(speed ** 2)),
            np.mean(np.abs(ds)),
            np.std(ds),
            np.max(np.abs(ds)),
            np.sqrt(np.mean(ds ** 2)),
            np.sum(np.abs(turns)),
            np.sum(turns),
            np.mean(np.abs(turns)),
            np.std(turns),
            np.max(np.abs(turns)),
            np.sqrt(np.mean(turns ** 2)),
        ],
        dtype=np.float64,
    )

    if len(features) != 18:
        raise AssertionError("Expected 18 baseline features.")

    # IBI is the only intentionally permitted NaN.
    if not np.all(np.isfinite(features[2:])):
        raise ValueError("Non-finite baseline feature detected.")

    return features


# ---------------------------------------------------------------------------
# Input B
# ---------------------------------------------------------------------------

def resample_1d(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(values) == target_length:
        return values.copy()

    if len(values) < 2:
        raise ValueError("Cannot resample fewer than two samples.")

    old_phase = np.linspace(0.0, 1.0, len(values), dtype=float)
    new_phase = np.linspace(0.0, 1.0, target_length, dtype=float)

    return np.interp(new_phase, old_phase, values)


def make_ssl_raw(bout: Any) -> np.ndarray:
    heading = as_1d(bout.Heading)
    speed = pointwise_speed_mm_s(bout)

    n = min(len(heading), len(speed))

    if n < 2:
        raise ValueError("Insufficient temporal samples for SSL input.")

    heading = heading[:n]
    speed = speed[:n]

    if not (
        np.all(np.isfinite(heading))
        and np.all(np.isfinite(speed))
    ):
        raise ValueError("Non-finite SSL source values.")

    # Interpolate the continuous unwrapped angle, then convert back to the
    # circular sin/cos representation. This avoids directly interpolating
    # across the +/-pi discontinuity.
    angle = np.unwrap(heading)
    angle_rs = resample_1d(angle, SSL_TARGET_LENGTH)
    speed_rs = resample_1d(speed, SSL_TARGET_LENGTH)

    out = np.column_stack(
        (
            np.sin(angle_rs),
            np.cos(angle_rs),
            speed_rs,
        )
    ).astype(np.float32)

    if out.shape != (SSL_TARGET_LENGTH, 3):
        raise AssertionError(
            f"Unexpected SSL shape: {out.shape}"
        )

    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite values in SSL representation.")

    return out


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def allocate_family_counts(n: int) -> tuple[int, int, int]:
    """
    Approximately 70/15/15 while guaranteeing at least one validation and
    one test recording for families with >= 3 recordings.
    """
    if n < 3:
        raise ValueError(
            "Recording-level split requires at least 3 recordings per family."
        )

    n_val = max(1, int(round(n * VALIDATION_FRACTION)))
    n_test = max(1, int(round(n * TEST_FRACTION)))
    n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
        remaining = n - 1
        n_val = max(1, remaining // 2)
        n_test = remaining - n_val

    return n_train, n_val, n_test


def make_recording_split(
    recordings_by_family: dict[str, list[str]],
) -> dict[str, str]:
    rng = np.random.RandomState(SPLIT_SEED)

    assignment: dict[str, str] = {}

    for fam in sorted(recordings_by_family):
        recordings = sorted(set(recordings_by_family[fam]))
        perm = list(np.asarray(recordings)[rng.permutation(len(recordings))])

        n_train, n_val, n_test = allocate_family_counts(len(perm))

        train = perm[:n_train]
        val = perm[n_train : n_train + n_val]
        test = perm[n_train + n_val :]

        assert len(test) == n_test

        for r in train:
            assignment[r] = "train"
        for r in val:
            assignment[r] = "validation"
        for r in test:
            assignment[r] = "test"

    return assignment


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_baseline_preprocessor(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TRAIN-only preprocessing:
    1. featurewise median imputation
    2. featurewise z-score
    """
    medians = np.nanmedian(X_train, axis=0)

    if not np.all(np.isfinite(medians)):
        raise RuntimeError("Non-finite TRAIN imputation medians.")

    filled = np.where(np.isnan(X_train), medians, X_train)

    means = np.mean(filled, axis=0)
    stds = np.std(filled, axis=0)

    if np.any(stds <= 0):
        bad = [
            FEATURE_NAMES[i]
            for i in np.where(stds <= 0)[0]
        ]
        raise RuntimeError(
            f"Zero-variance TRAIN baseline features: {bad}"
        )

    return medians, means, stds


def transform_baseline(
    X: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    filled = np.where(np.isnan(X), medians, X)
    scaled = (filled - means) / stds

    if not np.all(np.isfinite(scaled)):
        raise RuntimeError("Non-finite scaled baseline values.")

    return scaled.astype(np.float32)


def fit_ssl_speed_normalization(
    X_train: np.ndarray,
) -> tuple[float, float]:
    speed = X_train[:, :, 2].astype(np.float64)

    mean = float(np.mean(speed))
    std = float(np.std(speed))

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid TRAIN SSL speed normalization.")

    return mean, std


def transform_ssl(
    X: np.ndarray,
    speed_mean: float,
    speed_std: float,
) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    out[:, :, 2] = (
        out[:, :, 2] - speed_mean
    ) / speed_std

    if not np.all(np.isfinite(out)):
        raise RuntimeError("Non-finite normalized SSL values.")

    return out


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_npz(
    path: Path,
    X: np.ndarray,
    bout_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        bout_id=bout_ids,
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    condition_map = load_condition_map(args.condition_map)
    mat_files = list_mat_files(args.raw_root)

    rng_qc = np.random.RandomState(QC_SEED)

    raw_bouts = 0
    well_excluded_bouts = 0
    qc_candidates = 0
    accepted_count = 0
    rejected_count = 0
    rejection_reasons = Counter()

    accepted_bouts: list[AcceptedBout] = []
    recordings_by_family: dict[str, list[str]] = defaultdict(list)

    empty_wells: list[tuple[str, int]] = []
    bad_wells_seen: list[tuple[str, int]] = []

    print("DS-006 PREPARATION")
    print("=" * 40)
    print("Pass 1/3: loading, frozen QC, feature construction")

    for file_i, path in enumerate(mat_files, start=1):
        recording_id, wells = load_recording(path)
        family = family_from_recording(recording_id)

        recordings_by_family[family].append(recording_id)

        for well_index, well in enumerate(wells, start=1):
            key = (recording_id, well_index)

            if key not in condition_map:
                raise RuntimeError(
                    f"No condition mapping for {recording_id} well {well_index}"
                )

            condition_code, condition_label = condition_map[key]

            if well.size == 0:
                empty_wells.append(key)
                continue

            raw_bouts += len(well)

            if condition_label == "bad_data":
                bad_wells_seen.append(key)
                well_excluded_bouts += len(well)
                continue

            previous_accepted_bout: Any | None = None

            for bout_index, bout in enumerate(well):
                qc_candidates += 1

                ok, reasons = qc_bout(bout, rng_qc)

                if not ok:
                    rejected_count += 1
                    for reason in reasons:
                        rejection_reasons[reason] += 1
                    continue

                baseline = baseline_features(
                    bout,
                    previous_accepted_bout,
                )
                ssl_raw = make_ssl_raw(bout)

                fish_id = canonical_fish_id(
                    recording_id,
                    well_index,
                )
                bout_id = canonical_bout_id(
                    recording_id,
                    well_index,
                    bout_index,
                )

                accepted_bouts.append(
                    AcceptedBout(
                        recording_id=recording_id,
                        family=family,
                        well=well_index,
                        condition_code=condition_code,
                        condition_label=condition_label,
                        bout_index=bout_index,
                        bout_id=bout_id,
                        fish_id=fish_id,
                        bout=bout,
                        baseline=baseline,
                        ssl_raw=ssl_raw,
                    )
                )

                accepted_count += 1
                previous_accepted_bout = bout

        print(
            f"  [{file_i:02d}/{len(mat_files)}] "
            f"{recording_id}"
        )

    # ------------------------------------------------------------------
    # Frozen QC integrity check
    # ------------------------------------------------------------------

    observed = {
        "raw_bouts": raw_bouts,
        "well_excluded_bouts": well_excluded_bouts,
        "qc_candidates": qc_candidates,
        "accepted_bouts": accepted_count,
        "rejected_bouts": rejected_count,
    }

    expected = {
        "raw_bouts": EXPECTED_RAW_BOUTS,
        "well_excluded_bouts": EXPECTED_WELL_EXCLUDED_BOUTS,
        "qc_candidates": EXPECTED_QC_CANDIDATES,
        "accepted_bouts": EXPECTED_ACCEPTED_BOUTS,
        "rejected_bouts": EXPECTED_REJECTED_BOUTS,
    }

    if observed != expected:
        print("\nFROZEN QC MISMATCH")
        print("Observed:", observed)
        print("Expected:", expected)
        raise RuntimeError(
            "DS-006 QC no longer matches the frozen manifest. "
            "No processed artifacts will be trusted."
        )

    # ------------------------------------------------------------------
    # Recording-level split
    # ------------------------------------------------------------------

    print("\nPass 2/3: assigning recording-level partitions")

    recording_split = make_recording_split(recordings_by_family)

    split_counts_recordings = Counter(recording_split.values())

    # Verify no recording crosses partitions by construction.
    if len(recording_split) != 32:
        raise RuntimeError(
            f"Expected split assignment for 32 recordings, got "
            f"{len(recording_split)}"
        )

    for fam in sorted(recordings_by_family):
        fam_recs = sorted(set(recordings_by_family[fam]))
        counts = Counter(recording_split[r] for r in fam_recs)
        print(
            f"  {fam}: "
            f"train={counts['train']} "
            f"validation={counts['validation']} "
            f"test={counts['test']}"
        )

    # ------------------------------------------------------------------
    # Materialize arrays
    # ------------------------------------------------------------------

    print("\nPass 3/3: materializing arrays and TRAIN-only normalization")

    by_split: dict[str, list[AcceptedBout]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for rec in accepted_bouts:
        by_split[recording_split[rec.recording_id]].append(rec)

    baseline_raw: dict[str, np.ndarray] = {}
    ssl_raw: dict[str, np.ndarray] = {}
    bout_ids: dict[str, np.ndarray] = {}

    for split in ("train", "validation", "test"):
        rows = by_split[split]

        baseline_raw[split] = np.vstack(
            [r.baseline for r in rows]
        ).astype(np.float64)

        ssl_raw[split] = np.stack(
            [r.ssl_raw for r in rows]
        ).astype(np.float32)

        bout_ids[split] = np.asarray(
            [r.bout_id for r in rows],
            dtype="U256",
        )

        print(
            f"  {split:<10} "
            f"bouts={len(rows):>7,} "
            f"baseline={baseline_raw[split].shape} "
            f"ssl={ssl_raw[split].shape}"
        )

    # Fit TRAIN only.
    medians, baseline_mean, baseline_std = fit_baseline_preprocessor(
        baseline_raw["train"]
    )

    baseline_scaled = {
        split: transform_baseline(
            baseline_raw[split],
            medians,
            baseline_mean,
            baseline_std,
        )
        for split in ("train", "validation", "test")
    }

    ssl_speed_mean, ssl_speed_std = fit_ssl_speed_normalization(
        ssl_raw["train"]
    )

    ssl_scaled = {
        split: transform_ssl(
            ssl_raw[split],
            ssl_speed_mean,
            ssl_speed_std,
        )
        for split in ("train", "validation", "test")
    }

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------

    baseline_dir = args.output_root / "baseline"
    ssl_dir = args.output_root / "ssl"
    metadata_dir = args.output_root / "metadata"

    for split in ("train", "validation", "test"):
        save_npz(
            baseline_dir / f"{split}_core_raw.npz",
            baseline_raw[split],
            bout_ids[split],
        )
        save_npz(
            baseline_dir / f"{split}_core_scaled.npz",
            baseline_scaled[split],
            bout_ids[split],
        )
        save_npz(
            ssl_dir / f"{split}.npz",
            ssl_scaled[split],
            bout_ids[split],
        )

    write_json(
        baseline_dir / "feature_manifest.json",
        {
            "dataset_id": DATASET_ID,
            "feature_count": 18,
            "feature_names": FEATURE_NAMES,
            "source_mapping": {
                "bout_duration": "len(HeadX) / 160",
                "inter_bout_interval": (
                    "(BoutStart_current - BoutEnd_previous_accepted) / 160"
                ),
                "speed": (
                    "sqrt(diff(HeadX)^2 + diff(HeadY)^2) "
                    "* 0.071 * 160"
                ),
                "turn": "diff(unwrap(Heading))",
            },
            "units": {
                "bout_duration": "seconds",
                "inter_bout_interval": "seconds",
                "speed_features": "mm/s",
                "speed_change_features": "mm/s per sample",
                "turn_features": "radians",
            },
            "ibi_policy": (
                "Undefined for the first accepted bout in each usable well; "
                "stored NaN in raw arrays and median-imputed using TRAIN only "
                "for scaled arrays."
            ),
        },
    )

    write_json(
        baseline_dir / "normalization.json",
        {
            "fit_partition": "train",
            "imputation": "featurewise_train_median",
            "scaling": "featurewise_train_zscore",
            "feature_names": FEATURE_NAMES,
            "train_median": medians.tolist(),
            "train_mean_after_imputation": baseline_mean.tolist(),
            "train_std_after_imputation": baseline_std.tolist(),
        },
    )

    write_json(
        ssl_dir / "input_manifest.json",
        {
            "dataset_id": DATASET_ID,
            "shape": ["n_bouts", SSL_TARGET_LENGTH, 3],
            "target_length": SSL_TARGET_LENGTH,
            "channels": [
                "sin(Heading_resampled)",
                "cos(Heading_resampled)",
                "derived_head_speed_mm_s_resampled",
            ],
            "heading_units": "radians",
            "temporal_policy": (
                "unwrap Heading; linearly resample angle and speed over "
                "normalized bout phase to 175 samples; then convert angle "
                "to sin/cos"
            ),
            "architecture_change_required": False,
            "metadata_used_as_encoder_input": False,
        },
    )

    write_json(
        ssl_dir / "normalization.json",
        {
            "fit_partition": "train",
            "sin_cos_standardized": False,
            "speed_standardized": True,
            "speed_mean": ssl_speed_mean,
            "speed_std": ssl_speed_std,
            "fit_over": (
                "all resampled temporal speed samples in DS-006 TRAIN only"
            ),
        },
    )

    # Recording split manifest.
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with (
        metadata_dir / "split_assignments.csv"
    ).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["recording_id", "family", "partition"]
        )

        for recording_id in sorted(recording_split):
            writer.writerow(
                [
                    recording_id,
                    family_from_recording(recording_id),
                    recording_split[recording_id],
                ]
            )

    # Accepted-bout metadata.
    with (
        metadata_dir / "bout_metadata.csv"
    ).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bout_id",
                "fish_id",
                "recording_id",
                "family",
                "well",
                "bout_index",
                "condition_code",
                "condition_label",
                "partition",
                "bout_start",
                "bout_end",
                "original_length",
            ]
        )

        for r in accepted_bouts:
            writer.writerow(
                [
                    r.bout_id,
                    r.fish_id,
                    r.recording_id,
                    r.family,
                    r.well,
                    r.bout_index,
                    r.condition_code,
                    r.condition_label,
                    recording_split[r.recording_id],
                    scalar_float(r.bout.BoutStart),
                    scalar_float(r.bout.BoutEnd),
                    len(as_1d(r.bout.HeadX)),
                ]
            )

    # QC and preparation summary.
    qc_summary = {
        "dataset_id": DATASET_ID,
        "status": "PREPARED",
        "frozen_qc": {
            **observed,
            "acceptance_rate": (
                accepted_count / qc_candidates
                if qc_candidates
                else None
            ),
            "rejection_reason_counts": dict(rejection_reasons),
            "qc_seed": QC_SEED,
            "fps_hz": FPS,
            "px_to_mm": PX_TO_MM,
        },
        "well_qc": {
            "empty_wells": [
                {"recording_id": r, "well": w}
                for r, w in sorted(empty_wells)
            ],
            "author_bad_wells": [
                {"recording_id": r, "well": w}
                for r, w in sorted(set(bad_wells_seen))
            ],
            "total_excluded_wells": (
                len(empty_wells) + len(set(bad_wells_seen))
            ),
        },
        "split": {
            "grouping_unit": "recording_id",
            "seed": SPLIT_SEED,
            "target_fractions": {
                "train": TRAIN_FRACTION,
                "validation": VALIDATION_FRACTION,
                "test": TEST_FRACTION,
            },
            "recording_counts": dict(split_counts_recordings),
            "bout_counts": {
                split: len(by_split[split])
                for split in ("train", "validation", "test")
            },
            "test_used_to_fit_preprocessing": False,
        },
        "baseline": {
            "feature_count": 18,
            "train_only_imputation": True,
            "train_only_scaling": True,
        },
        "ssl": {
            "target_length": SSL_TARGET_LENGTH,
            "channels": 3,
            "train_only_speed_normalization": True,
        },
        "replication_constraints": {
            "allowed_to_change_primary_method": False,
            "allowed_for_primary_hyperparameter_selection": False,
            "allowed_for_ssl_architecture_selection": False,
            "allowed_for_cluster_k_selection": False,
        },
    }

    write_json(
        metadata_dir / "qc_summary.json",
        qc_summary,
    )

    # ------------------------------------------------------------------
    # Final integrity checks
    # ------------------------------------------------------------------

    total_split_bouts = sum(
        len(by_split[s])
        for s in ("train", "validation", "test")
    )

    if total_split_bouts != EXPECTED_ACCEPTED_BOUTS:
        raise RuntimeError(
            "Split bout counts do not sum to frozen accepted count."
        )

    # Ensure recording disjointness.
    split_recordings = {
        split: {
            r.recording_id
            for r in by_split[split]
        }
        for split in ("train", "validation", "test")
    }

    assert split_recordings["train"].isdisjoint(
        split_recordings["validation"]
    )
    assert split_recordings["train"].isdisjoint(
        split_recordings["test"]
    )
    assert split_recordings["validation"].isdisjoint(
        split_recordings["test"]
    )

    print("\nDS-006 PREPARATION COMPLETE")
    print("=" * 40)
    print(f"Accepted bouts: {accepted_count:,}")
    print(
        "Recording split: "
        f"{split_counts_recordings['train']} train / "
        f"{split_counts_recordings['validation']} validation / "
        f"{split_counts_recordings['test']} test"
    )
    print(
        "Bout split:      "
        f"{len(by_split['train']):,} train / "
        f"{len(by_split['validation']):,} validation / "
        f"{len(by_split['test']):,} test"
    )
    print(
        f"SSL speed normalization (TRAIN): "
        f"mean={ssl_speed_mean:.12f}, "
        f"std={ssl_speed_std:.12f}"
    )
    print()
    print(f"Artifacts written to: {args.output_root}")
    print()
    print("TEST partition status: CREATED BUT NOT USED FOR FITTING")
    print(
        "Do not load the DS-006 test arrays during model/cluster selection."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
