#!/usr/bin/env python3
"""
DS-006 replication feasibility audit.

Purpose
-------
Verify whether the frozen external-replication dataset can support:

Input A
    The 18-feature hand-engineered locomotion / pose baseline used for DS-005.

Input B
    A temporal representation analogous to the DS-005 SSL input:
        sin(orientation), cos(orientation), speed

This script is an AUDIT. It does not train a model, select hyperparameters,
select cluster k, touch the DS-005 test partition, or modify the active SSL run.

Frozen DS-006 QC
----------------
FPS                 = 160 Hz
PX_TO_MM            = 0.071
QC seed             = 20260822
Author interpolation = splprep(..., s=10) with Gaussian jitter SD 0.1

Author bout rejection:
    time < 0.04 s or time > 1.2 s
    distance > 25 mm or distance < 0 mm
    speed > 50 mm/s or speed < 1 mm/s
    abs(deltahead) > 180 degrees

The original author notebook did not seed NumPy before adding interpolation
noise, so this replication fixes the seed for reproducibility.

Usage
-----
From repository root:

    python3 src/data/ds006_feasibility.py

Optional:

    python3 src/data/ds006_feasibility.py \
        --raw-root data/raw/DS-006/extracted/Data_all \
        --condition-map data/manifests/DS-006/well-condition-map.csv \
        --output data/processed/DS-006/feasibility/ds006_feasibility.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.io import loadmat


FPS = 160.0
PX_TO_MM = 0.071
QC_SEED = 20260822
INTERPOLATION_NOISE_SD = 0.1
INTERPOLATION_S = 10.0

EXPECTED_RAW_BOUTS = 165_579
EXPECTED_WELL_EXCLUDED_BOUTS = 1_556
EXPECTED_QC_CANDIDATES = 164_023
EXPECTED_ACCEPTED_BOUTS = 163_065
EXPECTED_REJECTED_BOUTS = 958

INPUT_A_FEATURES = (
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
)


@dataclass
class BoutRecord:
    recording_id: str
    family: str
    well: int
    bout_index: int
    bout: Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit DS-006 replication feasibility.")
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
        "--output",
        type=Path,
        default=Path(
            "data/processed/DS-006/feasibility/ds006_feasibility.json"
        ),
    )
    return p.parse_args()


def family_from_recording(recording_id: str) -> str:
    for fam in ("pH_1a", "pH_2a", "pH_2b", "pH_2c"):
        if fam in recording_id:
            return fam
    return "UNKNOWN"


def scalar_text(x: Any) -> str:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(x)


def as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def load_bad_wells(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Condition map not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Condition map is empty: {path}")

    required = {"recording_id", "well", "condition_label"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(
            f"Condition map missing required columns: {sorted(missing)}"
        )

    return {
        (r["recording_id"], int(r["well"]))
        for r in rows
        if r["condition_label"] == "bad_data"
    }


def iter_mat_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("results_*.mat"))
    if not files:
        raise FileNotFoundError(
            f"No results_*.mat files found under {root}. "
            "Run from the repository root or pass --raw-root."
        )
    return files


def iter_bouts(path: Path) -> tuple[str, list[np.ndarray]]:
    obj = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )["videoDataResults"]

    recording_id = scalar_text(obj.organization.videoName)
    wells = [np.atleast_1d(w) for w in np.atleast_1d(obj.wellPoissMouv)]
    return recording_id, wells


def author_interp_head(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    # RandomState is intentional: the original notebook used legacy
    # np.random.randn rather than default_rng().
    noisy_x = x + INTERPOLATION_NOISE_SD * rng.randn(len(x))
    noisy_y = y + INTERPOLATION_NOISE_SD * rng.randn(len(y))
    tck, u = splprep([noisy_x, noisy_y], s=INTERPOLATION_S)
    new_points = splev(u, tck)
    return np.asarray(new_points[0]), np.asarray(new_points[1])


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
    return float(np.sum(np.sqrt(dx * dx + dy * dy)) * PX_TO_MM)


def author_deltahead_deg(bout: Any) -> float:
    numps = 6

    tx = np.asarray(bout.TailX_VideoReferential, dtype=float)
    ty = np.asarray(bout.TailY_VideoReferential, dtype=float)
    hx = as_1d(bout.HeadX)
    hy = as_1d(bout.HeadY)

    if tx.ndim < 2 or ty.ndim < 2 or len(hx) == 0 or len(hy) == 0:
        raise ValueError("Insufficient tail/head data for author deltahead.")

    bxs = np.concatenate((np.asarray(tx[0][-numps:]).reshape(-1), [hx[0]]))
    bys = np.concatenate((np.asarray(ty[0][-numps:]).reshape(-1), [hy[0]]))
    slope0 = math.degrees(
        math.atan2(bys[-1] - bys[0], bxs[-1] - bxs[0])
    )

    bxs = np.concatenate((np.asarray(tx[-1][-numps:]).reshape(-1), [hx[-1]]))
    bys = np.concatenate((np.asarray(ty[-1][-numps:]).reshape(-1), [hy[-1]]))
    slope1 = math.degrees(
        math.atan2(bys[-1] - bys[0], bxs[-1] - bxs[0])
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
) -> tuple[bool, list[str], dict[str, float]]:
    x = as_1d(bout.HeadX)
    duration_s = len(x) / FPS

    try:
        distance_mm = author_distance_mm(bout, rng)
    except Exception:
        return False, ["interpolation_failure"], {
            "duration_s": float(duration_s),
            "distance_mm": float("nan"),
            "speed_mm_s": float("nan"),
            "deltahead_deg": float("nan"),
        }

    speed_mm_s = (
        distance_mm / duration_s if duration_s > 0 else float("inf")
    )

    try:
        deltahead_deg = author_deltahead_deg(bout)
    except Exception:
        return False, ["deltahead_failure"], {
            "duration_s": float(duration_s),
            "distance_mm": float(distance_mm),
            "speed_mm_s": float(speed_mm_s),
            "deltahead_deg": float("nan"),
        }

    failed: list[str] = []

    if duration_s < 0.04:
        failed.append("time_too_short")
    if duration_s > 1.2:
        failed.append("time_too_long")
    if distance_mm > 25:
        failed.append("distance_too_large")
    if distance_mm < 0:
        failed.append("distance_negative")
    if speed_mm_s > 50:
        failed.append("speed_too_high")
    if speed_mm_s < 1:
        failed.append("speed_too_low")
    if abs(deltahead_deg) > 180:
        failed.append("heading_change_too_large")

    return (not failed), failed, {
        "duration_s": float(duration_s),
        "distance_mm": float(distance_mm),
        "speed_mm_s": float(speed_mm_s),
        "deltahead_deg": float(deltahead_deg),
    }


def infer_heading_unit(values: np.ndarray) -> tuple[str, dict[str, float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "unknown", {}

    absvals = np.abs(finite)
    stats = {
        "max_abs": float(np.max(absvals)),
        "p99_abs": float(np.percentile(absvals, 99)),
    }

    # This is only a diagnostic inference. The audit reports it rather than
    # silently converting the source data.
    if stats["p99_abs"] <= 2 * np.pi + 0.25:
        return "radians_likely", stats
    if stats["p99_abs"] <= 360.0 + 5.0:
        return "degrees_likely", stats
    return "unknown", stats


def pointwise_speed_mm_s(bout: Any) -> np.ndarray:
    x = as_1d(bout.HeadX)
    y = as_1d(bout.HeadY)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.zeros(1, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    step_speed = np.sqrt(dx * dx + dy * dy) * PX_TO_MM * FPS

    # Keep temporal length equal to the positional/orientation series.
    return np.concatenate(([step_speed[0]], step_speed))


def orientation_radians(heading: np.ndarray, unit: str) -> np.ndarray:
    if unit == "radians_likely":
        return heading.astype(float, copy=False)
    if unit == "degrees_likely":
        return np.deg2rad(heading)
    raise ValueError("Heading unit could not be inferred safely.")


def feature_vector_feasibility(
    bout: Any,
    heading_unit: str,
    prev_bout: Any | None,
) -> dict[str, bool]:
    result = {name: False for name in INPUT_A_FEATURES}

    x = as_1d(bout.HeadX)
    y = as_1d(bout.HeadY)
    heading = as_1d(bout.Heading)

    n = min(len(x), len(y), len(heading))
    if n < 2:
        return result

    x, y, heading = x[:n], y[:n], heading[:n]
    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(heading))
    ):
        return result

    speed = pointwise_speed_mm_s(bout)[:n]

    if len(speed) == n and np.all(np.isfinite(speed)):
        result["bout_duration"] = True
        for k in (
            "speed_mean",
            "speed_std",
            "speed_median",
            "speed_max",
            "speed_p95",
            "speed_rms",
        ):
            result[k] = True

        if n >= 3:
            ds = np.diff(speed)
            if np.all(np.isfinite(ds)):
                for k in (
                    "speed_change_abs_mean",
                    "speed_change_std",
                    "speed_change_max",
                    "speed_change_rms",
                ):
                    result[k] = True

    if prev_bout is not None:
        try:
            start = float(np.asarray(bout.BoutStart).reshape(-1)[0])
            prev_end = float(np.asarray(prev_bout.BoutEnd).reshape(-1)[0])
            ibi = (start - prev_end) / FPS
            result["inter_bout_interval"] = np.isfinite(ibi)
        except Exception:
            result["inter_bout_interval"] = False

    try:
        ori = orientation_radians(heading, heading_unit)
        unwrapped = np.unwrap(ori)
        turns = np.diff(unwrapped)

        if len(turns) and np.all(np.isfinite(turns)):
            for k in (
                "turn_total_abs",
                "turn_net",
                "turn_abs_mean",
                "turn_std",
                "turn_max",
                "turn_rms",
            ):
                result[k] = True
    except ValueError:
        pass

    return result


def ssl_input_feasibility(
    bout: Any,
    heading_unit: str,
) -> tuple[bool, str | None, int | None]:
    x = as_1d(bout.HeadX)
    y = as_1d(bout.HeadY)
    heading = as_1d(bout.Heading)

    n = min(len(x), len(y), len(heading))
    if n < 2:
        return False, "too_short", n

    x, y, heading = x[:n], y[:n], heading[:n]

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.all(np.isfinite(heading))
    ):
        return False, "nonfinite_source", n

    try:
        ori = orientation_radians(heading, heading_unit)
    except ValueError:
        return False, "heading_unit_unknown", n

    speed = pointwise_speed_mm_s(bout)[:n]

    if len(speed) != n:
        return False, "length_mismatch", n
    if not np.all(np.isfinite(speed)):
        return False, "nonfinite_speed", n

    ssl = np.column_stack((np.sin(ori), np.cos(ori), speed))
    if ssl.shape != (n, 3):
        return False, "shape_failure", n
    if not np.all(np.isfinite(ssl)):
        return False, "nonfinite_ssl_input", n

    return True, None, n


def jsonable_counter(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def main() -> int:
    args = parse_args()

    mat_files = iter_mat_files(args.raw_root)
    bad_wells = load_bad_wells(args.condition_map)

    # First pass: gather Heading values only. This pass is lightweight and is
    # used solely to infer source units for the feasibility audit.
    heading_samples: list[np.ndarray] = []
    raw_bouts_first_pass = 0

    for path in mat_files:
        _, wells = iter_bouts(path)
        for well in wells:
            if well.size == 0:
                continue
            for bout in well:
                raw_bouts_first_pass += 1
                try:
                    h = as_1d(bout.Heading)
                    if h.size:
                        # Bound memory while still sampling the whole archive.
                        heading_samples.append(h[:: max(1, len(h) // 20)])
                except Exception:
                    pass

    heading_values = (
        np.concatenate(heading_samples)
        if heading_samples
        else np.empty(0, dtype=float)
    )
    heading_unit, heading_stats = infer_heading_unit(heading_values)

    rng = np.random.RandomState(QC_SEED)

    raw_bouts = 0
    empty_wells = 0
    author_bad_wells_seen = 0
    well_excluded_bouts = 0
    qc_candidates = 0
    accepted = 0
    rejected = 0

    qc_reasons = Counter()
    family_qc = defaultdict(Counter)

    feature_success = Counter()
    feature_failure = Counter()

    ssl_success = 0
    ssl_failure = 0
    ssl_failure_reasons = Counter()
    accepted_lengths: list[int] = []

    for path in mat_files:
        recording_id, wells = iter_bouts(path)
        fam = family_from_recording(recording_id)

        for well_index, well in enumerate(wells, start=1):
            if well.size == 0:
                empty_wells += 1
                continue

            raw_bouts += len(well)

            if (recording_id, well_index) in bad_wells:
                author_bad_wells_seen += 1
                well_excluded_bouts += len(well)
                family_qc[fam]["well_excluded_bouts"] += len(well)
                continue

            # Empty wells contain no bouts and were already skipped above.
            prev_accepted_bout: Any | None = None

            for bout_index, bout in enumerate(well):
                qc_candidates += 1
                family_qc[fam]["evaluated"] += 1

                ok, reasons, _ = qc_bout(bout, rng)

                if not ok:
                    rejected += 1
                    family_qc[fam]["rejected"] += 1
                    for reason in reasons:
                        qc_reasons[reason] += 1
                    continue

                accepted += 1
                family_qc[fam]["accepted"] += 1

                feas = feature_vector_feasibility(
                    bout,
                    heading_unit,
                    prev_accepted_bout,
                )
                for feature, possible in feas.items():
                    if possible:
                        feature_success[feature] += 1
                    else:
                        feature_failure[feature] += 1

                ssl_ok, ssl_reason, n = ssl_input_feasibility(
                    bout,
                    heading_unit,
                )
                if ssl_ok:
                    ssl_success += 1
                    if n is not None:
                        accepted_lengths.append(int(n))
                else:
                    ssl_failure += 1
                    ssl_failure_reasons[ssl_reason or "unknown"] += 1

                prev_accepted_bout = bout

    # The frozen count of ten excluded wells consists of seven author-bad wells
    # plus three empty wells. Empty wells contribute zero bouts.
    total_excluded_wells = author_bad_wells_seen + empty_wells

    feature_report: dict[str, dict[str, Any]] = {}
    for name in INPUT_A_FEATURES:
        yes = int(feature_success[name])
        no = int(feature_failure[name])
        denom = yes + no
        feature_report[name] = {
            "computable_bouts": yes,
            "not_computable_bouts": no,
            "computable_fraction": (yes / denom) if denom else None,
            # IBI is naturally undefined for the first accepted bout in a well.
            "note": (
                "First accepted bout in each usable well has no preceding "
                "accepted bout, so IBI is not expected to be 100%."
                if name == "inter_bout_interval"
                else None
            ),
        }

    length_report: dict[str, Any] = {}
    if accepted_lengths:
        a = np.asarray(accepted_lengths)
        length_report = {
            "min": int(np.min(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": int(np.max(a)),
            "unique_lengths": int(len(np.unique(a))),
        }

    frozen_qc_matches = {
        "raw_bouts": raw_bouts == EXPECTED_RAW_BOUTS,
        "well_excluded_bouts": (
            well_excluded_bouts == EXPECTED_WELL_EXCLUDED_BOUTS
        ),
        "qc_candidates": qc_candidates == EXPECTED_QC_CANDIDATES,
        "accepted_bouts": accepted == EXPECTED_ACCEPTED_BOUTS,
        "rejected_bouts": rejected == EXPECTED_REJECTED_BOUTS,
    }

    input_a_core_possible = all(
        feature_success[name] > 0 for name in INPUT_A_FEATURES
    )
    input_b_possible = ssl_success > 0 and ssl_failure == 0

    report = {
        "dataset_id": "DS-006",
        "audit": "replication_feasibility",
        "constants": {
            "fps_hz": FPS,
            "px_to_mm": PX_TO_MM,
            "qc_seed": QC_SEED,
        },
        "archive": {
            "mat_files": len(mat_files),
            "raw_bouts_first_pass": raw_bouts_first_pass,
            "raw_bouts": raw_bouts,
        },
        "heading": {
            "unit_inference": heading_unit,
            **heading_stats,
            "warning": (
                None
                if heading_unit != "unknown"
                else "Heading units could not be inferred safely."
            ),
        },
        "frozen_qc_recheck": {
            "empty_wells_seen": empty_wells,
            "author_bad_wells_seen": author_bad_wells_seen,
            "total_excluded_wells_seen": total_excluded_wells,
            "well_excluded_bouts": well_excluded_bouts,
            "qc_candidates": qc_candidates,
            "accepted_bouts": accepted,
            "rejected_bouts": rejected,
            "rejection_reasons": jsonable_counter(qc_reasons),
            "by_family": {
                fam: jsonable_counter(c)
                for fam, c in sorted(family_qc.items())
            },
            "matches_frozen_counts": frozen_qc_matches,
            "all_frozen_counts_match": all(frozen_qc_matches.values()),
        },
        "input_a": {
            "target_feature_count": len(INPUT_A_FEATURES),
            "target_features": list(INPUT_A_FEATURES),
            "feature_feasibility": feature_report,
            "all_18_have_computable_examples": input_a_core_possible,
            "interpretation": (
                "FEASIBLE_CANDIDATE"
                if input_a_core_possible
                else "NOT_YET_FEASIBLE"
            ),
            "caveat": (
                "This audit tests computability. It does not yet assert that "
                "DS-006 feature definitions are numerically identical to the "
                "frozen DS-005 baseline definitions."
            ),
        },
        "input_b": {
            "candidate_channels": [
                "sin(Heading)",
                "cos(Heading)",
                "derived_head_speed_mm_s",
            ],
            "successful_bouts": ssl_success,
            "failed_bouts": ssl_failure,
            "failure_reasons": jsonable_counter(ssl_failure_reasons),
            "temporal_length_summary": length_report,
            "all_accepted_bouts_shape_and_finite_valid": input_b_possible,
            "interpretation": (
                "FEASIBLE_CANDIDATE"
                if input_b_possible
                else "NEEDS_REVIEW"
            ),
            "caveat": (
                "DS-005 uses orientation_smooth and speed_head. DS-006 uses "
                "Heading plus speed derived from HeadX/HeadY, so semantic "
                "comparability must be documented before replication is frozen."
            ),
        },
        "constraints": {
            "changes_primary_method": False,
            "used_for_primary_hyperparameter_selection": False,
            "used_for_ssl_architecture_selection": False,
            "used_for_cluster_k_selection": False,
            "loads_ds005_test_partition": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print("DS-006 REPLICATION FEASIBILITY AUDIT")
    print("=" * 39)
    print(f"MAT files:             {len(mat_files):,}")
    print(f"Raw bouts:             {raw_bouts:,}")
    print(f"Excluded-well bouts:   {well_excluded_bouts:,}")
    print(f"QC candidates:         {qc_candidates:,}")
    print(f"Accepted:              {accepted:,}")
    print(f"Rejected:              {rejected:,}")
    print(f"Heading unit:          {heading_unit}")
    print()
    print("Frozen QC counts match:", all(frozen_qc_matches.values()))
    print("Input A:", report["input_a"]["interpretation"])
    print(
        "Input B:",
        report["input_b"]["interpretation"],
        f"({ssl_success:,} valid / {accepted:,} accepted)",
    )
    print()
    print("Input A feature computability:")
    for name in INPUT_A_FEATURES:
        x = feature_report[name]
        frac = x["computable_fraction"]
        pct = "n/a" if frac is None else f"{100 * frac:.3f}%"
        print(
            f"  {name:<25} "
            f"{x['computable_bouts']:>7,}/{accepted:<7,}  {pct}"
        )
    print()
    print(f"Report written to: {args.output}")

    # A mismatch against the already-frozen QC result is important enough to
    # make the command fail visibly.
    if not all(frozen_qc_matches.values()):
        print(
            "\nWARNING: deterministic QC does not match the frozen DS-006 "
            "counts. Do not freeze feasibility conclusions until resolved."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
