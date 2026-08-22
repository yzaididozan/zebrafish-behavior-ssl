"""Validate DS-005 SSL augmentations across a sample of TRAIN bouts.

This script performs broader train-only QC for the candidate ML-02
augmentation policy.

It:
1. Loads the frozen DS-005 dataset.
2. Uses only primary-QC-valid TRAIN bouts.
3. Builds the candidate SSL input tensor for each sampled bout.
4. Applies fitted train-only normalization from configs/ssl/normalization.json.
5. Generates two augmented views per bout.
6. Records per-bout change metrics.
7. Writes:
       results/augmentation_qc/bout_metrics.csv
       results/augmentation_qc/summary.json

Run from repository root:

    PYTHONPATH=. python3 scripts/validate_ssl_augmentations.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.data.ds005 import DS005
from src.ssl.augmentations import AugmentationConfig, make_two_views
from src.ssl.input import bout_to_ssl_input


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_PATH = REPO_ROOT / "configs" / "ssl" / "normalization.json"
OUTPUT_DIR = REPO_ROOT / "results" / "augmentation_qc"

SAMPLE_BOUTS = 1000
BASE_SEED = 20260822


def load_normalization() -> tuple[float, float]:
    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing normalization config: {NORMALIZATION_PATH}"
        )

    with NORMALIZATION_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    speed_cfg = cfg.get("normalization", {}).get("speed_head", {})
    mean = speed_cfg.get("mean")
    std = speed_cfg.get("std")

    if mean is None or std is None:
        raise RuntimeError(
            "Normalization stats are not fitted. "
            "Run scripts/fit_ssl_normalization.py first."
        )

    mean = float(mean)
    std = float(std)

    if not np.isfinite(mean):
        raise RuntimeError("Normalization mean is non-finite.")

    if not np.isfinite(std) or std <= 0:
        raise RuntimeError("Normalization std is invalid.")

    return mean, std


def normalize_input(
    X: np.ndarray,
    speed_mean: float,
    speed_std: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()

    if X.shape != (175, 3):
        raise ValueError(f"Unexpected SSL input shape: {X.shape}")

    X[:, 2] = (X[:, 2] - speed_mean) / speed_std

    if not np.all(np.isfinite(X)):
        raise RuntimeError("Normalization produced non-finite values.")

    return X


def view_metrics(
    original: np.ndarray,
    view: np.ndarray,
) -> dict[str, float | int | bool]:
    changed = view != original

    changed_elements = int(np.count_nonzero(changed))
    changed_timesteps = int(
        np.count_nonzero(np.any(changed, axis=1))
    )

    return {
        "changed_elements": changed_elements,
        "changed_element_fraction": changed_elements / original.size,
        "changed_timesteps": changed_timesteps,
        "changed_timestep_fraction": changed_timesteps / original.shape[0],
        "mean_abs_change": float(np.mean(np.abs(view - original))),
        "max_abs_change": float(np.max(np.abs(view - original))),
        "finite": bool(np.all(np.isfinite(view))),
        "shape_ok": bool(view.shape == original.shape),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    speed_mean, speed_std = load_normalization()

    config = AugmentationConfig(
        temporal_mask_probability=0.75,
        temporal_mask_max_fraction=0.10,
        feature_mask_probability=0.0,
        allow_orientation_mask=True,
        allow_speed_mask=True,
        mask_value=0.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "bout_metrics.csv"
    summary_path = OUTPUT_DIR / "summary.json"

    rows: list[dict[str, object]] = []

    print("=" * 68)
    print("DS-005 SSL AUGMENTATION QC")
    print("=" * 68)
    print(f"Target sample: {SAMPLE_BOUTS:,} TRAIN bouts")
    print("Validation/test used: NO")
    print(f"Normalization mean: {speed_mean:.12g}")
    print(f"Normalization std:  {speed_std:.12g}")
    print()

    with DS005() as dataset:
        iterator = dataset.iter_bouts(
            partition="train",
            primary_qc_only=True,
            include_optional=False,
        )

        for i, bout in enumerate(iterator):
            if i >= SAMPLE_BOUTS:
                break

            X = bout_to_ssl_input(bout)
            X = normalize_input(X, speed_mean, speed_std)

            # Deterministic per-bout seed.
            seed = BASE_SEED + i

            view_a, view_b = make_two_views(
                X,
                seed=seed,
                config=config,
            )

            ma = view_metrics(X, view_a)
            mb = view_metrics(X, view_b)

            rows.append(
                {
                    "sample_index": i,
                    "fish_id": bout.key.fish_id,
                    "bout_index": bout.key.bout_index,
                    "partition": bout.key.partition,
                    "view_a_changed_elements": ma["changed_elements"],
                    "view_a_changed_element_fraction": ma["changed_element_fraction"],
                    "view_a_changed_timesteps": ma["changed_timesteps"],
                    "view_a_changed_timestep_fraction": ma["changed_timestep_fraction"],
                    "view_a_mean_abs_change": ma["mean_abs_change"],
                    "view_a_max_abs_change": ma["max_abs_change"],
                    "view_a_finite": ma["finite"],
                    "view_a_shape_ok": ma["shape_ok"],
                    "view_b_changed_elements": mb["changed_elements"],
                    "view_b_changed_element_fraction": mb["changed_element_fraction"],
                    "view_b_changed_timesteps": mb["changed_timesteps"],
                    "view_b_changed_timestep_fraction": mb["changed_timestep_fraction"],
                    "view_b_mean_abs_change": mb["mean_abs_change"],
                    "view_b_max_abs_change": mb["max_abs_change"],
                    "view_b_finite": mb["finite"],
                    "view_b_shape_ok": mb["shape_ok"],
                    "views_equal": bool(np.array_equal(view_a, view_b)),
                }
            )

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1:,}/{SAMPLE_BOUTS:,} bouts...")

    if not rows:
        raise RuntimeError("No TRAIN bouts were processed.")

    fieldnames = list(rows[0].keys())

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    a_timestep = [float(r["view_a_changed_timestep_fraction"]) for r in rows]
    b_timestep = [float(r["view_b_changed_timestep_fraction"]) for r in rows]
    a_element = [float(r["view_a_changed_element_fraction"]) for r in rows]
    b_element = [float(r["view_b_changed_element_fraction"]) for r in rows]

    finite_failures = sum(
        (not bool(r["view_a_finite"])) or (not bool(r["view_b_finite"]))
        for r in rows
    )

    shape_failures = sum(
        (not bool(r["view_a_shape_ok"])) or (not bool(r["view_b_shape_ok"]))
        for r in rows
    )

    equal_views = sum(bool(r["views_equal"]) for r in rows)

    summary = {
        "dataset_id": "DS-005",
        "partition": "train",
        "sample_bouts_requested": SAMPLE_BOUTS,
        "sample_bouts_processed": len(rows),
        "base_seed": BASE_SEED,
        "normalization": {
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "fit_partition": "train",
        },
        "augmentation_config": {
            "temporal_mask_probability": 0.50,
            "temporal_mask_max_fraction": 0.10,
            "feature_mask_probability": 0.0,
            "mask_value": 0.0,
        },
        "view_a_changed_timestep_fraction": summarize(a_timestep),
        "view_b_changed_timestep_fraction": summarize(b_timestep),
        "view_a_changed_element_fraction": summarize(a_element),
        "view_b_changed_element_fraction": summarize(b_element),
        "finite_failures": finite_failures,
        "shape_failures": shape_failures,
        "equal_view_pairs": equal_views,
        "acceptance_checks": {
            "all_outputs_finite": finite_failures == 0,
            "all_shapes_preserved": shape_failures == 0,
            "max_changed_timestep_fraction_le_0_10": (
                max(a_timestep + b_timestep) <= 0.10 + 1e-12
            ),
        },
        "status": "QC_COMPLETE_NOT_YET_FROZEN",
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print()
    print("QC complete.")
    print(f"Processed bouts: {len(rows):,}")
    print(f"Finite failures: {finite_failures}")
    print(f"Shape failures: {shape_failures}")
    print(f"Equal view pairs: {equal_views}")
    print(
        "Max changed timestep fraction: "
        f"{max(a_timestep + b_timestep):.4f}"
    )
    print()
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {summary_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
