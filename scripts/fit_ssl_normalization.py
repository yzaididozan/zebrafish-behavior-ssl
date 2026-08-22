"""Fit train-only SSL normalization statistics for frozen DS-005.

This script:
1. Loads the frozen DS-005 dataset through the canonical loader.
2. Uses only primary-QC-valid TRAIN bouts.
3. Computes global speed_head mean/std across all temporal samples.
4. Writes those values into configs/ssl/normalization.json.
5. Refuses to proceed if the target JSON is missing or malformed.

Run from the repository root:

    PYTHONPATH=. python3 scripts/fit_ssl_normalization.py

The validation and test partitions are never used to fit normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.ds005 import DS005


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_PATH = REPO_ROOT / "configs" / "ssl" / "normalization.json"


def fit_training_speed_stats(dataset: DS005) -> tuple[float, float, int, int]:
    """Compute global TRAIN-only speed mean/std with batchwise aggregation.

    Returns
    -------
    mean : float
        Mean speed over all finite temporal samples from included train bouts.
    std : float
        Population standard deviation over the same samples.
    sample_count : int
        Number of temporal speed samples used.
    bout_count : int
        Number of train bouts used.
    """

    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    bout_count = 0

    for bout in dataset.iter_bouts(
        partition="train",
        primary_qc_only=True,
        include_optional=False,
    ):
        speed = np.asarray(bout.speed_head, dtype=np.float64)

        if speed.shape != (175,):
            raise ValueError(
                f"Unexpected speed shape for "
                f"{bout.key.fish_id}/bout-{bout.key.bout_index}: "
                f"{speed.shape}"
            )

        if not np.all(np.isfinite(speed)):
            raise ValueError(
                f"Non-finite speed values found in "
                f"{bout.key.fish_id}/bout-{bout.key.bout_index}"
            )

        total_count += int(speed.size)
        total_sum += float(speed.sum(dtype=np.float64))
        total_sum_sq += float(np.square(speed, dtype=np.float64).sum(dtype=np.float64))
        bout_count += 1

    if total_count < 2:
        raise RuntimeError("Insufficient training speed samples.")

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean * mean)

    # Guard against tiny negative values from floating-point roundoff.
    variance = max(variance, 0.0)
    std = float(np.sqrt(variance))

    if not np.isfinite(mean):
        raise RuntimeError(f"Computed non-finite training speed mean: {mean}")

    if not np.isfinite(std) or std <= 0.0:
        raise RuntimeError(f"Computed invalid training speed std: {std}")

    return float(mean), std, total_count, bout_count


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization config not found:\n{path}\n\n"
            "Place normalization.json at configs/ssl/normalization.json first."
        )

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        speed_cfg = config["normalization"]["speed_head"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "normalization.json does not contain "
            "normalization.speed_head."
        ) from exc

    if speed_cfg.get("method") != "zscore":
        raise RuntimeError(
            "Expected normalization.speed_head.method to be 'zscore'."
        )

    if config.get("fit_partition") != "train":
        raise RuntimeError(
            "Frozen protocol requires fit_partition to be 'train'."
        )

    return config


def main() -> None:
    config = load_config(NORMALIZATION_PATH)

    print("=" * 68)
    print("FIT DS-005 TRAIN-ONLY SSL NORMALIZATION")
    print("=" * 68)
    print(f"Target config: {NORMALIZATION_PATH}")
    print("Partition used for fitting: train")
    print("Validation/test used for fitting: NO")
    print()

    with DS005() as dataset:
        mean, std, sample_count, bout_count = fit_training_speed_stats(dataset)

    config["normalization"]["speed_head"]["mean"] = mean
    config["normalization"]["speed_head"]["std"] = std
    config["status"] = "FITTED_TRAIN_ONLY"
    config["fit_summary"] = {
        "partition": "train",
        "primary_qc_only": True,
        "bout_count": bout_count,
        "temporal_sample_count": sample_count,
        "temporal_samples_per_bout": 175,
    }

    with NORMALIZATION_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("Completed.")
    print(f"train bouts used: {bout_count:,}")
    print(f"temporal speed samples used: {sample_count:,}")
    print(f"speed_mean: {mean:.12g}")
    print(f"speed_std:  {std:.12g}")
    print()
    print(f"Updated: {NORMALIZATION_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
