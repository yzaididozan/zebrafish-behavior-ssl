"""Inspect SSL augmentations on one real DS-005 training bout.

This script:
1. Loads the frozen DS-005 dataset.
2. Selects one primary-QC-valid TRAIN bout.
3. Converts it to the candidate SSL input tensor.
4. Optionally applies frozen train-only normalization if available.
5. Generates two augmented views of the same bout.
6. Prints structural and change diagnostics.

Run from the repository root:

    PYTHONPATH=. python3 scripts/inspect_ssl_augmentations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.ds005 import DS005
from src.ssl.augmentations import AugmentationConfig, make_two_views
from src.ssl.input import bout_to_ssl_input


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZATION_PATH = REPO_ROOT / "configs" / "ssl" / "normalization.json"


def maybe_normalize(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Apply frozen speed normalization if fitted stats are available."""

    if not NORMALIZATION_PATH.exists():
        return X.astype(np.float32, copy=True), False

    with NORMALIZATION_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    speed_cfg = cfg.get("normalization", {}).get("speed_head", {})
    mean = speed_cfg.get("mean")
    std = speed_cfg.get("std")

    if mean is None or std is None:
        return X.astype(np.float32, copy=True), False

    mean = float(mean)
    std = float(std)

    if not np.isfinite(mean):
        raise RuntimeError("Normalization speed mean is non-finite.")

    if not np.isfinite(std) or std <= 0.0:
        raise RuntimeError("Normalization speed std is invalid.")

    Xn = X.astype(np.float32, copy=True)
    Xn[:, 2] = (Xn[:, 2] - mean) / std

    if not np.all(np.isfinite(Xn)):
        raise RuntimeError("Normalization produced NaN or Inf.")

    return Xn, True


def summarize_change(
    name: str,
    original: np.ndarray,
    view: np.ndarray,
) -> None:
    """Print simple change diagnostics for one augmented view."""

    changed_elements = int(np.count_nonzero(view != original))
    total_elements = int(original.size)

    changed_timesteps = int(
        np.count_nonzero(
            np.any(view != original, axis=1)
        )
    )

    mean_abs_change = float(
        np.mean(np.abs(view - original))
    )

    max_abs_change = float(
        np.max(np.abs(view - original))
    )

    print(name)
    print("-" * 60)
    print(f"shape: {view.shape}")
    print(f"dtype: {view.dtype}")
    print(f"finite: {np.all(np.isfinite(view))}")
    print(
        f"changed elements: "
        f"{changed_elements:,}/{total_elements:,} "
        f"({changed_elements / total_elements:.2%})"
    )
    print(
        f"changed timesteps: "
        f"{changed_timesteps}/{original.shape[0]} "
        f"({changed_timesteps / original.shape[0]:.2%})"
    )
    print(f"mean absolute change: {mean_abs_change:.6f}")
    print(f"max absolute change: {max_abs_change:.6f}")
    print()


def main() -> None:
    config = AugmentationConfig(
    temporal_mask_probability=1.0,
    temporal_mask_max_fraction=0.10,
    feature_mask_probability=0.0,
    allow_orientation_mask=True,
    allow_speed_mask=True,
    mask_value=0.0,
    )

    with DS005() as dataset:
        bout = next(
            dataset.iter_bouts(
                partition="train",
                primary_qc_only=True,
                include_optional=False,
            )
        )

        X_raw = bout_to_ssl_input(bout)
        X, normalized = maybe_normalize(X_raw)

        view_a, view_b = make_two_views(
            X,
            seed=20260822,
            config=config,
        )

    print("=" * 60)
    print("REAL DS-005 SSL AUGMENTATION INSPECTION")
    print("=" * 60)
    print(f"fish_id: {bout.key.fish_id}")
    print(f"bout_index: {bout.key.bout_index}")
    print(f"partition: {bout.key.partition}")
    print(f"normalized with fitted train stats: {normalized}")
    print(f"original shape: {X.shape}")
    print(f"original dtype: {X.dtype}")
    print(f"original finite: {np.all(np.isfinite(X))}")
    print()

    summarize_change("VIEW A", X, view_a)
    summarize_change("VIEW B", X, view_b)

    print("PAIR COMPARISON")
    print("-" * 60)
    print(f"view_a equals view_b: {np.array_equal(view_a, view_b)}")
    print(
        "mean absolute difference between views: "
        f"{np.mean(np.abs(view_a - view_b)):.6f}"
    )
    print()

    print("FIRST 5 TIMESTEPS")
    print("-" * 60)
    print("original:")
    print(X[:5])
    print()
    print("view_a:")
    print(view_a[:5])
    print()
    print("view_b:")
    print(view_b[:5])
    print()

    print("=" * 60)


if __name__ == "__main__":
    main()
