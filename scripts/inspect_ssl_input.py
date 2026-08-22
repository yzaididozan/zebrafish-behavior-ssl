"""Inspect one real DS-005 training bout for SSL input design."""

from __future__ import annotations
from src.ssl.input import bout_to_ssl_input
import numpy as np

from src.data.ds005 import DS005


def count_nonfinite(*arrays: np.ndarray) -> tuple[int, int]:
    """Return total NaN and Inf counts across supplied arrays."""
    nan_count = 0
    inf_count = 0

    for array in arrays:
        arr = np.asarray(array)

        if np.issubdtype(arr.dtype, np.number):
            nan_count += int(np.isnan(arr).sum())
            inf_count += int(np.isinf(arr).sum())

    return nan_count, inf_count


def main() -> None:
    # DS005 validates the frozen dataset and split by default.
    with DS005() as dataset:

        # Get the first valid bout belonging to a TRAIN fish only.
        bout = next(
            dataset.iter_bouts(
                partition="train",
                primary_qc_only=True,
                include_optional=False,
            )
        )
        X = bout_to_ssl_input(bout)

        print()
        print("CANDIDATE SSL INPUT")
        print("-" * 60)
        print(f"X.shape: {X.shape}")
        print(f"X.dtype: {X.dtype}")
        print(f"finite: {np.all(np.isfinite(X))}")
        print()
        print("First 5 timesteps:")
        print(X[:5])
        head_pos = np.asarray(bout.head_pos)
        orientation = np.asarray(bout.orientation_smooth)
        speed = np.asarray(bout.speed_head)
        times = np.asarray(bout.times_bouts)

        nan_count, inf_count = count_nonfinite(
            head_pos,
            orientation,
            speed,
            times,
        )

        print("=" * 60)
        print("DS-005 SSL INPUT INSPECTION")
        print("=" * 60)

        print(f"fish_id: {bout.key.fish_id}")
        print(f"bout_index: {bout.key.bout_index}")
        print(f"partition: {bout.key.partition}")

        print()
        print("ARRAY SHAPES")
        print("-" * 60)
        print(f"head_pos.shape: {head_pos.shape}")
        print(
            f"orientation_smooth.shape: "
            f"{orientation.shape}"
        )
        print(f"speed_head.shape: {speed.shape}")

        print()
        print("TIMING")
        print("-" * 60)
        print(f"times_bouts: {times}")

        print()
        print("SPEED")
        print("-" * 60)
        print(f"min speed: {np.min(speed)}")
        print(f"max speed: {np.max(speed)}")

        print()
        print("ORIENTATION")
        print("-" * 60)
        print(f"min orientation: {np.min(orientation)}")
        print(f"max orientation: {np.max(orientation)}")

        print()
        print("NON-FINITE VALUES")
        print("-" * 60)
        print(f"NaN count: {nan_count}")
        print(f"Inf count: {inf_count}")

        print()
        print("QC")
        print("-" * 60)
        print(f"primary_exclude: {bout.qc.primary_exclude}")
        print(f"all_zero_speed: {bout.qc.all_zero_speed}")
        print(
            f"extreme_speed_gt_100: "
            f"{bout.qc.extreme_speed_gt_100}"
        )

        print("=" * 60)


if __name__ == "__main__":
    main()