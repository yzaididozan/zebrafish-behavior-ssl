"""SSL input construction for frozen DS-005.

Converts one valid DS-005 BoutData object into the temporal tensor used
by the self-supervised representation-learning pipeline.

Primary candidate representation:

    sin(orientation)
    cos(orientation)
    speed_head

Output shape:

    (175, 3)

No fish identity, context, stimulus, bout label, or partition information
is included in the model input.
"""

from __future__ import annotations

import numpy as np

from src.data.ds005 import BoutData


EXPECTED_TEMPORAL_SAMPLES = 175
EXPECTED_CHANNELS = 3


def bout_to_ssl_input(bout: BoutData) -> np.ndarray:
    """Convert one valid DS-005 bout into an SSL input sequence.

    Returns
    -------
    np.ndarray
        Array with shape (175, 3), where columns are:

        0: sin(orientation_smooth)
        1: cos(orientation_smooth)
        2: speed_head
    """

    orientation = np.asarray(
        bout.orientation_smooth,
        dtype=np.float32,
    )

    speed = np.asarray(
        bout.speed_head,
        dtype=np.float32,
    )

    if orientation.shape != (EXPECTED_TEMPORAL_SAMPLES,):
        raise ValueError(
            f"Unexpected orientation shape: {orientation.shape}"
        )

    if speed.shape != (EXPECTED_TEMPORAL_SAMPLES,):
        raise ValueError(
            f"Unexpected speed shape: {speed.shape}"
        )

    if not np.all(np.isfinite(orientation)):
        raise ValueError("Orientation contains NaN or Inf.")

    if not np.all(np.isfinite(speed)):
        raise ValueError("Speed contains NaN or Inf.")

    X = np.column_stack(
        [
            np.sin(orientation),
            np.cos(orientation),
            speed,
        ]
    ).astype(np.float32)

    expected_shape = (
        EXPECTED_TEMPORAL_SAMPLES,
        EXPECTED_CHANNELS,
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected SSL input shape: {X.shape}; "
            f"expected {expected_shape}"
        )

    if not np.all(np.isfinite(X)):
        raise RuntimeError("Constructed SSL input contains NaN or Inf.")

    return X