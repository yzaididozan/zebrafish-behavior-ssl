"""SSL augmentations for frozen DS-005 bout-level inputs.

This module implements conservative, leakage-safe augmentations for the
candidate Input B representation used by the zebrafish-behavior-ssl project.

Expected input
--------------
One valid DS-005 bout converted by ``src.ssl.input.bout_to_ssl_input``:

    X.shape == (175, 3)

Channels:
    0 = sin(orientation_smooth)
    1 = cos(orientation_smooth)
    2 = speed_head

Design rules
------------
1. Augmentations operate only within a single bout.
2. Fish/session/context/stimulus metadata are never mixed into model inputs.
3. No samples are borrowed from another fish, bout, or partition.
4. Orientation sine/cosine channels are treated as a paired feature.
5. Temporal order is preserved.
6. Time reversal and temporal shuffling are intentionally not implemented.
7. All randomness is controlled by an explicit NumPy Generator.
8. Augmentations preserve shape and finite values.

The initial MVP includes:
    - temporal masking;
    - grouped feature masking;
    - deterministic two-view generation.

Noise, temporal cropping, warping, and spatial transformations should only be
added after they are separately justified and validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


EXPECTED_TIMESTEPS = 175
EXPECTED_CHANNELS = 3
EXPECTED_SHAPE = (EXPECTED_TIMESTEPS, EXPECTED_CHANNELS)

ORIENTATION_CHANNELS = (0, 1)
SPEED_CHANNEL = 2


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for conservative DS-005 SSL augmentations.

    Parameters
    ----------
    temporal_mask_probability:
        Probability that temporal masking is applied to a view.
    temporal_mask_max_fraction:
        Maximum fraction of time steps that may be masked in one contiguous
        block. Must be in [0, 1].
    feature_mask_probability:
        Probability that grouped feature masking is applied to a view.
    allow_orientation_mask:
        Whether the paired sin/cos orientation channels may be masked.
    allow_speed_mask:
        Whether the speed channel may be masked.
    mask_value:
        Value written into masked elements. Zero is appropriate after
        normalization and is also safe before normalization for the initial
        structural tests.
    """

    temporal_mask_probability: float = 0.50
    temporal_mask_max_fraction: float = 0.10
    feature_mask_probability: float = 0.25
    allow_orientation_mask: bool = True
    allow_speed_mask: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        _validate_probability(
            self.temporal_mask_probability,
            "temporal_mask_probability",
        )
        _validate_probability(
            self.feature_mask_probability,
            "feature_mask_probability",
        )

        if not 0.0 <= self.temporal_mask_max_fraction <= 1.0:
            raise ValueError(
                "temporal_mask_max_fraction must be between 0 and 1."
            )

        if not np.isfinite(self.mask_value):
            raise ValueError("mask_value must be finite.")

        if not self.allow_orientation_mask and not self.allow_speed_mask:
            raise ValueError(
                "At least one feature group must be eligible for masking."
            )


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def validate_ssl_input(X: np.ndarray) -> np.ndarray:
    """Validate and return one SSL input as float32.

    This function does not modify the original array.
    """

    X = np.asarray(X, dtype=np.float32)

    if X.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected SSL input shape {EXPECTED_SHAPE}, got {X.shape}."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("SSL input contains NaN or Inf.")

    return X


def temporal_mask(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    max_fraction: float,
    mask_value: float = 0.0,
) -> np.ndarray:
    """Mask one contiguous temporal block across all channels.

    Temporal order is preserved. The selected time steps are replaced by the
    mask value; no samples are moved, shuffled, or borrowed from other bouts.
    """

    X = validate_ssl_input(X).copy()

    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError("max_fraction must be between 0 and 1.")

    max_steps = int(np.floor(EXPECTED_TIMESTEPS * max_fraction))

    if max_steps <= 0:
        return X

    mask_length = int(rng.integers(1, max_steps + 1))
    max_start = EXPECTED_TIMESTEPS - mask_length
    start = int(rng.integers(0, max_start + 1))
    stop = start + mask_length

    X[start:stop, :] = np.float32(mask_value)

    return X


def feature_mask(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    allow_orientation_mask: bool = True,
    allow_speed_mask: bool = True,
    mask_value: float = 0.0,
) -> np.ndarray:
    """Mask one complete feature group for the entire bout.

    The two orientation channels are always treated together. Masking only
    sin(theta) or only cos(theta) would break the circular representation.

    Eligible groups:
        - orientation pair: channels 0 and 1;
        - speed: channel 2.
    """

    X = validate_ssl_input(X).copy()

    groups = []

    if allow_orientation_mask:
        groups.append("orientation")

    if allow_speed_mask:
        groups.append("speed")

    if not groups:
        raise ValueError("No feature groups are eligible for masking.")

    group = groups[int(rng.integers(0, len(groups)))]

    if group == "orientation":
        X[:, ORIENTATION_CHANNELS] = np.float32(mask_value)
    elif group == "speed":
        X[:, SPEED_CHANNEL] = np.float32(mask_value)
    else:
        raise RuntimeError(f"Unexpected feature group: {group}")

    return X


def augment_view(
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    config: Optional[AugmentationConfig] = None,
) -> np.ndarray:
    """Generate one augmented view of a single bout.

    Augmentations are applied independently according to the configured
    probabilities. The returned array always preserves the input shape.
    """

    config = config or AugmentationConfig()
    view = validate_ssl_input(X).copy()

    if rng.random() < config.temporal_mask_probability:
        view = temporal_mask(
            view,
            rng=rng,
            max_fraction=config.temporal_mask_max_fraction,
            mask_value=config.mask_value,
        )

    if rng.random() < config.feature_mask_probability:
        view = feature_mask(
            view,
            rng=rng,
            allow_orientation_mask=config.allow_orientation_mask,
            allow_speed_mask=config.allow_speed_mask,
            mask_value=config.mask_value,
        )

    if view.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Augmentation changed shape from {EXPECTED_SHAPE} to {view.shape}."
        )

    if not np.all(np.isfinite(view)):
        raise RuntimeError("Augmentation produced NaN or Inf.")

    return view.astype(np.float32, copy=False)


def make_two_views(
    X: np.ndarray,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[AugmentationConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two independently augmented views of the same bout.

    Exactly one of ``seed`` or ``rng`` may be supplied.

    Examples
    --------
    Deterministic:

        view_a, view_b = make_two_views(X, seed=20260822)

    Stateful generator:

        rng = np.random.default_rng(20260822)
        view_a, view_b = make_two_views(X, rng=rng)
    """

    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")

    if rng is None:
        rng = np.random.default_rng(seed)

    X = validate_ssl_input(X)

    view_a = augment_view(
        X,
        rng=rng,
        config=config,
    )

    view_b = augment_view(
        X,
        rng=rng,
        config=config,
    )

    return view_a, view_b


__all__ = [
    "AugmentationConfig",
    "augment_view",
    "feature_mask",
    "make_two_views",
    "temporal_mask",
    "validate_ssl_input",
]
