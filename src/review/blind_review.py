"""Blind-review tooling for zebrafish representation comparisons.

The purpose is to reduce interpretation bias by replacing meaningful
representation/seed/cluster identifiers with deterministic anonymous codes.

This module does not alter numerical data. It only creates reversible label
maps. Keep the key file separate from material provided to blinded reviewers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_BLIND_SEED = 20260822


def _stable_digest(text: str, seed: int) -> str:
    payload = f"{seed}::{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_blind_map(
    labels: Iterable[str | int],
    *,
    prefix: str = "R",
    seed: int = DEFAULT_BLIND_SEED,
) -> dict[str, str]:
    """Map identifiers to anonymous codes using a deterministic shuffle."""
    unique = sorted({str(label) for label in labels})
    ordered = sorted(unique, key=lambda x: _stable_digest(x, seed))
    return {
        original: f"{prefix}{i + 1:02d}"
        for i, original in enumerate(ordered)
    }


def invert_blind_map(mapping: Mapping[str, str]) -> dict[str, str]:
    inverse = {value: key for key, value in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Blind map values are not unique.")
    return inverse


def blind_sequence(
    values: Sequence[str | int],
    mapping: Mapping[str, str],
) -> list[str]:
    """Apply a blind map to a sequence."""
    result: list[str] = []
    for value in values:
        key = str(value)
        if key not in mapping:
            raise KeyError(f"No blind code exists for {key!r}.")
        result.append(mapping[key])
    return result


def make_representation_blind_map(
    *,
    include_input_a: bool = True,
    ssl_seeds: Sequence[int] = (11, 23, 37, 51, 79),
    seed: int = DEFAULT_BLIND_SEED,
) -> dict[str, str]:
    """Blind Input A and all frozen SSL seed identities together."""
    identities: list[str] = []
    if include_input_a:
        identities.append("INPUT_A")
    identities.extend(f"SSL_SEED_{s}" for s in ssl_seeds)
    return make_blind_map(identities, prefix="R", seed=seed)


def make_cluster_blind_map(
    cluster_labels: Iterable[str | int],
    *,
    seed: int = DEFAULT_BLIND_SEED,
) -> dict[str, str]:
    """Create anonymous cluster IDs such as C01, C02, ..."""
    return make_blind_map(
        cluster_labels,
        prefix="C",
        seed=seed,
    )


def write_blind_key(
    mapping: Mapping[str, str],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the reversible key. Treat this file as reviewer-hidden."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "warning": "BLIND KEY - DO NOT PROVIDE TO BLINDED REVIEWERS",
        "mapping": dict(mapping),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_blinded_manifest(
    original_items: Sequence[str],
    mapping: Mapping[str, str],
    path: str | Path,
) -> Path:
    """Write a reviewer-safe manifest containing blind codes only."""
    blinded = blind_sequence(original_items, mapping)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "blinded_items": blinded,
        "contains_unblinded_identity": False,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
