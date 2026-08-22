"""Embedding export utilities for DS-005 SSL.

This module contains reusable mechanics for converting normalized DS-005 bouts
into encoder embeddings using a frozen SSL checkpoint.

Important
---------
- Downstream discovery uses encoder embeddings, NOT projection-head outputs.
- Metadata is exported separately from the embedding matrix.
- This module does not perform model selection.
- TEST export is allowed only when explicitly requested after all design choices
  are frozen. The default launcher exports TRAIN and VALIDATION only.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def normalize_ssl_input(
    x: np.ndarray,
    *,
    speed_mean: float,
    speed_std: float,
) -> np.ndarray:
    """Apply frozen TRAIN-only normalization to one SSL input."""
    x = np.asarray(x, dtype=np.float32).copy()

    if x.shape != (175, 3):
        raise ValueError(f"Expected SSL input shape (175, 3), got {x.shape}.")

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN or Inf before normalization.")

    x[:, 2] = (x[:, 2] - float(speed_mean)) / float(speed_std)

    if not np.isfinite(x).all():
        raise ValueError("SSL input contains NaN or Inf after normalization.")

    return x


@torch.no_grad()
def encode_batch(
    *,
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return encoder embeddings for a batch of normalized SSL inputs.

    Parameters
    ----------
    model:
        ContrastiveModel-like module returning (embedding, projection).
    batch:
        Float tensor of shape (B, 175, 3).
    device:
        Torch device.

    Returns
    -------
    torch.Tensor
        CPU tensor of shape (B, embedding_dim).
    """
    if batch.ndim != 3:
        raise ValueError(
            f"Expected batch with rank 3 (B, T, C), got {tuple(batch.shape)}."
        )

    if batch.shape[1:] != (175, 3):
        raise ValueError(
            f"Expected batch shape (B, 175, 3), got {tuple(batch.shape)}."
        )

    if not torch.isfinite(batch).all():
        raise ValueError("Embedding input batch contains NaN or Inf.")

    model.eval()

    batch = batch.to(
        device=device,
        dtype=torch.float32,
    )

    embedding, _projection = model(batch)

    if embedding.ndim != 2:
        raise RuntimeError(
            f"Encoder returned wrong embedding rank: {tuple(embedding.shape)}."
        )

    if embedding.shape[0] != batch.shape[0]:
        raise RuntimeError(
            "Encoder embedding batch dimension does not match input batch."
        )

    if not torch.isfinite(embedding).all():
        raise RuntimeError("Encoder produced NaN or Inf embeddings.")

    return embedding.detach().cpu()


def save_embedding_partition(
    *,
    output_dir: Path,
    partition: str,
    seed: int,
    embeddings: np.ndarray,
    metadata_rows: Sequence[Dict[str, Any]],
    checkpoint_path: Path,
    split_seed: int,
    dataset_id: str = "DS-005",
) -> Tuple[Path, Path, Path]:
    """Save embeddings, metadata CSV, and a compact manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must have shape (N, D), got {embeddings.shape}."
        )

    if len(metadata_rows) != embeddings.shape[0]:
        raise ValueError(
            "Metadata row count must match number of embedding rows."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Cannot save embeddings containing NaN or Inf.")

    stem = f"ssl_seed{seed}_{partition}"

    embedding_path = output_dir / f"{stem}_embeddings.npy"
    metadata_path = output_dir / f"{stem}_metadata.csv"
    manifest_path = output_dir / f"{stem}_manifest.json"

    np.save(embedding_path, embeddings)

    if metadata_rows:
        fieldnames: List[str] = []
        seen = set()
        for row in metadata_rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
    else:
        metadata_path.write_text("", encoding="utf-8")

    import json

    manifest = {
        "dataset_id": dataset_id,
        "partition": partition,
        "training_seed": int(seed),
        "split_seed": int(split_seed),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "projection_head_exported": False,
        "metadata_stored_separately": True,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return embedding_path, metadata_path, manifest_path
