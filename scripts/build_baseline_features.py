"""Build frozen DS-005 baseline feature artifacts.

This script constructs Input A for the zebrafish-behavior-ssl project from
the frozen DS-005 primary dataset.

Outputs
-------
data/processed/DS-005/baseline/
├── train_core_raw.npz
├── validation_core_raw.npz
├── test_core_raw.npz
├── train_core_scaled.npz
├── validation_core_scaled.npz
├── test_core_scaled.npz
├── scaler_core.json
├── feature_schema_core.json
├── build_audit_core.json
└── SHA256SUMS

Protocol safeguards
-------------------
- Fish-level frozen split is inherited from DS005.
- Raw features are extracted independently for train/validation/test.
- Scaler is fit on TRAIN only.
- Validation/test are transformed using the frozen train scaler.
- Context, stimulus code, and bout type are metadata only.
- Core confirmatory profile excludes raw head-position-derived features.
- Existing outputs are not overwritten unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DATA = REPO_ROOT / "src" / "data"
SRC_FEATURES = REPO_ROOT / "src" / "features"

for path in (SRC_DATA, SRC_FEATURES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ds005 import DS005  # noqa: E402
from baseline import (  # noqa: E402
    CORE_FEATURE_NAMES,
    audit_feature_matrix,
    extract_partition_features,
    fit_train_scaler,
    save_feature_matrix_npz,
    save_feature_schema,
    transform_feature_matrix,
)


PARTITIONS = ("train", "validation", "test")
PROFILE = "core"

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Return SHA-256 without loading the entire file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_json(
    payload: Mapping[str, object],
    path: Path,
) -> None:
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        temp_name = tmp.name

    os.replace(temp_name, path)


def assert_output_policy(
    paths: Iterable[Path],
    *,
    overwrite: bool,
) -> None:
    """Refuse accidental overwrite of frozen-like outputs."""
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        formatted = "\n".join(
            f"  - {path}" for path in existing
        )
        raise FileExistsError(
            "Baseline outputs already exist.\n"
            "Refusing to overwrite them without --overwrite:\n"
            f"{formatted}"
        )


def write_sha256sums(
    output_dir: Path,
    paths: Sequence[Path],
) -> Path:
    """Write deterministic SHA256SUMS file."""
    sums_path = output_dir / "SHA256SUMS"

    rows = []

    for path in sorted(paths, key=lambda p: p.name):
        digest = sha256_file(path)
        rows.append(f"{digest}  {path.name}")

    sums_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    return sums_path


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def verify_partition_metadata(
    features,
    expected_partition: str,
) -> None:
    observed = {
        row.partition
        for row in features.metadata
    }

    if observed != {expected_partition}:
        raise RuntimeError(
            f"{expected_partition}: unexpected partition metadata: "
            f"{observed}"
        )


def verify_no_fish_overlap(
    matrices: Mapping[str, object],
) -> None:
    fish = {
        partition: {
            row.fish_id
            for row in matrix.metadata
        }
        for partition, matrix in matrices.items()
    }

    if fish["train"] & fish["validation"]:
        raise RuntimeError(
            "Train/validation fish overlap detected."
        )

    if fish["train"] & fish["test"]:
        raise RuntimeError(
            "Train/test fish overlap detected."
        )

    if fish["validation"] & fish["test"]:
        raise RuntimeError(
            "Validation/test fish overlap detected."
        )


def verify_scaled_matrices(
    raw_matrices,
    scaled_matrices,
) -> None:
    for partition in PARTITIONS:
        raw = raw_matrices[partition]
        scaled = scaled_matrices[partition]

        if raw.X.shape != scaled.X.shape:
            raise RuntimeError(
                f"{partition}: raw/scaled shape mismatch."
            )

        if raw.feature_names != scaled.feature_names:
            raise RuntimeError(
                f"{partition}: raw/scaled schema mismatch."
            )

        if raw.metadata != scaled.metadata:
            raise RuntimeError(
                f"{partition}: scaling changed row metadata."
            )

        if not np.all(np.isfinite(scaled.X)):
            raise RuntimeError(
                f"{partition}: scaled matrix contains non-finite values."
            )


def verify_train_scaling(
    train_scaled,
) -> Dict[str, object]:
    """Audit expected z-score behavior on training data."""
    means = np.mean(
        train_scaled.X.astype(np.float64),
        axis=0,
    )
    stds = np.std(
        train_scaled.X.astype(np.float64),
        axis=0,
        ddof=0,
    )

    # Float32 persistence introduces tiny deviations from exact zero/one.
    centered = bool(
        np.allclose(
            means,
            np.zeros(len(means)),
            atol=1e-4,
        )
    )

    unit_scaled = bool(
        np.allclose(
            stds,
            np.ones(len(stds)),
            atol=1e-4,
        )
    )

    if not centered:
        raise RuntimeError(
            "Training scaled features are not centered near zero."
        )

    if not unit_scaled:
        raise RuntimeError(
            "Training scaled features are not unit variance."
        )

    return {
        "mean_abs_max": float(np.max(np.abs(means))),
        "std_abs_deviation_from_one_max": float(
            np.max(np.abs(stds - 1.0))
        ),
        "centered_with_atol_1e-4": centered,
        "unit_variance_with_atol_1e-4": unit_scaled,
    }


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_baseline_features(
    *,
    repo_root: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Build all frozen core baseline artifacts."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {
        "train_raw": output_dir / "train_core_raw.npz",
        "validation_raw": (
            output_dir / "validation_core_raw.npz"
        ),
        "test_raw": output_dir / "test_core_raw.npz",
        "train_scaled": (
            output_dir / "train_core_scaled.npz"
        ),
        "validation_scaled": (
            output_dir / "validation_core_scaled.npz"
        ),
        "test_scaled": (
            output_dir / "test_core_scaled.npz"
        ),
        "scaler": output_dir / "scaler_core.json",
        "schema": (
            output_dir / "feature_schema_core.json"
        ),
        "audit": output_dir / "build_audit_core.json",
    }

    assert_output_policy(
        output_paths.values(),
        overwrite=overwrite,
    )

    print("DS-005 BASELINE FEATURE BUILD")
    print("=============================")
    print(f"Repository: {repo_root}")
    print(f"Output:     {output_dir}")
    print(f"Profile:    {PROFILE}")
    print()

    raw_matrices = {}

    with DS005(repo_root=repo_root) as ds:
        ds.assert_no_fish_overlap()

        print("Dataset validation: PASSED")
        print(
            f"Fish partitions: {ds.partition_counts()}"
        )
        print()

        # ---------------------------------------------------------------
        # Raw extraction
        # ---------------------------------------------------------------

        for partition in PARTITIONS:
            print(
                f"Extracting raw {partition} features..."
            )

            matrix = extract_partition_features(
                ds,
                partition,
                profile=PROFILE,
            )

            verify_partition_metadata(
                matrix,
                partition,
            )

            raw_matrices[partition] = matrix

            print(
                f"  rows={matrix.n_rows:,}, "
                f"features={matrix.n_features}, "
                f"fish={len(set(matrix.fish_ids()))}"
            )

        verify_no_fish_overlap(
            raw_matrices
        )

        total_rows = sum(
            matrix.n_rows
            for matrix in raw_matrices.values()
        )

        if total_rows != ds.n_valid_bouts:
            raise RuntimeError(
                "Baseline row total does not equal DS-005 "
                f"valid bout total: {total_rows:,} vs "
                f"{ds.n_valid_bouts:,}"
            )

        print()
        print(
            "Raw feature extraction integrity: PASSED"
        )
        print(
            f"Total baseline rows: {total_rows:,}"
        )

        # ---------------------------------------------------------------
        # Train-only scaler
        # ---------------------------------------------------------------

        print()
        print("Fitting scaler on TRAIN only...")

        scaler = fit_train_scaler(
            raw_matrices["train"]
        )

        if scaler.fitted_on_partition != "train":
            raise RuntimeError(
                "Scaler was not marked as train-fitted."
            )

        print("Train-only scaler fit: PASSED")

        # ---------------------------------------------------------------
        # Transform all partitions
        # ---------------------------------------------------------------

        scaled_matrices = {}

        for partition in PARTITIONS:
            print(
                f"Transforming {partition} using train scaler..."
            )

            scaled_matrices[partition] = (
                transform_feature_matrix(
                    raw_matrices[partition],
                    scaler,
                )
            )

        verify_scaled_matrices(
            raw_matrices,
            scaled_matrices,
        )

        train_scaling_audit = verify_train_scaling(
            scaled_matrices["train"]
        )

        print("Scaled matrix integrity: PASSED")

        # ---------------------------------------------------------------
        # Save artifacts
        # ---------------------------------------------------------------

        print()
        print("Writing artifacts...")

        save_feature_matrix_npz(
            raw_matrices["train"],
            output_paths["train_raw"],
        )
        save_feature_matrix_npz(
            raw_matrices["validation"],
            output_paths["validation_raw"],
        )
        save_feature_matrix_npz(
            raw_matrices["test"],
            output_paths["test_raw"],
        )

        save_feature_matrix_npz(
            scaled_matrices["train"],
            output_paths["train_scaled"],
        )
        save_feature_matrix_npz(
            scaled_matrices["validation"],
            output_paths["validation_scaled"],
        )
        save_feature_matrix_npz(
            scaled_matrices["test"],
            output_paths["test_scaled"],
        )

        scaler.save_json(
            output_paths["scaler"]
        )

        save_feature_schema(
            output_paths["schema"],
            profile=PROFILE,
        )

        raw_audits = {
            partition: audit_feature_matrix(
                raw_matrices[partition]
            )
            for partition in PARTITIONS
        }

        scaled_audits = {
            partition: audit_feature_matrix(
                scaled_matrices[partition]
            )
            for partition in PARTITIONS
        }

        build_audit = {
            "dataset_id": "DS-005",
            "artifact_type": (
                "Input A hand-engineered baseline"
            ),
            "profile": PROFILE,
            "feature_count": len(
                CORE_FEATURE_NAMES
            ),
            "feature_names": list(
                CORE_FEATURE_NAMES
            ),
            "split_unit": "fish",
            "normalization_fit_partition": "train",
            "fish_partition_counts": (
                ds.partition_counts()
            ),
            "total_valid_bouts": (
                ds.n_valid_bouts
            ),
            "total_feature_rows": total_rows,
            "raw": raw_audits,
            "scaled": scaled_audits,
            "train_scaling": (
                train_scaling_audit
            ),
            "integrity": {
                "no_fish_overlap": True,
                "all_valid_bouts_represented": (
                    total_rows
                    == ds.n_valid_bouts
                ),
                "raw_scaled_metadata_equal": True,
                "scaled_values_finite": True,
                "train_only_normalization": True,
            },
        }

        atomic_write_json(
            build_audit,
            output_paths["audit"],
        )

    # -------------------------------------------------------------------
    # Hash completed outputs
    # -------------------------------------------------------------------

    artifact_files = [
        output_paths["train_raw"],
        output_paths["validation_raw"],
        output_paths["test_raw"],
        output_paths["train_scaled"],
        output_paths["validation_scaled"],
        output_paths["test_scaled"],
        output_paths["scaler"],
        output_paths["schema"],
        output_paths["audit"],
    ]

    sums_path = write_sha256sums(
        output_dir,
        artifact_files,
    )

    hashes = {
        path.name: sha256_file(path)
        for path in artifact_files
    }

    print()
    print("BUILD COMPLETE")
    print("==============")
    print(
        f"Artifacts written to: {output_dir}"
    )
    print()

    for path in artifact_files:
        size_mb = (
            path.stat().st_size / (1024 ** 2)
        )
        print(
            f"  {path.name:30s} "
            f"{size_mb:10.2f} MiB"
        )

    print(
        f"  {sums_path.name:30s} "
        f"{sums_path.stat().st_size / 1024:10.2f} KiB"
    )

    print()
    print("SHA-256:")
    for name in sorted(hashes):
        print(
            f"  {hashes[name]}  {name}"
        )

    print()
    print(
        "Next: review build_audit_core.json and "
        "record these hashes before baseline clustering."
    )

    return {
        "output_dir": str(output_dir),
        "hashes": hashes,
        "sha256sums_path": str(sums_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build frozen DS-005 hand-engineered "
            "baseline feature matrices."
        )
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Repository root. Defaults to the "
            "parent of scripts/."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory. Defaults to "
            "data/processed/DS-005/baseline."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of existing baseline "
            "artifacts. Use only for an intentional rebuild."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (
            repo_root / output_dir
        )
    output_dir = output_dir.resolve()

    build_baseline_features(
        repo_root=repo_root,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
