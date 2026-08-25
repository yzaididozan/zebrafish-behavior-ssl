#!/usr/bin/env python3
"""Nonlinear probe: can handcrafted Input A reconstruct SSL cluster labels?

Purpose
-------
Follow up the linear-probe result from ``src/comparison/baseline_vs_ssl.py``.

Question:
    Are SSL k=8 cluster assignments recoverable from the 18 handcrafted
    baseline features by a stronger nonlinear model?

This script uses:
- TRAIN/VALIDATION frozen handcrafted baseline features
- TRAIN/VALIDATION SSL cluster labels from ssl_cluster_stability
- five frozen SSL encoder seeds

TEST is never loaded.

Model
-----
HistGradientBoostingClassifier

Why this model?
---------------
- nonlinear
- handles large tabular datasets efficiently
- no feature scaling requirement beyond the already-frozen baseline matrices
- provides a stronger sensitivity probe than multinomial logistic regression
- avoids introducing a deep model that would make interpretation unnecessarily
  complicated

Important guardrail
-------------------
This is a sensitivity analysis, not a new clustering model-selection stage.

High nonlinear-probe accuracy would imply that much of the SSL cluster
structure is recoverable from the handcrafted features, even if the frozen
baseline clustering itself does not recover it.

Low/modest nonlinear-probe accuracy would support the interpretation that
Input A does not fully capture the distinctions represented by SSL.

However, no finite probe can prove that no conceivable mapping from Input A
could reproduce SSL.

Outputs
-------
data/processed/DS-005/baseline_vs_ssl_nonlinear/
    seed11/
        nonlinear_probe.json
    seed23/
        ...
    aggregate_summary.json
    NONLINEAR_PROBE_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/comparison/baseline_vs_ssl_nonlinear.py

Intentional rerun:

    PYTHONPATH=. python3 src/comparison/baseline_vs_ssl_nonlinear.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)
DEFAULT_SSL_LABEL_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "baseline_vs_ssl_nonlinear"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_FEATURES = 18
EXPECTED_K = 8
PARTITIONS = ("train", "validation")
PROBE_SEED = 20260822


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(payload)
        tmp = handle.name

    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def write_checksums(path: Path, artifacts: Sequence[Path]) -> None:
    path.write_text(
        "".join(
            f"{sha256_file(artifact)}  {artifact.relative_to(path.parent)}\n"
            for artifact in artifacts
        ),
        encoding="utf-8",
    )


def prohibit_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise RuntimeError(
            f"TEST access prohibited during nonlinear probe: {path}"
        )


def assert_no_test_artifacts(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(root)

    hits: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        if (
            name.startswith("test_")
            or "_test_" in name
            or name in {"test.npy", "test.npz", "test.csv", "test.json"}
        ):
            hits.append(path)

    if hits:
        raise RuntimeError(
            "Protected TEST artifacts found under SSL label root; refusing "
            "nonlinear probe:\n"
            + "\n".join(str(path) for path in hits[:20])
        )


def configured_ssl_seeds(path: Path) -> Tuple[int, ...]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)

    training = obj.get("training", {})
    seeds = training.get("seeds", {}).get("values")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("No frozen SSL seeds found in training.yaml.")

    return tuple(int(seed) for seed in seeds)


def load_baseline_features(
    baseline_dir: Path,
    *,
    partition: str,
) -> Tuple[np.ndarray, List[str]]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    # Use frozen scaled Input-A matrices for consistency with linear probe.
    path = baseline_dir / f"{partition}_core_scaled.npz"
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as npz:
        if "X" not in npz.files or "feature_names" not in npz.files:
            raise RuntimeError(
                f"{path} must contain X and feature_names."
            )

        X = np.asarray(npz["X"], dtype=np.float32)
        feature_names = (
            np.asarray(npz["feature_names"])
            .astype(str)
            .tolist()
        )

    expected_shape = (
        EXPECTED_ROWS[partition],
        EXPECTED_FEATURES,
    )

    if X.shape != expected_shape:
        raise RuntimeError(
            f"{path}: expected {expected_shape}, got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"{path}: non-finite handcrafted features detected."
        )

    if len(feature_names) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_FEATURES} feature names, "
            f"got {len(feature_names)}"
        )

    return X, feature_names


def load_ssl_labels(
    label_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> np.ndarray:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = (
        label_root
        / f"seed{ssl_seed}"
        / f"{partition}_labels.npy"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.int64,
    )

    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(
            f"{path}: expected ({EXPECTED_ROWS[partition]},), "
            f"got {labels.shape}"
        )

    unique = np.unique(labels)

    if not np.array_equal(
        unique,
        np.arange(EXPECTED_K),
    ):
        raise RuntimeError(
            f"{path}: expected SSL labels 0..{EXPECTED_K - 1}, "
            f"got {unique.tolist()}"
        )

    return labels


def majority_accuracy(labels: np.ndarray) -> float:
    counts = np.bincount(
        labels.astype(int),
        minlength=EXPECTED_K,
    )

    return float(
        np.max(counts) / np.sum(counts)
    )


def classification_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                truth,
                pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                truth,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "majority_accuracy": (
            majority_accuracy(truth)
        ),
        "uniform_chance": float(
            1.0 / EXPECTED_K
        ),
    }


def fit_nonlinear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> Dict[str, Any]:
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=PROBE_SEED,
    )

    model.fit(
        train_x,
        train_y,
    )

    train_pred = model.predict(
        train_x
    )
    validation_pred = model.predict(
        validation_x
    )

    return {
        "model": "HistGradientBoostingClassifier",
        "purpose": (
            "Nonlinear sensitivity probe: predict SSL k=8 cluster labels "
            "from the same 18 handcrafted Input-A features."
        ),
        "hyperparameters": {
            "learning_rate": 0.08,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "max_depth": None,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 20,
            "random_state": PROBE_SEED,
        },
        "iterations_used": int(
            getattr(model, "n_iter_", -1)
        ),
        "train": classification_metrics(
            train_y,
            train_pred,
        ),
        "validation": classification_metrics(
            validation_y,
            validation_pred,
        ),
        "test_partition_used": False,
    }


def aggregate_results(
    per_seed: Mapping[int, Dict[str, Any]],
) -> Dict[str, Any]:
    metric_names = (
        "train_accuracy",
        "validation_accuracy",
        "train_balanced_accuracy",
        "validation_balanced_accuracy",
        "train_macro_f1",
        "validation_macro_f1",
    )

    values: Dict[str, List[float]] = {
        name: []
        for name in metric_names
    }

    for result in per_seed.values():
        probe = result["nonlinear_probe"]

        values["train_accuracy"].append(
            probe["train"]["accuracy"]
        )
        values["validation_accuracy"].append(
            probe["validation"]["accuracy"]
        )
        values["train_balanced_accuracy"].append(
            probe["train"]["balanced_accuracy"]
        )
        values["validation_balanced_accuracy"].append(
            probe["validation"]["balanced_accuracy"]
        )
        values["train_macro_f1"].append(
            probe["train"]["macro_f1"]
        )
        values["validation_macro_f1"].append(
            probe["validation"]["macro_f1"]
        )

    output: Dict[str, Any] = {}

    for name, vals in values.items():
        arr = np.asarray(
            vals,
            dtype=np.float64,
        )

        output[name] = {
            "mean": float(
                np.mean(arr)
            ),
            "std": float(
                np.std(arr)
            ),
            "min": float(
                np.min(arr)
            ),
            "max": float(
                np.max(arr)
            ),
        }

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nonlinear probe from frozen 18-feature handcrafted Input A "
            "to selected SSL k=8 cluster labels."
        )
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )

    parser.add_argument(
        "--ssl-label-root",
        type=Path,
        default=DEFAULT_SSL_LABEL_ROOT,
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_dir = (
        args.baseline_dir.resolve()
    )
    ssl_label_root = (
        args.ssl_label_root.resolve()
    )
    training_config = (
        args.training_config.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    for path in (
        baseline_dir,
        ssl_label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(path)

    assert_no_test_artifacts(
        ssl_label_root
    )

    seeds = configured_ssl_seeds(
        training_config
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_path = (
        output_dir
        / "aggregate_summary.json"
    )
    checksum_path = (
        output_dir
        / "NONLINEAR_PROBE_SHA256SUMS"
    )

    if (
        not args.overwrite
        and aggregate_path.exists()
    ):
        raise FileExistsError(
            f"{aggregate_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 NONLINEAR INPUT-A -> SSL PROBE")
    print("=" * 80)
    print("Input:        18 frozen handcrafted features")
    print("Target:       SSL KMeans k=8 cluster labels")
    print("Model:        HistGradientBoostingClassifier")
    print(f"SSL seeds:    {list(seeds)}")
    print("Fit:          TRAIN only")
    print("Evaluation:   TRAIN + VALIDATION")
    print("TEST:         PROTECTED / NOT LOADED")
    print()

    train_x, train_feature_names = (
        load_baseline_features(
            baseline_dir,
            partition="train",
        )
    )

    validation_x, validation_feature_names = (
        load_baseline_features(
            baseline_dir,
            partition="validation",
        )
    )

    if (
        train_feature_names
        != validation_feature_names
    ):
        raise RuntimeError(
            "TRAIN/VALIDATION feature-name mismatch."
        )

    print(
        f"TRAIN rows:      {train_x.shape[0]:,}"
    )
    print(
        f"VALIDATION rows: {validation_x.shape[0]:,}"
    )
    print(
        f"Features:        {train_x.shape[1]}"
    )
    print()

    per_seed: Dict[int, Dict[str, Any]] = {}
    written_artifacts: List[Path] = []

    for ssl_seed in seeds:
        print("=" * 80)
        print(f"SSL SEED {ssl_seed}")
        print("=" * 80)

        train_y = load_ssl_labels(
            ssl_label_root,
            ssl_seed=ssl_seed,
            partition="train",
        )

        validation_y = load_ssl_labels(
            ssl_label_root,
            ssl_seed=ssl_seed,
            partition="validation",
        )

        probe = fit_nonlinear_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
        )

        seed_result = {
            "ssl_seed": int(
                ssl_seed
            ),
            "nonlinear_probe": probe,
            "test_partition_used": False,
        }

        per_seed[
            ssl_seed
        ] = seed_result

        seed_dir = (
            output_dir
            / f"seed{ssl_seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            seed_dir
            / "nonlinear_probe.json"
        )

        atomic_write_json(
            output_path,
            {
                "ssl_seed": int(
                    ssl_seed
                ),
                "feature_names": (
                    train_feature_names
                ),
                **probe,
            },
        )

        written_artifacts.append(
            output_path
        )

        print(
            f"Iterations used:                  "
            f"{probe['iterations_used']}"
        )
        print(
            f"TRAIN balanced accuracy:          "
            f"{probe['train']['balanced_accuracy']:.6f}"
        )
        print(
            f"VALIDATION balanced accuracy:     "
            f"{probe['validation']['balanced_accuracy']:.6f}"
        )
        print(
            f"VALIDATION macro F1:              "
            f"{probe['validation']['macro_f1']:.6f}"
        )
        print(
            f"VALIDATION ordinary accuracy:     "
            f"{probe['validation']['accuracy']:.6f}"
        )
        print(
            f"VALIDATION majority baseline:     "
            f"{probe['validation']['majority_accuracy']:.6f}"
        )
        print(
            f"Uniform chance:                   "
            f"{probe['validation']['uniform_chance']:.6f}"
        )
        print(
            "TEST partition used: NO"
        )
        print()

    aggregate = aggregate_results(
        per_seed
    )

    aggregate_payload = {
        "dataset_id": "DS-005",
        "input_a": (
            "18 frozen handcrafted core features"
        ),
        "target": (
            "selected SSL KMeans k=8 cluster labels"
        ),
        "model": (
            "HistGradientBoostingClassifier"
        ),
        "ssl_training_seeds": list(
            seeds
        ),
        "aggregate_metrics": aggregate,
        "per_seed": {
            str(seed): result
            for seed, result
            in per_seed.items()
        },
        "interpretation_guardrails": {
            "purpose": (
                "Sensitivity test for nonlinear recoverability of SSL "
                "cluster membership from Input A."
            ),
            "high_performance": (
                "Would indicate SSL cluster distinctions are substantially "
                "recoverable from handcrafted features."
            ),
            "low_or_moderate_performance": (
                "Would support incomplete recoverability from Input A, but "
                "would not prove no other nonlinear mapping could do better."
            ),
            "not_model_selection": True,
            "test_partition_used": False,
        },
        "test_partition_used": False,
    }

    atomic_write_json(
        aggregate_path,
        aggregate_payload,
    )

    written_artifacts.append(
        aggregate_path
    )

    write_checksums(
        checksum_path,
        written_artifacts,
    )

    print("=" * 80)
    print("NONLINEAR PROBE SUMMARY")
    print("=" * 80)
    print(
        "Mean TRAIN balanced accuracy:      "
        f"{aggregate['train_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION balanced accuracy: "
        f"{aggregate['validation_balanced_accuracy']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION macro F1:          "
        f"{aggregate['validation_macro_f1']['mean']:.6f}"
    )
    print(
        "Mean VALIDATION accuracy:          "
        f"{aggregate['validation_accuracy']['mean']:.6f}"
    )
    print()
    print(
        f"Uniform chance for k=8:            "
        f"{1.0 / EXPECTED_K:.6f}"
    )
    print("TEST partition used: NO")
    print(f"Aggregate:  {aggregate_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
