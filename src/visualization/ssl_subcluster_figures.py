#!/usr/bin/env python3
"""Publication-style SSL subcluster figures for DS-005.

Purpose
-------
Visualize the strongest candidate within-class SSL subdivisions using
interpretable handcrafted movement features rather than relying on UMAP/t-SNE.

Default classes
---------------
BS, Long_CS, LLC

Default feature choices
-----------------------
BS:
    speed_mean, speed_rms, speed_std

Long_CS:
    bout_duration_s, accel_rms, accel_abs_std

LLC:
    speed_max, speed_p95, turn_net_rad

The script generates separate TRAIN and VALIDATION figures so the same visual
pattern can be inspected on held-out fish.

Figures per class / partition
-----------------------------
1. subcluster_occupancy.png
2. feature_distributions_<feature>.png   (boxplots)
3. top2_feature_scatter.png

TEST is never loaded.

Inputs
------
- Frozen baseline raw features:
    data/processed/DS-005/baseline/{train,validation}_core_raw.npz
- SSL metadata:
    data/processed/DS-005/ssl/seed*/{train,validation}_metadata.csv
- SSL cluster labels:
    data/processed/DS-005/ssl_cluster_stability/seed*/{train,validation}_labels.npy
- Frozen SSL seed list:
    configs/ssl/training.yaml

Cluster alignment
-----------------
Seed 11 is used as the deterministic reference numbering.
Mappings for seeds 23/37/51/79 are estimated using TRAIN assignments only.
Those mappings are then applied unchanged to VALIDATION.

Default visualization seed
--------------------------
Seed 11.

This keeps the main figures simple and avoids mixing incompatible per-bout
cluster assignments from different independently trained encoders. Cross-seed
reproducibility is already quantified separately by the validation scripts.

Optional:
    --seed 23
can be used for a sensitivity figure set.

Output
------
results/figures/subclusters/
    BS/
        train/
        validation/
    Long_CS/
        train/
        validation/
    LLC/
        train/
        validation/
    figure_manifest.json

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/visualization/ssl_subcluster_figures.py

Optional classes:

    PYTHONPATH=. python3 src/visualization/ssl_subcluster_figures.py \
        --classes BS Long_CS LLC

Optional seed:

    PYTHONPATH=. python3 src/visualization/ssl_subcluster_figures.py --seed 11

Intentional rerun:

    PYTHONPATH=. python3 src/visualization/ssl_subcluster_figures.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)
DEFAULT_METADATA_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_LABEL_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "DS-005"
    / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "figures" / "subclusters"
)

EXPECTED_ROWS = {
    "train": 842_841,
    "validation": 168_464,
}
EXPECTED_FEATURES = 18
EXPECTED_SSL_K = 8
REFERENCE_SEED = 11
PARTITIONS = ("train", "validation")

DEFAULT_CLASSES = ("BS", "Long_CS", "LLC")

DEFAULT_FEATURE_MAP = {
    "BS": (
        "speed_mean",
        "speed_rms",
        "speed_std",
    ),
    "Long_CS": (
        "bout_duration_s",
        "accel_rms",
        "accel_abs_std",
    ),
    "LLC": (
        "speed_max",
        "speed_p95",
        "turn_net_rad",
    ),
}

DEFAULT_CLASS_NAMES = [
    "Short_CS",
    "Long_CS",
    "BS",
    "O_bend",
    "J_turn",
    "SLC",
    "Slow1",
    "RT",
    "Slow2",
    "LLC",
    "AS",
    "SAT",
    "HAT",
]

LABEL_COLUMN_CANDIDATES = (
    "bout_type",
    "bout_type_name",
    "known_bout_type",
    "known_behavior",
    "behavior_label",
    "bout_label",
)


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


def prohibit_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise RuntimeError(
            f"TEST access prohibited during subcluster visualization: {path}"
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
            "Protected TEST artifacts found beneath an SSL input root; "
            "refusing to continue:\n"
            + "\n".join(str(path) for path in hits[:20])
        )


def configured_ssl_seeds(path: Path) -> Tuple[int, ...]:
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle)

    seeds = obj.get("training", {}).get("seeds", {}).get("values")

    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError("No frozen SSL seeds found in training.yaml.")

    return tuple(int(seed) for seed in seeds)


def detect_label_column(
    fieldnames: Sequence[str],
    requested: Optional[str],
) -> str:
    available = set(fieldnames)

    if requested is not None:
        if requested not in available:
            raise RuntimeError(
                f"Requested label column {requested!r} not found. "
                f"Available: {sorted(available)}"
            )
        return requested

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate

    raise RuntimeError(
        "Could not locate known-behavior label column. "
        f"Tried {list(LABEL_COLUMN_CANDIDATES)}."
    )


def normalize_known_label(raw: str) -> str:
    value = raw.strip()

    if value == "":
        return "__MISSING__"

    if value in DEFAULT_CLASS_NAMES:
        return value

    try:
        numeric = float(value)
        integer = int(numeric)

        if math.isclose(numeric, integer):
            if 0 <= integer < len(DEFAULT_CLASS_NAMES):
                return DEFAULT_CLASS_NAMES[integer]

            if 1 <= integer <= len(DEFAULT_CLASS_NAMES):
                return DEFAULT_CLASS_NAMES[integer - 1]
    except ValueError:
        pass

    return value


def load_baseline_raw(
    baseline_dir: Path,
    *,
    partition: str,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    path = baseline_dir / f"{partition}_core_raw.npz"
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as npz:
        required = {
            "X",
            "feature_names",
            "fish_id",
            "session_id",
            "bout_index",
            "partition",
            "context_id",
        }

        missing = required - set(npz.files)

        if missing:
            raise RuntimeError(
                f"{path} missing arrays: {sorted(missing)}"
            )

        X = np.asarray(npz["X"], dtype=np.float64)
        feature_names = np.asarray(
            npz["feature_names"]
        ).astype(str).tolist()

        alignment = {
            "fish_id": np.asarray(npz["fish_id"]).astype(str),
            "session_id": np.asarray(npz["session_id"]).astype(str),
            "bout_index": np.asarray(npz["bout_index"]).astype(str),
            "partition": np.asarray(npz["partition"]).astype(str),
            "context_id": np.asarray(npz["context_id"]).astype(str),
        }

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
            f"{path}: non-finite handcrafted features."
        )

    return X, feature_names, alignment


def load_metadata(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
    requested_label_column: Optional[str],
) -> Tuple[Dict[str, np.ndarray], str]:
    path = (
        metadata_root
        / f"seed{ssl_seed}"
        / f"{partition}_metadata.csv"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    required_values: Dict[str, List[str]] = {
        "fish_id": [],
        "session_id": [],
        "bout_index": [],
        "partition": [],
        "context_id": [],
        "bout_id": [],
        "known_label": [],
    }

    numeric_optional_names = {
        "speed_mean",
        "speed_std",
        "speed_max",
        "speed_rms",
    }
    numeric_optional: Dict[str, List[float]] = {
        key: []
        for key in numeric_optional_names
    }

    detected_label_column: Optional[str] = None

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: missing header.")

        detected_label_column = detect_label_column(
            reader.fieldnames,
            requested_label_column,
        )

        required = {
            "row_index",
            "fish_id",
            "session_id",
            "bout_index",
            "partition",
            "context_id",
            "bout_id",
            "training_seed",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise RuntimeError(
                f"{path} missing metadata fields: {sorted(missing)}"
            )

        available = set(reader.fieldnames)

        for expected_row, row in enumerate(reader):
            if int(row["row_index"]) != expected_row:
                raise RuntimeError(
                    f"{path}: row_index mismatch at row {expected_row}."
                )

            if row["partition"] != partition:
                raise RuntimeError(
                    f"{path}: partition mismatch at row {expected_row}."
                )

            if int(row["training_seed"]) != ssl_seed:
                raise RuntimeError(
                    f"{path}: seed mismatch at row {expected_row}."
                )

            required_values["fish_id"].append(row["fish_id"])
            required_values["session_id"].append(row["session_id"])
            required_values["bout_index"].append(
                str(int(row["bout_index"]))
            )
            required_values["partition"].append(row["partition"])
            required_values["context_id"].append(row["context_id"])
            required_values["bout_id"].append(row["bout_id"])
            required_values["known_label"].append(
                normalize_known_label(
                    row[detected_label_column]
                )
            )

            for name in numeric_optional_names:
                if (
                    name in available
                    and row[name].strip() != ""
                ):
                    numeric_optional[name].append(
                        float(row[name])
                    )
                else:
                    numeric_optional[name].append(
                        float("nan")
                    )

    if len(required_values["fish_id"]) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"got {len(required_values['fish_id']):,}"
        )

    output: Dict[str, np.ndarray] = {
        key: np.asarray(values, dtype=str)
        for key, values
        in required_values.items()
    }

    for key, values in numeric_optional.items():
        output[key] = np.asarray(
            values,
            dtype=np.float64,
        )

    return output, detected_label_column


def verify_alignment(
    baseline_alignment: Mapping[str, np.ndarray],
    metadata: Mapping[str, np.ndarray],
    *,
    seed: int,
    partition: str,
) -> None:
    for field in (
        "fish_id",
        "session_id",
        "bout_index",
        "partition",
        "context_id",
    ):
        left = np.asarray(
            baseline_alignment[field]
        ).astype(str)

        right = np.asarray(
            metadata[field]
        ).astype(str)

        if not np.array_equal(left, right):
            mismatch = np.flatnonzero(left != right)

            idx = (
                int(mismatch[0])
                if mismatch.size
                else -1
            )

            raise RuntimeError(
                f"Alignment failed: seed={seed}, partition={partition}, "
                f"field={field}, first mismatch row={idx}."
            )


def load_ssl_labels(
    label_root: Path,
    *,
    seed: int,
    partition: str,
) -> np.ndarray:
    path = (
        label_root
        / f"seed{seed}"
        / f"{partition}_labels.npy"
    )
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(
        np.load(path, allow_pickle=False),
        dtype=np.int64,
    )

    if labels.shape != (
        EXPECTED_ROWS[partition],
    ):
        raise RuntimeError(
            f"{path}: invalid label shape {labels.shape}."
        )

    return labels


def hungarian_map_to_reference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> Dict[int, int]:
    table = np.zeros(
        (EXPECTED_SSL_K, EXPECTED_SSL_K),
        dtype=np.int64,
    )

    np.add.at(
        table,
        (
            reference_labels.astype(int),
            candidate_labels.astype(int),
        ),
        1,
    )

    row_ind, col_ind = linear_sum_assignment(-table)

    return {
        int(candidate): int(reference)
        for reference, candidate
        in zip(row_ind, col_ind)
    }


def apply_mapping(
    labels: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    return np.asarray(
        [
            mapping[int(label)]
            for label in labels
        ],
        dtype=np.int16,
    )


def feature_vector(
    name: str,
    *,
    X: np.ndarray,
    feature_names: Sequence[str],
    metadata: Mapping[str, np.ndarray],
) -> np.ndarray:
    if name in feature_names:
        idx = feature_names.index(name)

        return np.asarray(
            X[:, idx],
            dtype=np.float64,
        )

    if name in metadata:
        values = np.asarray(
            metadata[name]
        )

        if np.issubdtype(
            values.dtype,
            np.number,
        ):
            return values.astype(
                np.float64
            )

    raise RuntimeError(
        f"Requested plotting feature {name!r} is not available."
    )


def safe_sample_indices(
    indices: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if indices.size <= max_points:
        return indices

    rng = np.random.default_rng(seed)

    return np.sort(
        rng.choice(
            indices,
            size=max_points,
            replace=False,
        )
    )


def save_occupancy_plot(
    path: Path,
    *,
    labels: np.ndarray,
    class_name: str,
    partition: str,
    seed: int,
) -> None:
    counts = np.bincount(
        labels.astype(int),
        minlength=EXPECTED_SSL_K,
    )

    fractions = counts / np.sum(counts)

    fig, ax = plt.subplots(
        figsize=(8.0, 5.0)
    )

    x = np.arange(
        EXPECTED_SSL_K
    )

    ax.bar(
        x,
        fractions,
    )

    ax.set_xlabel(
        "Aligned SSL subcluster"
    )
    ax.set_ylabel(
        "Fraction of bouts"
    )
    ax.set_title(
        f"{class_name}: SSL subcluster occupancy ({partition.upper()}, seed {seed})"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(i) for i in x]
    )

    ax.set_ylim(
        0,
        max(
            0.05,
            float(np.max(fractions)) * 1.15,
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_boxplot(
    path: Path,
    *,
    values: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
    class_name: str,
    partition: str,
    seed: int,
) -> None:
    groups = []
    group_labels = []

    for cluster in range(
        EXPECTED_SSL_K
    ):
        subset = values[
            labels == cluster
        ]
        subset = subset[
            np.isfinite(subset)
        ]

        if subset.size == 0:
            continue

        groups.append(subset)
        group_labels.append(
            str(cluster)
        )

    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    ax.boxplot(
        groups,
        tick_labels=group_labels,
        showfliers=False,
        whis=(5, 95),
    )

    ax.set_xlabel(
        "Aligned SSL subcluster"
    )
    ax.set_ylabel(
        feature_name
    )
    ax.set_title(
        f"{class_name}: {feature_name} by SSL subcluster "
        f"({partition.upper()}, seed {seed})"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_scatter(
    path: Path,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: np.ndarray,
    x_name: str,
    y_name: str,
    class_name: str,
    partition: str,
    seed: int,
    max_points: int,
) -> int:
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
    )

    valid_indices = np.flatnonzero(
        valid
    )

    sampled = safe_sample_indices(
        valid_indices,
        max_points=max_points,
        seed=20260822 + seed,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6.0)
    )

    for cluster in range(
        EXPECTED_SSL_K
    ):
        cluster_indices = sampled[
            labels[sampled] == cluster
        ]

        if cluster_indices.size == 0:
            continue

        ax.scatter(
            x_values[cluster_indices],
            y_values[cluster_indices],
            s=9,
            alpha=0.32,
            label=f"SSL {cluster}",
        )

    ax.set_xlabel(
        x_name
    )
    ax.set_ylabel(
        y_name
    )
    ax.set_title(
        f"{class_name}: top-feature view "
        f"({partition.upper()}, seed {seed})"
    )

    ax.legend(
        title="Subcluster",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        ncol=2,
    )

    ax.grid(
        alpha=0.18,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    return int(
        sampled.size
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-style interpretable feature figures for "
            "candidate DS-005 SSL within-class substructure."
        )
    )

    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )

    parser.add_argument(
        "--label-root",
        type=Path,
        default=DEFAULT_LABEL_ROOT,
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
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=REFERENCE_SEED,
        help="SSL encoder/clustering seed used for plotted bout assignments.",
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=20_000,
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
    metadata_root = (
        args.metadata_root.resolve()
    )
    label_root = (
        args.label_root.resolve()
    )
    training_config = (
        args.training_config.resolve()
    )
    output_dir = (
        args.output_dir.resolve()
    )

    for path in (
        baseline_dir,
        metadata_root,
        label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(path)

    assert_no_test_artifacts(
        metadata_root
    )
    assert_no_test_artifacts(
        label_root
    )

    seeds = configured_ssl_seeds(
        training_config
    )

    if REFERENCE_SEED not in seeds:
        raise RuntimeError(
            f"Reference seed {REFERENCE_SEED} missing."
        )

    if args.seed not in seeds:
        raise RuntimeError(
            f"--seed {args.seed} is not one of frozen SSL seeds {list(seeds)}."
        )

    unknown_classes = [
        value
        for value in args.classes
        if value not in DEFAULT_FEATURE_MAP
    ]

    if unknown_classes:
        raise RuntimeError(
            "No preregistered/default feature map for: "
            + ", ".join(unknown_classes)
            + ". Add them to DEFAULT_FEATURE_MAP before plotting."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "figure_manifest.json"
    )

    if (
        manifest_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{manifest_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 SSL SUBCLUSTER FIGURES")
    print("=" * 80)
    print(
        f"Classes:          {args.classes}"
    )
    print(
        f"Visualization seed: {args.seed}"
    )
    print(
        f"Reference seed:   {REFERENCE_SEED}"
    )
    print(
        "Alignment:        TRAIN-derived Hungarian mapping"
    )
    print(
        "Partitions:       TRAIN + VALIDATION"
    )
    print(
        "TEST:             PROTECTED / NOT LOADED"
    )
    print()

    baseline_x: Dict[
        str,
        np.ndarray,
    ] = {}

    baseline_alignment: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    metadata_by_seed: Dict[
        int,
        Dict[str, Dict[str, np.ndarray]],
    ] = {}

    labels_by_seed: Dict[
        int,
        Dict[str, np.ndarray],
    ] = {}

    feature_names: Optional[
        List[str]
    ] = None

    reference_known: Dict[
        str,
        np.ndarray,
    ] = {}

    reference_bouts: Dict[
        str,
        np.ndarray,
    ] = {}

    for partition in PARTITIONS:
        X, names, alignment = (
            load_baseline_raw(
                baseline_dir,
                partition=partition,
            )
        )

        baseline_x[
            partition
        ] = X

        baseline_alignment[
            partition
        ] = alignment

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError(
                "TRAIN/VALIDATION feature names differ."
            )

    assert feature_names is not None

    # Load all seeds because TRAIN alignment requires the reference seed and the
    # plotted seed; loading the full frozen set also verifies metadata ordering.
    for seed in seeds:
        metadata_by_seed[
            seed
        ] = {}

        labels_by_seed[
            seed
        ] = {}

        for partition in PARTITIONS:
            metadata, _ = (
                load_metadata(
                    metadata_root,
                    ssl_seed=seed,
                    partition=partition,
                    requested_label_column=(
                        args.label_column
                    ),
                )
            )

            verify_alignment(
                baseline_alignment[
                    partition
                ],
                metadata,
                seed=seed,
                partition=partition,
            )

            labels = load_ssl_labels(
                label_root,
                seed=seed,
                partition=partition,
            )

            metadata_by_seed[
                seed
            ][
                partition
            ] = metadata

            labels_by_seed[
                seed
            ][
                partition
            ] = labels

            known = metadata[
                "known_label"
            ]
            bouts = metadata[
                "bout_id"
            ]

            if (
                partition
                not in reference_known
            ):
                reference_known[
                    partition
                ] = known.copy()

                reference_bouts[
                    partition
                ] = bouts.copy()
            else:
                if not np.array_equal(
                    reference_known[
                        partition
                    ],
                    known,
                ):
                    raise RuntimeError(
                        f"Known labels differ across seeds for {partition}."
                    )

                if not np.array_equal(
                    reference_bouts[
                        partition
                    ],
                    bouts,
                ):
                    raise RuntimeError(
                        f"Bout ordering differs across seeds for {partition}."
                    )

    reference_train = labels_by_seed[
        REFERENCE_SEED
    ][
        "train"
    ]

    mappings: Dict[
        int,
        Dict[int, int],
    ] = {
        REFERENCE_SEED: {
            cluster: cluster
            for cluster in range(
                EXPECTED_SSL_K
            )
        }
    }

    for seed in seeds:
        if seed == REFERENCE_SEED:
            continue

        mappings[
            seed
        ] = hungarian_map_to_reference(
            reference_train,
            labels_by_seed[
                seed
            ][
                "train"
            ],
        )

    aligned_labels = {
        partition: apply_mapping(
            labels_by_seed[
                args.seed
            ][
                partition
            ],
            mappings[
                args.seed
            ],
        )
        for partition in PARTITIONS
    }

    manifest: Dict[str, Any] = {
        "dataset_id": "DS-005",
        "analysis": "ssl_subcluster_figures",
        "classes": list(
            args.classes
        ),
        "visualization_seed": int(
            args.seed
        ),
        "reference_seed": int(
            REFERENCE_SEED
        ),
        "cluster_mapping_to_reference": {
            str(src): int(dst)
            for src, dst
            in mappings[
                args.seed
            ].items()
        },
        "figure_design": (
            "interpretable feature distributions, occupancy, and top-two-feature "
            "scatter; no UMAP/t-SNE used as primary evidence"
        ),
        "partitions": {},
        "test_partition_used": False,
    }

    generated_paths: List[
        Path
    ] = []

    for class_name in args.classes:
        print(
            "=" * 80
        )
        print(
            class_name
        )
        print(
            "=" * 80
        )

        features = (
            DEFAULT_FEATURE_MAP[
                class_name
            ]
        )

        manifest[
            "partitions"
        ][
            class_name
        ] = {}

        for partition in PARTITIONS:
            mask = (
                reference_known[
                    partition
                ]
                == class_name
            )

            indices = np.flatnonzero(
                mask
            )

            if indices.size == 0:
                raise RuntimeError(
                    f"No {partition} bouts for {class_name}."
                )

            class_labels = (
                aligned_labels[
                    partition
                ][
                    indices
                ]
            )

            class_dir = (
                output_dir
                / class_name
                / partition
            )

            class_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            occupancy_path = (
                class_dir
                / "subcluster_occupancy.png"
            )

            save_occupancy_plot(
                occupancy_path,
                labels=class_labels,
                class_name=class_name,
                partition=partition,
                seed=args.seed,
            )

            generated_paths.append(
                occupancy_path
            )

            feature_data: Dict[
                str,
                np.ndarray,
            ] = {}

            for feature_name in features:
                full_values = feature_vector(
                    feature_name,
                    X=baseline_x[
                        partition
                    ],
                    feature_names=(
                        feature_names
                    ),
                    metadata=(
                        metadata_by_seed[
                            args.seed
                        ][
                            partition
                        ]
                    ),
                )

                values = full_values[
                    indices
                ]

                feature_data[
                    feature_name
                ] = values

                plot_path = (
                    class_dir
                    / (
                        "feature_distributions_"
                        + feature_name
                        + ".png"
                    )
                )

                save_boxplot(
                    plot_path,
                    values=values,
                    labels=class_labels,
                    feature_name=feature_name,
                    class_name=class_name,
                    partition=partition,
                    seed=args.seed,
                )

                generated_paths.append(
                    plot_path
                )

            scatter_path = (
                class_dir
                / "top2_feature_scatter.png"
            )

            scatter_points = (
                save_scatter(
                    scatter_path,
                    x_values=feature_data[
                        features[0]
                    ],
                    y_values=feature_data[
                        features[1]
                    ],
                    labels=class_labels,
                    x_name=features[0],
                    y_name=features[1],
                    class_name=class_name,
                    partition=partition,
                    seed=args.seed,
                    max_points=(
                        args.scatter_max_points
                    ),
                )
            )

            generated_paths.append(
                scatter_path
            )

            counts = np.bincount(
                class_labels.astype(
                    int
                ),
                minlength=EXPECTED_SSL_K,
            )

            manifest[
                "partitions"
            ][
                class_name
            ][
                partition
            ] = {
                "bout_count": int(
                    indices.size
                ),
                "feature_names": list(
                    features
                ),
                "subcluster_counts": (
                    counts.astype(
                        int
                    ).tolist()
                ),
                "scatter_points_plotted": int(
                    scatter_points
                ),
                "files": [
                    str(
                        occupancy_path.relative_to(
                            output_dir
                        )
                    ),
                    *[
                        str(
                            (
                                class_dir
                                / (
                                    "feature_distributions_"
                                    + feature_name
                                    + ".png"
                                )
                            ).relative_to(
                                output_dir
                            )
                        )
                        for feature_name
                        in features
                    ],
                    str(
                        scatter_path.relative_to(
                            output_dir
                        )
                    ),
                ],
            }

            print(
                f"{partition.upper():<11} "
                f"n={indices.size:,}  "
                f"features={list(features)}"
            )

        print()

    checksums = {
        str(path.relative_to(output_dir)): (
            sha256_file(path)
        )
        for path in generated_paths
    }

    manifest[
        "sha256"
    ] = checksums

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print(
        "=" * 80
    )
    print(
        "FIGURE GENERATION COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        f"Output root: {output_dir}"
    )

    print(
        f"Figures:     {len(generated_paths)}"
    )

    print(
        f"Manifest:    {manifest_path}"
    )

    print(
        "TEST partition used: NO"
    )


if __name__ == "__main__":
    main()
