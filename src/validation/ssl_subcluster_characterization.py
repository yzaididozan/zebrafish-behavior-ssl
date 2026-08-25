#!/usr/bin/env python3
"""Characterize reproducible SSL subclusters within known DS-005 behavior classes.

Purpose
-------
Follow up ``ssl_within_class_substructure.py`` by asking:

    What movement or experimental variables distinguish the reproducible SSL
    subclusters found inside conventional bout classes?

Default candidate classes
-------------------------
BS, Long_CS, HAT, SLC, Slow2, LLC

These are the strongest/most interesting candidates from the preceding
within-class analysis. A custom subset can be provided with ``--classes``.

Inputs
------
- Frozen handcrafted baseline matrices:
    data/processed/DS-005/baseline/train_core_raw.npz
    data/processed/DS-005/baseline/validation_core_raw.npz
- SSL cluster labels:
    data/processed/DS-005/ssl_cluster_stability/seed*/{partition}_labels.npy
- SSL metadata:
    data/processed/DS-005/ssl/seed*/{partition}_metadata.csv
- Frozen SSL seeds:
    configs/ssl/training.yaml

TRAIN and VALIDATION only. TEST is never loaded.

Analyses
--------
For each known behavior class and SSL seed:
1. Select only bouts with that known class.
2. Characterize every occupied SSL subcluster using:
   - all 18 handcrafted features
   - speed_mean, speed_std, speed_max, speed_rms if available
   - stimulus_code if available
   - context_id / context_name if available
3. For continuous variables:
   - per-subcluster count, mean, std, median, IQR
   - eta-squared: fraction of variable variance associated with SSL subcluster
   - standardized range of subcluster means
4. For categorical variables:
   - normalized mutual information (NMI)
   - adjusted mutual information (AMI)
   - Cramer's V
5. Rank continuous variables by VALIDATION eta-squared averaged across SSL seeds.
6. Rank categorical variables by VALIDATION NMI averaged across SSL seeds.
7. Report TRAIN-to-VALIDATION consistency.

Important guardrails
--------------------
- This is post-hoc characterization, not model selection.
- Large effects identify variables associated with SSL subclusters; they do not
  prove that the variable causes the substructure.
- If speed dominates, the subdivision may largely reflect movement intensity.
- If multiple pose/turning variables distinguish subclusters while context and
  identity remain weak, that is more supportive of behavioral substructure.
- Reproducible characterization still does NOT establish a new biological
  behavior without external validation.

Outputs
-------
data/processed/DS-005/ssl_subcluster_characterization/
    per_class/
        BS.json
        Long_CS.json
        HAT.json
        SLC.json
        Slow2.json
        LLC.json
    summary.json
    SUBCLUSTER_CHARACTERIZATION_SHA256SUMS

Usage
-----
From repository root:

    PYTHONPATH=. python3 src/validation/ssl_subcluster_characterization.py

Optional subset:

    PYTHONPATH=. python3 src/validation/ssl_subcluster_characterization.py \
        --classes BS Long_CS

Intentional rerun:

    PYTHONPATH=. python3 src/validation/ssl_subcluster_characterization.py --overwrite
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

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASELINE_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "baseline"
)
DEFAULT_METADATA_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl"
)
DEFAULT_LABEL_ROOT = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_cluster_stability"
)
DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "ssl" / "training.yaml"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "data" / "processed" / "DS-005" / "ssl_subcluster_characterization"
)

EXPECTED_ROWS = {"train": 842_841, "validation": 168_464}
EXPECTED_FEATURES = 18
EXPECTED_SSL_K = 8
REFERENCE_SEED = 11
PARTITIONS = ("train", "validation")

DEFAULT_CLASSES = ("BS", "Long_CS", "HAT", "SLC", "Slow2", "LLC")

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

CONTINUOUS_METADATA_CANDIDATES = (
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_rms",
)

CATEGORICAL_METADATA_CANDIDATES = (
    "stimulus_code",
    "context_id",
    "context_name",
)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
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
            f"TEST access prohibited during subcluster characterization: {path}"
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
                f"Requested label column {requested!r} not present. "
                f"Available columns: {sorted(available)}"
            )
        return requested

    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in available:
            return candidate

    raise RuntimeError(
        "Could not find known-behavior label column. "
        f"Tried: {list(LABEL_COLUMN_CANDIDATES)}"
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
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

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
        feature_names = np.asarray(npz["feature_names"]).astype(str).tolist()

        alignment = {
            "fish_id": np.asarray(npz["fish_id"]).astype(str),
            "session_id": np.asarray(npz["session_id"]).astype(str),
            "bout_index": np.asarray(npz["bout_index"]).astype(str),
            "partition": np.asarray(npz["partition"]).astype(str),
            "context_id": np.asarray(npz["context_id"]).astype(str),
        }

    if X.shape != (EXPECTED_ROWS[partition], EXPECTED_FEATURES):
        raise RuntimeError(
            f"{path}: expected {(EXPECTED_ROWS[partition], EXPECTED_FEATURES)}, "
            f"got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(f"{path}: non-finite baseline features detected.")

    if len(feature_names) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_FEATURES} features, "
            f"got {len(feature_names)}."
        )

    return X, feature_names, alignment


def load_metadata(
    metadata_root: Path,
    *,
    ssl_seed: int,
    partition: str,
    requested_label_column: Optional[str],
) -> Tuple[Dict[str, np.ndarray], str]:
    if partition not in PARTITIONS:
        raise RuntimeError("Only TRAIN and VALIDATION are permitted.")

    path = metadata_root / f"seed{ssl_seed}" / f"{partition}_metadata.csv"
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    rows: Dict[str, List[Any]] = {
        "fish_id": [],
        "session_id": [],
        "bout_index": [],
        "partition": [],
        "context_id": [],
        "context_name": [],
        "bout_id": [],
        "known_label": [],
    }

    optional_continuous: Dict[str, List[float]] = {
        key: [] for key in CONTINUOUS_METADATA_CANDIDATES
    }
    optional_categorical: Dict[str, List[str]] = {
        key: [] for key in CATEGORICAL_METADATA_CANDIDATES
    }

    detected_label_column: Optional[str] = None

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no CSV header.")

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
                f"{path} missing required metadata columns: {sorted(missing)}"
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
                    f"{path}: training_seed mismatch at row {expected_row}."
                )

            rows["fish_id"].append(row["fish_id"])
            rows["session_id"].append(row["session_id"])
            rows["bout_index"].append(str(int(row["bout_index"])))
            rows["partition"].append(row["partition"])
            rows["context_id"].append(row["context_id"])
            rows["context_name"].append(
                row["context_name"] if "context_name" in available else "__MISSING__"
            )
            rows["bout_id"].append(row["bout_id"])
            rows["known_label"].append(
                normalize_known_label(row[detected_label_column])
            )

            for key in CONTINUOUS_METADATA_CANDIDATES:
                if key in available and row[key].strip() != "":
                    value = float(row[key])
                    optional_continuous[key].append(value)
                else:
                    optional_continuous[key].append(float("nan"))

            for key in CATEGORICAL_METADATA_CANDIDATES:
                if key in available:
                    value = row[key].strip() or "__MISSING__"
                else:
                    value = "__MISSING__"
                optional_categorical[key].append(value)

    if len(rows["fish_id"]) != EXPECTED_ROWS[partition]:
        raise RuntimeError(
            f"{path}: expected {EXPECTED_ROWS[partition]:,} rows, "
            f"got {len(rows['fish_id']):,}"
        )

    output: Dict[str, np.ndarray] = {
        key: np.asarray(value, dtype=str)
        for key, value in rows.items()
    }

    for key, value in optional_continuous.items():
        output[key] = np.asarray(value, dtype=np.float64)

    for key, value in optional_categorical.items():
        output[key] = np.asarray(value, dtype=str)

    return output, detected_label_column


def verify_alignment(
    baseline_alignment: Mapping[str, np.ndarray],
    metadata: Mapping[str, np.ndarray],
    *,
    partition: str,
    ssl_seed: int,
) -> None:
    for field in ("fish_id", "session_id", "bout_index", "partition", "context_id"):
        a = np.asarray(baseline_alignment[field]).astype(str)
        b = np.asarray(metadata[field]).astype(str)

        if a.shape != b.shape:
            raise RuntimeError(
                f"Alignment shape mismatch seed={ssl_seed}, "
                f"partition={partition}, field={field}."
            )

        unequal = np.flatnonzero(a != b)

        if unequal.size:
            idx = int(unequal[0])
            raise RuntimeError(
                f"Input alignment FAILED seed={ssl_seed}, partition={partition}, "
                f"field={field}, row={idx}: {a[idx]!r} != {b[idx]!r}"
            )


def load_ssl_labels(
    label_root: Path,
    *,
    ssl_seed: int,
    partition: str,
) -> np.ndarray:
    path = label_root / f"seed{ssl_seed}" / f"{partition}_labels.npy"
    prohibit_test_path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    labels = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)

    if labels.shape != (EXPECTED_ROWS[partition],):
        raise RuntimeError(
            f"{path}: expected ({EXPECTED_ROWS[partition]},), got {labels.shape}"
        )

    return labels


def hungarian_map_to_reference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> Dict[int, int]:
    table = np.zeros((EXPECTED_SSL_K, EXPECTED_SSL_K), dtype=np.int64)

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
        for reference, candidate in zip(row_ind, col_ind)
    }


def apply_mapping(
    labels: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    return np.asarray(
        [mapping[int(label)] for label in labels],
        dtype=np.int16,
    )


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values)
    values = values[mask]
    labels = labels[mask]

    if values.size == 0:
        return 0.0

    grand_mean = float(np.mean(values))
    ss_total = float(np.sum((values - grand_mean) ** 2))

    if ss_total <= 0:
        return 0.0

    ss_between = 0.0

    for cluster in np.unique(labels):
        subset = values[labels == cluster]

        if subset.size == 0:
            continue

        mean = float(np.mean(subset))
        ss_between += float(subset.size) * (mean - grand_mean) ** 2

    return float(ss_between / ss_total)


def standardized_mean_range(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    mask = np.isfinite(values)
    values = values[mask]
    labels = labels[mask]

    if values.size == 0:
        return 0.0

    global_std = float(np.std(values))

    if global_std <= 0:
        return 0.0

    means = []

    for cluster in np.unique(labels):
        subset = values[labels == cluster]

        if subset.size:
            means.append(float(np.mean(subset)))

    if len(means) <= 1:
        return 0.0

    return float(
        (max(means) - min(means))
        / global_std
    )


def summarize_continuous_by_cluster(
    values: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for cluster in sorted(np.unique(labels).tolist()):
        subset = values[labels == cluster]
        subset = subset[np.isfinite(subset)]

        if subset.size == 0:
            continue

        out[str(int(cluster))] = {
            "count": int(subset.size),
            "mean": float(np.mean(subset)),
            "std": float(np.std(subset)),
            "median": float(np.median(subset)),
            "p25": float(np.percentile(subset, 25)),
            "p75": float(np.percentile(subset, 75)),
            "min": float(np.min(subset)),
            "max": float(np.max(subset)),
        }

    return out


def cramers_v_from_labels(
    categories: np.ndarray,
    clusters: np.ndarray,
) -> float:
    category_names = sorted(set(categories.astype(str).tolist()))
    cluster_names = sorted(set(clusters.astype(int).tolist()))

    c_index = {name: i for i, name in enumerate(category_names)}
    k_index = {name: i for i, name in enumerate(cluster_names)}

    table = np.zeros(
        (len(category_names), len(cluster_names)),
        dtype=np.int64,
    )

    for category, cluster in zip(categories.astype(str), clusters.astype(int)):
        table[c_index[category], k_index[int(cluster)]] += 1

    n = float(np.sum(table))

    if n <= 0:
        return 0.0

    row_sums = np.sum(table, axis=1, keepdims=True)
    col_sums = np.sum(table, axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / n
    mask = expected > 0

    chi2 = float(
        np.sum(
            ((table[mask] - expected[mask]) ** 2)
            / expected[mask]
        )
    )

    denom = min(table.shape[0] - 1, table.shape[1] - 1)

    if denom <= 0:
        return 0.0

    return float(math.sqrt((chi2 / n) / denom))


def categorical_metrics(
    categories: np.ndarray,
    clusters: np.ndarray,
) -> Dict[str, Any]:
    categories = categories.astype(str)

    valid = categories != "__MISSING__"
    categories = categories[valid]
    clusters = clusters[valid]

    if categories.size == 0 or len(set(categories.tolist())) <= 1:
        return {
            "category_count": int(len(set(categories.tolist()))),
            "nmi": 0.0,
            "ami": 0.0,
            "cramers_v": 0.0,
        }

    return {
        "category_count": int(len(set(categories.tolist()))),
        "nmi": float(
            normalized_mutual_info_score(
                categories,
                clusters,
            )
        ),
        "ami": float(
            adjusted_mutual_info_score(
                categories,
                clusters,
            )
        ),
        "cramers_v": cramers_v_from_labels(
            categories,
            clusters,
        ),
    }


def characterize_partition_seed(
    *,
    class_mask: np.ndarray,
    aligned_ssl_labels: np.ndarray,
    baseline_x: np.ndarray,
    feature_names: Sequence[str],
    metadata: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    labels = aligned_ssl_labels[class_mask]

    continuous_results: Dict[str, Any] = {}

    for feature_index, feature_name in enumerate(feature_names):
        values = baseline_x[class_mask, feature_index]

        continuous_results[feature_name] = {
            "source": "handcrafted_baseline_feature",
            "eta_squared": eta_squared(values, labels),
            "standardized_mean_range": standardized_mean_range(
                values,
                labels,
            ),
            "per_cluster": summarize_continuous_by_cluster(
                values,
                labels,
            ),
        }

    for name in CONTINUOUS_METADATA_CANDIDATES:
        values = np.asarray(metadata[name], dtype=np.float64)[class_mask]

        if np.isfinite(values).sum() == 0:
            continue

        continuous_results[name] = {
            "source": "metadata",
            "eta_squared": eta_squared(values, labels),
            "standardized_mean_range": standardized_mean_range(
                values,
                labels,
            ),
            "per_cluster": summarize_continuous_by_cluster(
                values,
                labels,
            ),
        }

    categorical_results: Dict[str, Any] = {}

    for name in CATEGORICAL_METADATA_CANDIDATES:
        values = np.asarray(metadata[name]).astype(str)[class_mask]

        if set(values.tolist()) == {"__MISSING__"}:
            continue

        categorical_results[name] = categorical_metrics(
            values,
            labels,
        )

    cluster_counts = np.bincount(
        labels.astype(int),
        minlength=EXPECTED_SSL_K,
    )

    return {
        "count": int(labels.size),
        "occupied_ssl_clusters": int(np.sum(cluster_counts > 0)),
        "cluster_counts": cluster_counts.astype(int).tolist(),
        "continuous_variables": continuous_results,
        "categorical_variables": categorical_results,
    }


def aggregate_variable_across_seeds(
    seed_results: Mapping[int, Dict[str, Any]],
    *,
    variable_type: str,
) -> List[Dict[str, Any]]:
    if variable_type == "continuous":
        field = "continuous_variables"
        metric = "eta_squared"
        secondary = "standardized_mean_range"
    elif variable_type == "categorical":
        field = "categorical_variables"
        metric = "nmi"
        secondary = "cramers_v"
    else:
        raise ValueError(variable_type)

    all_names = sorted(
        {
            name
            for result in seed_results.values()
            for name in result[field]
        }
    )

    ranked: List[Dict[str, Any]] = []

    for name in all_names:
        primary_values = []
        secondary_values = []

        for result in seed_results.values():
            if name not in result[field]:
                continue

            primary_values.append(
                float(result[field][name][metric])
            )
            secondary_values.append(
                float(result[field][name][secondary])
            )

        if not primary_values:
            continue

        primary = np.asarray(primary_values, dtype=np.float64)
        secondary_arr = np.asarray(secondary_values, dtype=np.float64)

        ranked.append(
            {
                "variable": name,
                "seed_count": int(len(primary_values)),
                metric: {
                    "mean": float(np.mean(primary)),
                    "std": float(np.std(primary)),
                    "min": float(np.min(primary)),
                    "max": float(np.max(primary)),
                },
                secondary: {
                    "mean": float(np.mean(secondary_arr)),
                    "std": float(np.std(secondary_arr)),
                    "min": float(np.min(secondary_arr)),
                    "max": float(np.max(secondary_arr)),
                },
            }
        )

    ranked.sort(
        key=lambda item: (
            -item[metric]["mean"],
            item["variable"],
        )
    )

    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize movement/context variables associated with "
            "reproducible SSL subclusters inside known DS-005 behaviors."
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
        "--label-column",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_dir = args.baseline_dir.resolve()
    metadata_root = args.metadata_root.resolve()
    label_root = args.label_root.resolve()
    training_config = args.training_config.resolve()
    output_dir = args.output_dir.resolve()

    for path in (
        baseline_dir,
        metadata_root,
        label_root,
        training_config,
        output_dir,
    ):
        prohibit_test_path(path)

    assert_no_test_artifacts(metadata_root)
    assert_no_test_artifacts(label_root)

    seeds = configured_ssl_seeds(training_config)

    if REFERENCE_SEED not in seeds:
        raise RuntimeError(
            f"Reference seed {REFERENCE_SEED} missing from frozen seeds {list(seeds)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_dir = output_dir / "per_class"
    per_class_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    checksum_path = output_dir / "SUBCLUSTER_CHARACTERIZATION_SHA256SUMS"

    if not args.overwrite and summary_path.exists():
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite for an intentional rerun."
        )

    print("=" * 80)
    print("DS-005 SSL SUBCLUSTER CHARACTERIZATION")
    print("=" * 80)
    print(f"Candidate classes: {args.classes}")
    print(f"SSL seeds:         {list(seeds)}")
    print(f"Reference seed:    {REFERENCE_SEED}")
    print("Alignment:         TRAIN-derived Hungarian mapping")
    print("Inputs:            18 handcrafted features + available metadata")
    print("Evaluation:        TRAIN + VALIDATION")
    print("TEST partition:    PROTECTED / NOT LOADED")
    print()

    baseline_x: Dict[str, np.ndarray] = {}
    baseline_alignment: Dict[str, Dict[str, np.ndarray]] = {}
    feature_names: Optional[List[str]] = None

    for partition in PARTITIONS:
        X, names, alignment = load_baseline_raw(
            baseline_dir,
            partition=partition,
        )

        baseline_x[partition] = X
        baseline_alignment[partition] = alignment

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError(
                "TRAIN/VALIDATION baseline feature names differ."
            )

    assert feature_names is not None

    metadata_by_seed: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    ssl_labels_by_seed: Dict[int, Dict[str, np.ndarray]] = {}

    reference_known: Dict[str, np.ndarray] = {}
    reference_bouts: Dict[str, np.ndarray] = {}
    detected_columns: Dict[str, str] = {}

    for seed in seeds:
        metadata_by_seed[seed] = {}
        ssl_labels_by_seed[seed] = {}

        for partition in PARTITIONS:
            metadata, detected_column = load_metadata(
                metadata_root,
                ssl_seed=seed,
                partition=partition,
                requested_label_column=args.label_column,
            )

            verify_alignment(
                baseline_alignment[partition],
                metadata,
                partition=partition,
                ssl_seed=seed,
            )

            labels = load_ssl_labels(
                label_root,
                ssl_seed=seed,
                partition=partition,
            )

            metadata_by_seed[seed][partition] = metadata
            ssl_labels_by_seed[seed][partition] = labels

            known = metadata["known_label"]
            bouts = metadata["bout_id"]

            if partition not in reference_known:
                reference_known[partition] = known.copy()
                reference_bouts[partition] = bouts.copy()
                detected_columns[partition] = detected_column
            else:
                if not np.array_equal(
                    reference_known[partition],
                    known,
                ):
                    raise RuntimeError(
                        f"Known labels differ across seeds for {partition}."
                    )

                if not np.array_equal(
                    reference_bouts[partition],
                    bouts,
                ):
                    raise RuntimeError(
                        f"Bout ordering differs across seeds for {partition}."
                    )

    reference_train_labels = ssl_labels_by_seed[REFERENCE_SEED]["train"]

    mappings: Dict[int, Dict[int, int]] = {
        REFERENCE_SEED: {
            cluster: cluster
            for cluster in range(EXPECTED_SSL_K)
        }
    }

    for seed in seeds:
        if seed == REFERENCE_SEED:
            continue

        mappings[seed] = hungarian_map_to_reference(
            reference_train_labels,
            ssl_labels_by_seed[seed]["train"],
        )

    aligned_labels_by_seed: Dict[int, Dict[str, np.ndarray]] = {}

    for seed in seeds:
        aligned_labels_by_seed[seed] = {
            partition: apply_mapping(
                ssl_labels_by_seed[seed][partition],
                mappings[seed],
            )
            for partition in PARTITIONS
        }

    observed_classes = sorted(
        set(reference_known["train"].tolist())
        | set(reference_known["validation"].tolist())
    )

    missing_requested = [
        class_name
        for class_name in args.classes
        if class_name not in observed_classes
    ]

    if missing_requested:
        raise RuntimeError(
            "Requested known classes not observed: "
            + ", ".join(missing_requested)
        )

    class_outputs: Dict[str, Any] = {}
    written_artifacts: List[Path] = []

    for class_name in args.classes:
        print("=" * 80)
        print(class_name)
        print("=" * 80)

        class_result: Dict[str, Any] = {
            "class_name": class_name,
            "test_partition_used": False,
        }

        for partition in PARTITIONS:
            mask = reference_known[partition] == class_name

            if not np.any(mask):
                raise RuntimeError(
                    f"{class_name}: no {partition} bouts."
                )

            seed_results: Dict[int, Dict[str, Any]] = {}

            for seed in seeds:
                seed_results[seed] = characterize_partition_seed(
                    class_mask=mask,
                    aligned_ssl_labels=aligned_labels_by_seed[seed][partition],
                    baseline_x=baseline_x[partition],
                    feature_names=feature_names,
                    metadata=metadata_by_seed[seed][partition],
                )

            continuous_ranking = aggregate_variable_across_seeds(
                seed_results,
                variable_type="continuous",
            )

            categorical_ranking = aggregate_variable_across_seeds(
                seed_results,
                variable_type="categorical",
            )

            class_result[partition] = {
                "count": int(np.sum(mask)),
                "per_seed": {
                    str(seed): result
                    for seed, result in seed_results.items()
                },
                "continuous_variable_ranking": continuous_ranking,
                "categorical_variable_ranking": categorical_ranking,
            }

            print(f"{partition.upper()} count: {int(np.sum(mask)):,}")

            print("  Top continuous variables by eta^2:")
            for item in continuous_ranking[:8]:
                print(
                    f"    {item['variable']:<28} "
                    f"eta^2={item['eta_squared']['mean']:.4f} "
                    f"range_z={item['standardized_mean_range']['mean']:.3f}"
                )

            if categorical_ranking:
                print("  Categorical associations:")
                for item in categorical_ranking[:5]:
                    print(
                        f"    {item['variable']:<28} "
                        f"NMI={item['nmi']['mean']:.4f} "
                        f"Cramer's V={item['cramers_v']['mean']:.4f}"
                    )

            print()

        output_path = per_class_dir / f"{class_name}.json"

        atomic_write_json(
            output_path,
            class_result,
        )

        written_artifacts.append(output_path)
        class_outputs[class_name] = class_result

    summary_classes: Dict[str, Any] = {}

    for class_name, result in class_outputs.items():
        val_cont = result["validation"]["continuous_variable_ranking"]
        val_cat = result["validation"]["categorical_variable_ranking"]
        train_cont = {
            item["variable"]: item
            for item in result["train"]["continuous_variable_ranking"]
        }

        top_continuous = []

        for item in val_cont[:10]:
            train_item = train_cont.get(item["variable"])

            top_continuous.append(
                {
                    "variable": item["variable"],
                    "validation_mean_eta_squared": (
                        item["eta_squared"]["mean"]
                    ),
                    "train_mean_eta_squared": (
                        train_item["eta_squared"]["mean"]
                        if train_item is not None
                        else None
                    ),
                    "validation_mean_standardized_mean_range": (
                        item["standardized_mean_range"]["mean"]
                    ),
                }
            )

        summary_classes[class_name] = {
            "train_count": result["train"]["count"],
            "validation_count": result["validation"]["count"],
            "top_validation_continuous_variables": top_continuous,
            "top_validation_categorical_variables": [
                {
                    "variable": item["variable"],
                    "validation_mean_nmi": item["nmi"]["mean"],
                    "validation_mean_cramers_v": item["cramers_v"]["mean"],
                }
                for item in val_cat[:5]
            ],
        }

    summary = {
        "dataset_id": "DS-005",
        "analysis": "ssl_subcluster_characterization",
        "candidate_classes": list(args.classes),
        "ssl_training_seeds": list(seeds),
        "reference_seed": REFERENCE_SEED,
        "known_label_columns": detected_columns,
        "handcrafted_feature_names": feature_names,
        "train_derived_cluster_mappings_to_reference": {
            str(seed): {
                str(src): int(dst)
                for src, dst in mapping.items()
            }
            for seed, mapping in mappings.items()
        },
        "classes": summary_classes,
        "interpretation_guardrails": {
            "post_hoc_characterization": True,
            "association_not_causation": True,
            "reproducible_subclusters_not_automatically_new_behaviors": True,
            "speed_dominance_should_be_treated_as_possible_intensity_structure": True,
            "context_association_should_be_interpreted_with_experimental_design": True,
            "test_partition_used": False,
        },
        "test_partition_used": False,
    }

    atomic_write_json(summary_path, summary)
    written_artifacts.append(summary_path)

    write_checksums(
        checksum_path,
        written_artifacts,
    )

    print("=" * 80)
    print("SUBCLUSTER CHARACTERIZATION SUMMARY")
    print("=" * 80)

    for class_name in args.classes:
        info = summary_classes[class_name]
        print(class_name)

        for item in info["top_validation_continuous_variables"][:5]:
            print(
                f"  {item['variable']:<28} "
                f"VALIDATION eta^2={item['validation_mean_eta_squared']:.4f} "
                f"TRAIN eta^2={item['train_mean_eta_squared']:.4f}"
            )

        categorical = info["top_validation_categorical_variables"]

        if categorical:
            print(
                "  strongest categorical: "
                f"{categorical[0]['variable']} "
                f"(NMI={categorical[0]['validation_mean_nmi']:.4f})"
            )

        print()

    print("TEST partition used: NO")
    print(f"Summary:    {summary_path}")
    print(f"Checksums:  {checksum_path}")


if __name__ == "__main__":
    main()
