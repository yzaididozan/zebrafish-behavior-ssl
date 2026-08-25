#!/usr/bin/env python3
"""One-time, inference-only DS-006 held-out TEST evaluation.

This is the only project program authorized to open these sealed artifacts:

* data/processed/DS-006/ssl/test.npz
* data/processed/DS-006/baseline/test_core_raw.npz

Without ``--confirm-open-test`` the final mode exits before either TEST file is
opened.  Probe preparation is a separate TRAIN-only mode and must be completed,
hashed, committed, and frozen before final TEST execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import joblib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ssl.encoder import ContrastiveModel, EncoderConfig
from src.ssl.train import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 23, 37, 51, 79)
N_TEST = 26_130
N_RECORDINGS = 5
INPUT_SHAPE = (N_TEST, 175, 3)
EMBEDDING_SHAPE = (N_TEST, 64)
K = 8
SILHOUETTE_MAX = 20_000
SILHOUETTE_SEED = 20260822
OUTPUT = ROOT / "data/processed/DS-006/final_test_evaluation"
SSL_TEST = ROOT / "data/processed/DS-006/ssl/test.npz"
BASELINE_TEST = ROOT / "data/processed/DS-006/baseline/test_core_raw.npz"
BASELINE_TRAIN = ROOT / "data/processed/DS-006/baseline/train_core_raw.npz"
METADATA = ROOT / "data/processed/DS-006/metadata/bout_metadata.csv"
TRAINING_CONFIG = ROOT / "configs/ssl/training.yaml"
CLUSTER_ROOT = ROOT / "data/processed/DS-006/transfer_clustering"
BASELINE_CLUSTER_ROOT = ROOT / "data/processed/DS-006/baseline_clustering"
PROBE_ROOT = ROOT / "data/processed/DS-006/frozen_test_probes"

FEATURES = (
    "bout_duration", "inter_bout_interval", "speed_mean", "speed_std",
    "speed_median", "speed_max", "speed_p95", "speed_rms",
    "speed_change_abs_mean", "speed_change_std", "speed_change_max",
    "speed_change_rms", "turn_total_abs", "turn_net", "turn_abs_mean",
    "turn_std", "turn_max", "turn_rms",
)
AXES = (
    "speed_change_rms", "speed_change_std", "bout_duration", "turn_net",
    "turn_total_abs",
)
CONTEXT_FIELDS = ("recording_id", "family", "condition_label", "condition_code", "well")

# Frozen by DEC-023 and the committed TRAIN-only clustering manifests.
FROZEN_HASHES: Dict[str, str] = {
    "data/processed/DS-006/ssl/test.npz": "5b4291bd46ec06ddc0a5c03a7b4b595559d85f861dd882375f8bfee10ec81bd8",
    "data/processed/DS-006/baseline/test_core_raw.npz": "4c442917cf7e3549da712aff3cb25ef58be596bb0c86b639d3229611357032c6",
    "data/processed/DS-006/baseline/train_core_raw.npz": "41a39dd0f2035520b5a2f07514e6cd90c2e9478cb4bc54ea5e52604a00a36406",
    "data/processed/DS-006/baseline/feature_manifest.json": "5dcaf8447e969114cb7f1fa40ae24ed66194ec388fb02802583e2b716b14f315",
    "data/processed/DS-006/metadata/bout_metadata.csv": "7625b0f32731f0e8e67fbf09b32fac36a376128e0a2c8f85821e39ba71aad47e",
    "configs/ssl/training.yaml": "d70da9c8ac1064025f3eec0f8784770d2405cf543275ec01289dba32401271dc",
    "data/processed/DS-006/transfer_clustering/cross_seed_stability.json": "b45779bbe5622b4858f57f28a7731675c9f21c2951ef6d066377c5e8c7ad65d2",
    "results/ssl/checkpoints/ssl_seed11_best.pt": "f837bce2a8ba3fca50ff7689c51cf2f7c56904445e83941e7d0bfc83370ceeda",
    "results/ssl/checkpoints/ssl_seed23_best.pt": "2773586a0ddfbdb1427f5e344764a7fd37998fdc2c089cbf39001667e5b9b076",
    "results/ssl/checkpoints/ssl_seed37_best.pt": "21a048aee177492c21829b16099b792d722371e1ebaf1b9607130ef3bb2f8086",
    "results/ssl/checkpoints/ssl_seed51_best.pt": "66d6d2837f67ec9b4682a20ef001dd9af275d860e60c83147ba3ecea2f2a268f",
    "results/ssl/checkpoints/ssl_seed79_best.pt": "0c646674c8e1c0b35847a33f823552fb4f7b61ddb01921bcda34d120318525e8",
    "data/processed/DS-006/baseline_clustering/imputer.joblib": "6355836c005fcab678593f86f166b5c66d361f7c982e2c54840338edccfa481c",
    "data/processed/DS-006/baseline_clustering/scaler.joblib": "3ea3e5ba523aa62a7abdf820ec8dfbdaccc1c760b7bfdc3a90a9400e6c5fe106",
    "data/processed/DS-006/baseline_clustering/pca.joblib": "e25e1103944476984f5b8c338578e07d6d0d7d71b1bd56e8dbdfbb9fb7565693",
    "data/processed/DS-006/baseline_clustering/gmm.joblib": "1be3da5109825407c70693ea6485a605fea3aedcd6c911850abd208dd5fc0b55",
}

TRANSFER_HASHES = {
    11: ("9b6792f573c242ca41b71813c47363a975aa4bc0dfd9ef4e75a3b9e901ef1f30", "30c6598754752b22e816292a2bb93cd7af5506f2870c06610e9687f4c8a605a7", "837fb5bdf177c8ac1e6b885ef6edd1a0cff2dcfb8b8006f6e64cf0f202f193ed"),
    23: ("1035a824ce4c635d809ebe732135cbf09b261a08d688ee6cd2100cd4e0a5de8b", "c7621a7ba79dd627b9d3f70347a2f47241daa881e27800807167c0126f4b8e23", "07c68c763036e7fb0d9f572ae1742774b4e1a4bcfa6874574f381681d14780ba"),
    37: ("5a81f6a13849efacf747b04528b26a5f30c5cd04a78a1b64b11a819ae15c33b6", "1ce61eb58a493b7ea11fc581cac2db82b4f80814ad5f0a65f19a6197257104eb", "458c2b5e48e5caabc7c59122ea515b23829b4794613b74d2950584189ba20e92"),
    51: ("10e8f61c3705021a591ea7aa8a059a7e7490015d613cce810c08a19a1612285d", "6087c18e377fc4fa408b9255e42c593e16cfe5d9ac175619b51d69b172ed31ec", "be93667647aff5bb087d09ada9c80ac23ad3b4b2f898b2a86557355fe7e3f7ae"),
    79: ("704092fb4a796cf119543692ae68df94686e466cb0312380463054f8650873cb", "dc1ad864599f4e89105f37b2ac01b40d11f6a9d6953ea08d88f441a3ae2ff1ce", "88ca477abd6fa2f0249de657a8dd2f9bb1a0f5a3149650fdfbf3402d8405faf3"),
}
for _seed, _hashes in TRANSFER_HASHES.items():
    for _name, _hash in zip(("scaler.joblib", "pca.joblib", "kmeans.joblib"), _hashes):
        FROZEN_HASHES[f"data/processed/DS-006/transfer_clustering/seed{_seed}/{_name}"] = _hash

TRAIN_LABEL_HASHES = {
    11: "2459ef251872fd3cbe8ce451e5cac41eeb138f75f7316e2d33dc6cd982d54f6d",
    23: "379cbbdd0fd48ec66efb2a1a77c1b98886c51771850d9b573dcaefb444c33e4b",
    37: "6158ac40010558434068523be14e6e8291e34770dd25a38833ce358058e94c3a",
    51: "388db690cc288349ca2c88c35ef9b4da64c82767edd01438911fe69f9bba0363",
    79: "6f7f54a01a684632022ce5390c4230051a44de305e66d3bba077d35d18f07826",
}
TRAIN_PROFILE_HASHES = {
    11: "bdf1b629d69698de993f01ad786ed2858290fe91137fb078682d7a3beadbeb66",
    23: "46ed3c663f0aba3621c3e71cb0adb62138e623565d84b2310abec376c97e6679",
    37: "87c55172bb4da8d88fceb730bd874bf6e4cec0ba77bf25df66445c83b8954007",
    51: "cd68821c14439e0f43cc9a1502f813272448bbe6e39b3f83912154d844da93d4",
    79: "e1f93c0f8fa95a2871a6fb69c5a40262fa8bae3591e0312f00fcedf582b7982e",
}
for _seed in SEEDS:
    FROZEN_HASHES[f"data/processed/DS-006/transfer_clustering/seed{_seed}/train_labels_aligned.npy"] = TRAIN_LABEL_HASHES[_seed]
    FROZEN_HASHES[f"data/processed/DS-006/transfer_substructure/seed{_seed}/train_feature_characterization.json"] = TRAIN_PROFILE_HASHES[_seed]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-probes", action="store_true")
    modes.add_argument("--confirm-open-test", action="store_true")
    p.add_argument("--freeze-commit", help="Required full 40-character pre-TEST freeze commit")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp = f.name
    os.replace(tmp, path)


def verify(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Frozen SHA-256 mismatch for {path}: {observed} != {expected}")
    return observed


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def require_freeze_commit(value: str | None) -> str:
    if value is None or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise SystemExit("--freeze-commit requires the full 40-character frozen commit SHA")
    head = git_head()
    if head != value:
        raise SystemExit(f"Refusing TEST: repository HEAD {head} does not equal freeze commit {value}")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout
    if status:
        raise SystemExit("Refusing TEST: working tree is not clean")
    return head


def load_npz(
    path: Path,
    expected_shape: Tuple[int, ...],
    *,
    allow_nan_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if set(("X", "bout_id")) - set(z.files):
            raise RuntimeError(f"{path} lacks X/bout_id")
        X = np.asarray(z["X"])
        ids = np.asarray(z["bout_id"]).astype(str)
    if X.shape != expected_shape or ids.shape != (expected_shape[0],):
        raise RuntimeError(f"Unexpected shape in {path}: X={X.shape}, bout_id={ids.shape}")
    values_valid = not np.isinf(X).any() if allow_nan_features else np.isfinite(X).all()
    if not values_valid or len(np.unique(ids)) != len(ids) or np.any(np.char.str_len(ids) == 0):
        raise RuntimeError(f"Finite/unique/nonempty constraint failed for {path}")
    return X, ids


def load_metadata(ids: np.ndarray) -> Dict[str, np.ndarray]:
    required = {"bout_id", "fish_id", *CONTEXT_FIELDS, "partition"}
    rows: Dict[str, Mapping[str, str]] = {}
    with METADATA.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or ()):
            raise RuntimeError("Frozen metadata schema changed")
        for row in reader:
            if row["partition"] == "test":
                rows[row["bout_id"]] = row
    if len(rows) != N_TEST or any(x not in rows for x in ids):
        raise RuntimeError("TEST metadata/bout ID alignment failed")
    return {field: np.asarray([rows[x][field] for x in ids], dtype=str) for field in required}


def training_config() -> Dict[str, Any]:
    cfg = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))["training"]
    if tuple(cfg["seeds"]["values"]) != SEEDS or int(cfg["encoder"]["embedding_dim"]) != 64:
        raise RuntimeError("Frozen encoder configuration changed")
    return cfg


def model_for_seed(cfg: Mapping[str, Any], seed: int, device: torch.device) -> ContrastiveModel:
    enc, proj = cfg["encoder"], cfg["projection_head"]
    model = ContrastiveModel(EncoderConfig(
        input_channels=int(enc["architecture"]["input_channels"]),
        embedding_dim=64,
        projection_dim=int(proj["projection_dim"]),
        dropout=float(enc["architecture"]["dropout"]),
    )).to(device)
    checkpoint = ROOT / f"results/ssl/checkpoints/ssl_seed{seed}_best.pt"
    payload = load_checkpoint(path=checkpoint, model=model, optimizer=None, map_location=device)
    if int(payload.get("training_seed", seed)) != seed:
        raise RuntimeError("Checkpoint seed mismatch")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


@torch.inference_mode()
def encode(model: ContrastiveModel, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    out = np.empty(EMBEDDING_SHAPE, dtype=np.float32)
    for start in range(0, N_TEST, batch_size):
        stop = min(start + batch_size, N_TEST)
        z = model.encoder(torch.from_numpy(X[start:stop]).to(device=device, dtype=torch.float32))
        if z.shape != (stop - start, 64) or not torch.isfinite(z).all():
            raise RuntimeError("Invalid frozen encoder output")
        out[start:stop] = z.cpu().numpy()
    return out


def summarize(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    x, y = values[valid], labels[valid]
    total = float(np.sum((x - x.mean()) ** 2))
    if total <= 0:
        return 0.0
    between = sum(int(np.sum(y == c)) * float((x[y == c].mean() - x.mean()) ** 2) for c in np.unique(y))
    return float(between / total)


def profile(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.asarray([np.nanmean(values[labels == c]) for c in range(K)], dtype=float)


def contingency(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(values.astype(str), return_inverse=True)
    table = np.zeros((int(encoded.max()) + 1, K), dtype=np.int64)
    np.add.at(table, (encoded, labels), 1)
    return table


def cramers_v(table: np.ndarray) -> float:
    n = int(table.sum())
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    mask = expected > 0
    chi2 = float(np.sum((table[mask] - expected[mask]) ** 2 / expected[mask]))
    d = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(math.sqrt((chi2 / n) / d)) if d > 0 else 0.0


def association(values: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    _, encoded = np.unique(values.astype(str), return_inverse=True)
    table = contingency(values, labels)
    column = table / np.maximum(table.sum(0, keepdims=True), 1)
    p = column[column > 0]
    entropy = float(-np.sum(p * np.log(p)) / K / math.log(max(table.shape[0], 2)))
    return {
        "nmi": float(normalized_mutual_info_score(encoded, labels)),
        "ami": float(adjusted_mutual_info_score(encoded, labels)),
        "cramers_v": cramers_v(table),
        "mean_normalized_entropy": entropy,
        "maximum_concentration": float(column.max()),
    }


def conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    answer = 0.0
    for value in np.unique(x):
        subset = y[x == value]
        _, counts = np.unique(subset, return_counts=True)
        p = counts / counts.sum()
        answer += len(subset) / len(y) * float(-np.sum(p * np.log(p)))
    return answer


def comparison(baseline: np.ndarray, ssl: np.ndarray) -> Dict[str, float]:
    def entropy(x: np.ndarray) -> float:
        _, counts = np.unique(x, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p)))
    hs, hb = entropy(ssl), entropy(baseline)
    return {
        "ari": float(adjusted_rand_score(baseline, ssl)),
        "nmi": float(normalized_mutual_info_score(baseline, ssl)),
        "ami": float(adjusted_mutual_info_score(baseline, ssl)),
        "normalized_conditional_entropy_ssl_given_baseline": conditional_entropy(ssl, baseline) / hs,
        "normalized_conditional_entropy_baseline_given_ssl": conditional_entropy(baseline, ssl) / hb,
    }


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def prepare_probes() -> None:
    """Fit only the preregistered probes on frozen DS-006 TRAIN."""
    if SSL_TEST.exists() and BASELINE_TEST.exists():
        # Existence is allowed; this mode deliberately never opens either path.
        pass
    if PROBE_ROOT.exists():
        raise SystemExit(f"Probe output already exists; refusing overwrite: {PROBE_ROOT}")
    verify(BASELINE_TRAIN, FROZEN_HASHES[str(BASELINE_TRAIN.relative_to(ROOT))])
    X, ids = load_npz(BASELINE_TRAIN, (118_100, 18), allow_nan_features=True)
    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for seed in SEEDS:
        labels_path = CLUSTER_ROOT / f"seed{seed}/train_labels_aligned.npy"
        verify(labels_path, TRAIN_LABEL_HASHES[seed])
        y = np.load(labels_path, allow_pickle=False)
        if y.shape != (118_100,):
            raise RuntimeError("Frozen TRAIN label shape changed")
        linear = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=20260822)),
        ]).fit(X, y)
        nonlinear = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, l2_regularization=0.0, random_state=20260822)),
        ]).fit(X, y)
        speed_only = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=20260822)),
        ]).fit(X[:, [FEATURES.index("speed_mean")]], y)
        for name, obj in (("linear", linear), ("nonlinear", nonlinear), ("speed_only", speed_only)):
            path = PROBE_ROOT / f"seed{seed}_{name}_probe.joblib"
            joblib.dump(obj, path)
            artifacts.append(path)
    manifest = {
        "mode": "TRAIN_ONLY_PROBE_PREPARATION", "test_opened": False,
        "train_source": str(BASELINE_TRAIN.relative_to(ROOT)), "train_sha256": sha256(BASELINE_TRAIN),
        "definitions_frozen": True,
        "artifacts": {str(p.relative_to(ROOT)): sha256(p) for p in artifacts},
    }
    atomic_json(PROBE_ROOT / "probe_manifest.json", manifest)
    print("TRAIN-only probes prepared. Commit these artifacts before opening TEST.")


def device_for(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    return torch.device("mps" if name == "mps" or (name == "auto" and torch.backends.mps.is_available()) else "cpu")


def final_test(args: argparse.Namespace) -> None:
    freeze = require_freeze_commit(args.freeze_commit)
    if OUTPUT.exists():
        raise SystemExit(f"One-time output already exists; refusing overwrite: {OUTPUT}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    # All frozen hashes and the TRAIN-only probe manifest are verified before TEST opens.
    input_hashes = {rel: verify(ROOT / rel, expected) for rel, expected in FROZEN_HASHES.items()}
    probe_manifest_path = PROBE_ROOT / "probe_manifest.json"
    if not probe_manifest_path.is_file():
        raise SystemExit("Frozen probe objects are missing; run --prepare-probes and freeze them first")
    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    if probe_manifest.get("test_opened") is not False:
        raise RuntimeError("Invalid probe manifest")
    for rel, expected in probe_manifest["artifacts"].items():
        input_hashes[rel] = verify(ROOT / rel, expected)

    # Confirmation and all preconditions have passed. These are the only TEST opens.
    ssl_X, ssl_ids = load_npz(SSL_TEST, INPUT_SHAPE)
    baseline_X, baseline_ids = load_npz(
        BASELINE_TEST, (N_TEST, 18), allow_nan_features=True
    )
    if not np.array_equal(ssl_ids, baseline_ids):
        raise RuntimeError("SSL/baseline TEST bout IDs differ")
    meta = load_metadata(ssl_ids)
    if len(np.unique(meta["recording_id"])) != N_RECORDINGS:
        raise RuntimeError("Expected exactly five TEST recordings")

    OUTPUT.mkdir(parents=False)
    device = device_for(args.device)
    cfg = training_config()
    mappings = json.loads((CLUSTER_ROOT / "cross_seed_stability.json").read_text(encoding="utf-8"))["train_derived_mappings_to_reference"]

    imputer = joblib.load(BASELINE_CLUSTER_ROOT / "imputer.joblib")
    scaler_b = joblib.load(BASELINE_CLUSTER_ROOT / "scaler.joblib")
    pca_b = joblib.load(BASELINE_CLUSTER_ROOT / "pca.joblib")
    gmm = joblib.load(BASELINE_CLUSTER_ROOT / "gmm.joblib")
    baseline_labels = gmm.predict(pca_b.transform(scaler_b.transform(imputer.transform(baseline_X))))
    np.save(OUTPUT / "baseline_test_labels.npy", baseline_labels)

    seed_labels: Dict[int, np.ndarray] = {}
    seed_aligned: Dict[int, np.ndarray] = {}
    seed_metrics: Dict[int, Dict[str, Any]] = {}
    nuisance: Dict[str, Any] = {}
    axis_results: Dict[str, Any] = {axis: {"by_seed": {}} for axis in AXES}
    speed = baseline_X[:, FEATURES.index("speed_mean")]

    for seed in SEEDS:
        model = model_for_seed(cfg, seed, device)
        emb = encode(model, ssl_X, args.batch_size, device)
        seed_dir = OUTPUT / f"seed{seed}"
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "test_embeddings.npz", embeddings=emb, bout_id=ssl_ids)
        scaler = joblib.load(CLUSTER_ROOT / f"seed{seed}/scaler.joblib")
        pca = joblib.load(CLUSTER_ROOT / f"seed{seed}/pca.joblib")
        kmeans = joblib.load(CLUSTER_ROOT / f"seed{seed}/kmeans.joblib")
        reduced = pca.transform(scaler.transform(emb))
        labels = np.asarray(kmeans.predict(reduced), dtype=np.int64)
        mapping = {int(k): int(v) for k, v in mappings[str(seed)].items()}
        aligned = np.asarray([mapping[int(x)] for x in labels], dtype=np.int64)
        seed_labels[seed], seed_aligned[seed] = labels, aligned
        np.save(seed_dir / "test_labels.npy", aligned)

        distances = kmeans.transform(reduced)
        nearest = np.partition(distances, 1, axis=1)[:, :2]
        margin = (nearest[:, 1] - nearest[:, 0]) / np.maximum(nearest[:, 1], 1e-12)
        idx = np.random.default_rng(SILHOUETTE_SEED).choice(N_TEST, SILHOUETTE_MAX, replace=False)
        occupancy = {str(c): int(np.sum(aligned == c)) for c in range(K)}
        contributors = {
            str(c): {
                "fish_wells": int(len(np.unique(meta["fish_id"][aligned == c]))),
                "recordings": int(len(np.unique(meta["recording_id"][aligned == c]))),
            } for c in range(K)
        }
        linear = joblib.load(PROBE_ROOT / f"seed{seed}_linear_probe.joblib")
        nonlinear = joblib.load(PROBE_ROOT / f"seed{seed}_nonlinear_probe.joblib")
        speed_only = joblib.load(PROBE_ROOT / f"seed{seed}_speed_only_probe.joblib")
        seed_metrics[seed] = {
            "occupancy": occupancy, "contributors": contributors,
            "silhouette": float(silhouette_score(reduced[idx], labels[idx])),
            "silhouette_sample_size": SILHOUETTE_MAX,
            "distance_margin_confidence": summarize(margin),
            "baseline_vs_ssl": comparison(baseline_labels, aligned),
            "linear_18_feature_to_ssl_probe": classification_metrics(aligned, linear.predict(baseline_X)),
            "nonlinear_18_feature_to_ssl_probe": classification_metrics(aligned, nonlinear.predict(baseline_X)),
            "test_used_for_fitting": False,
        }
        nuisance[str(seed)] = {
            "mean_speed_eta_squared": eta_squared(speed, aligned),
            "speed_only": classification_metrics(aligned, speed_only.predict(speed.reshape(-1, 1))),
            "fish_well_identity": association(meta["fish_id"], aligned),
            "contexts": {field: association(meta[field], aligned) for field in CONTEXT_FIELDS},
        }
        for axis in AXES:
            values = baseline_X[:, FEATURES.index(axis)]
            train = json.loads((ROOT / f"data/processed/DS-006/transfer_substructure/seed{seed}/train_feature_characterization.json").read_text(encoding="utf-8"))
            train_profile = np.asarray([train["features"][axis]["cluster_profile"][str(c)]["mean"] for c in range(K)])
            test_profile = profile(values, aligned)
            axis_results[axis]["by_seed"][str(seed)] = {
                "test_eta_squared": eta_squared(values, aligned),
                "train_to_test_aligned_profile_spearman": float(spearmanr(train_profile, test_profile).statistic),
                "test_profile": test_profile.tolist(),
            }
        atomic_json(seed_dir / "metrics.json", seed_metrics[seed])

    pairs = []
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            pairs.append({
                "seed_a": a, "seed_b": b,
                "ari": float(adjusted_rand_score(seed_aligned[a], seed_aligned[b])),
                "nmi": float(normalized_mutual_info_score(seed_aligned[a], seed_aligned[b])),
                "aligned_agreement": float(np.mean(seed_aligned[a] == seed_aligned[b])),
            })
    cross = {"pairs": pairs, **{metric: summarize([p[metric] for p in pairs]) for metric in ("ari", "nmi", "aligned_agreement")}}
    for axis in AXES:
        profiles = [np.asarray(axis_results[axis]["by_seed"][str(s)]["test_profile"]) for s in SEEDS]
        rho = [float(spearmanr(profiles[i], profiles[j]).statistic) for i in range(5) for j in range(i + 1, 5)]
        axis_results[axis]["cross_seed_test_profile_reproducibility"] = summarize(rho)

    baseline_summary = {str(seed): seed_metrics[seed]["baseline_vs_ssl"] | {
        "linear_probe": seed_metrics[seed]["linear_18_feature_to_ssl_probe"],
        "nonlinear_probe": seed_metrics[seed]["nonlinear_18_feature_to_ssl_probe"],
    } for seed in SEEDS}
    mean_cross_ari = cross["ari"]["mean"]
    mean_speed_eta = float(np.mean([nuisance[str(s)]["mean_speed_eta_squared"] for s in SEEDS]))
    mean_speed_ba = float(np.mean([nuisance[str(s)]["speed_only"]["balanced_accuracy"] for s in SEEDS]))
    mean_identity_v = float(np.mean([nuisance[str(s)]["fish_well_identity"]["cramers_v"] for s in SEEDS]))
    mean_context_v = float(np.mean([
        nuisance[str(s)]["contexts"][field]["cramers_v"]
        for s in SEEDS for field in CONTEXT_FIELDS
    ]))
    mean_baseline_ari = float(np.mean([baseline_summary[str(s)]["ari"] for s in SEEDS]))
    mean_linear_ba = float(np.mean([baseline_summary[str(s)]["linear_probe"]["balanced_accuracy"] for s in SEEDS]))
    mean_nonlinear_ba = float(np.mean([baseline_summary[str(s)]["nonlinear_probe"]["balanced_accuracy"] for s in SEEDS]))

    def axis_mean(axis: str, field: str) -> float:
        return float(np.mean([axis_results[axis]["by_seed"][str(s)][field] for s in SEEDS]))

    speed_change_eta = float(np.mean([axis_mean(a, "test_eta_squared") for a in ("speed_change_rms", "speed_change_std")]))
    speed_change_rho = float(np.mean([axis_mean(a, "train_to_test_aligned_profile_spearman") for a in ("speed_change_rms", "speed_change_std")]))
    duration_eta = axis_mean("bout_duration", "test_eta_squared")
    turn_net_eta = axis_mean("turn_net", "test_eta_squared")
    turn_total_eta = axis_mean("turn_total_abs", "test_eta_squared")
    turn_total_rho = axis_mean("turn_total_abs", "train_to_test_aligned_profile_spearman")

    assessments = {
        "01_broad_ssl_representation_transfer_successful": "SUPPORTED" if all(len(np.unique(seed_aligned[s])) == K for s in SEEDS) else "CONTRADICTED",
        "02_frozen_k8_clustering_has_moderate_cross_seed_structure": "SUPPORTED" if mean_cross_ari >= 0.30 else ("WEAKENED" if mean_cross_ari >= 0.15 else "CONTRADICTED"),
        "03_speed_dependence_reproduces": "SUPPORTED" if mean_speed_eta >= 0.30 else ("WEAKENED" if mean_speed_eta >= 0.10 else "CONTRADICTED"),
        "04_mean_speed_only_collapse_is_rejected": "SUPPORTED" if mean_speed_ba < 0.50 else ("WEAKENED" if mean_speed_ba < 0.70 else "CONTRADICTED"),
        "05_fish_well_identity_leakage_is_low": "SUPPORTED" if mean_identity_v <= 0.20 else ("WEAKENED" if mean_identity_v <= 0.35 else "CONTRADICTED"),
        "06_recording_and_context_leakage_is_low": "SUPPORTED" if mean_context_v <= 0.20 else ("WEAKENED" if mean_context_v <= 0.35 else "CONTRADICTED"),
        "07_coarse_baseline_clustering_differs_from_ssl": "SUPPORTED" if mean_baseline_ari < 0.30 else ("WEAKENED" if mean_baseline_ari < 0.50 else "CONTRADICTED"),
        "08_handcrafted_feature_probes_are_only_moderately_predictive": "SUPPORTED" if mean_linear_ba < 0.65 and mean_nonlinear_ba < 0.65 else ("WEAKENED" if mean_nonlinear_ba < 0.80 else "CONTRADICTED"),
        "09_strong_ds005_nonlinear_recoverability_does_not_reproduce": "SUPPORTED" if mean_nonlinear_ba < 0.70 else ("WEAKENED" if mean_nonlinear_ba < 0.85 else "CONTRADICTED"),
        "10_acceleration_speed_change_heterogeneity_reproduces": "SUPPORTED" if speed_change_eta >= 0.25 and speed_change_rho >= 0.70 else ("WEAKENED" if speed_change_eta >= 0.10 else "CONTRADICTED"),
        "11_turning_magnitude_is_a_partial_analogue": "SUPPORTED" if 0.10 <= turn_total_eta < 0.50 and turn_total_rho >= 0.50 else ("WEAKENED" if turn_total_eta >= 0.05 else "CONTRADICTED"),
        "12_strong_duration_heterogeneity_does_not_reproduce": "SUPPORTED" if duration_eta < 0.20 else ("WEAKENED" if duration_eta < 0.35 else "CONTRADICTED"),
        "13_signed_net_turning_does_not_reproduce_strongly": "SUPPORTED" if turn_net_eta < 0.05 else ("WEAKENED" if turn_net_eta < 0.15 else "CONTRADICTED"),
        "14_direct_long_cs_llc_replication": "NOT_TESTABLE",
    }
    claims = {
        "allowed_statuses": ["SUPPORTED", "WEAKENED", "CONTRADICTED", "NOT_TESTABLE"],
        "assessments": assessments,
        "observed_summary": {
            "mean_cross_seed_ari": mean_cross_ari,
            "mean_speed_eta_squared": mean_speed_eta,
            "mean_speed_only_balanced_accuracy": mean_speed_ba,
            "mean_identity_cramers_v": mean_identity_v,
            "mean_context_cramers_v": mean_context_v,
            "mean_baseline_ssl_ari": mean_baseline_ari,
            "mean_linear_probe_balanced_accuracy": mean_linear_ba,
            "mean_nonlinear_probe_balanced_accuracy": mean_nonlinear_ba,
            "mean_speed_change_eta_squared": speed_change_eta,
            "mean_speed_change_train_test_spearman": speed_change_rho,
            "mean_duration_eta_squared": duration_eta,
            "mean_turn_net_eta_squared": turn_net_eta,
            "mean_turn_total_abs_eta_squared": turn_total_eta,
            "mean_turn_total_abs_train_test_spearman": turn_total_rho,
        },
        "rules_frozen_before_test": True,
    }
    atomic_json(OUTPUT / "baseline_vs_ssl_summary.json", baseline_summary)
    atomic_json(OUTPUT / "cross_seed_summary.json", cross)
    atomic_json(OUTPUT / "nuisance_summary.json", nuisance)
    atomic_json(OUTPUT / "kinematic_axes_summary.json", axis_results)
    atomic_json(OUTPUT / "claim_assessment.json", claims)

    outputs = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name not in {"run_manifest.json", "FINAL_TEST_SHA256SUMS"})
    output_hashes = {str(p.relative_to(OUTPUT)): sha256(p) for p in outputs}
    manifest = {
        "freeze_commit": freeze,
        "command": " ".join(sys.argv),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "joblib", "PyYAML")},
        "input_hashes": input_hashes, "output_hashes": output_hashes,
        "test_bouts": N_TEST, "recordings": N_RECORDINGS,
        "input_shape": list(INPUT_SHAPE), "embedding_shape": list(EMBEDDING_SHAPE),
        "ssl_seeds": list(SEEDS), "unique_nonempty_bout_ids": True,
        "test_used_for_fitting": False, "no_configuration_changed": True,
        "prohibited_operations_performed": [],
    }
    atomic_json(OUTPUT / "run_manifest.json", manifest)
    all_outputs = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "FINAL_TEST_SHA256SUMS")
    (OUTPUT / "FINAL_TEST_SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(OUTPUT)}\n" for p in all_outputs), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.prepare_probes:
        prepare_probes()
        return
    # argparse guarantees the explicit confirmation flag for this branch.
    final_test(args)


if __name__ == "__main__":
    main()
