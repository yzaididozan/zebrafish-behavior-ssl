"""Matched Input A vs Input B representation comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PartitionComparison:
    ari: float
    nmi: float


@dataclass(frozen=True)
class BaselinePredictsSSLResult:
    balanced_accuracy: float
    macro_f1: float


def compare_clusterings(
    input_a_labels: np.ndarray,
    input_b_labels: np.ndarray,
) -> PartitionComparison:
    """Compare matched Input A and Input B cluster assignments."""
    a = np.asarray(input_a_labels)
    b = np.asarray(input_b_labels)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("input_a_labels and input_b_labels must be equal-length 1D arrays.")

    return PartitionComparison(
        ari=float(adjusted_rand_score(a, b)),
        nmi=float(normalized_mutual_info_score(a, b)),
    )


def fit_input_a_to_ssl_cluster_probe(
    x_a_train: np.ndarray,
    ssl_cluster_train: np.ndarray,
    x_a_eval: np.ndarray,
    ssl_cluster_eval: np.ndarray,
) -> tuple[object, BaselinePredictsSSLResult]:
    """Predict SSL cluster membership from frozen Input A features.

    This implements the preregistered "can the engineered baseline reconstruct
    the SSL partition?" analysis using a fixed multinomial-compatible logistic
    regression.
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(x_a_train, dtype=float))
    x_eval = scaler.transform(np.asarray(x_a_eval, dtype=float))

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=20260822,
    )
    model.fit(x_train, np.asarray(ssl_cluster_train))
    pred = model.predict(x_eval)

    result = BaselinePredictsSSLResult(
        balanced_accuracy=float(
            balanced_accuracy_score(np.asarray(ssl_cluster_eval), pred)
        ),
        macro_f1=float(
            f1_score(
                np.asarray(ssl_cluster_eval),
                pred,
                average="macro",
                zero_division=0,
            )
        ),
    )

    return {"scaler": scaler, "model": model}, result


def summarize_seedwise_partition_comparisons(
    input_a_labels: np.ndarray,
    input_b_labels_by_seed: dict[int, np.ndarray],
) -> dict[str, object]:
    """Compare Input A with each SSL seed and return frozen summaries."""
    rows: dict[int, dict[str, float]] = {}

    for seed in sorted(input_b_labels_by_seed):
        result = compare_clusterings(input_a_labels, input_b_labels_by_seed[seed])
        rows[seed] = {"ari": result.ari, "nmi": result.nmi}

    ari = np.asarray([rows[s]["ari"] for s in sorted(rows)], dtype=float)
    nmi = np.asarray([rows[s]["nmi"] for s in sorted(rows)], dtype=float)

    return {
        "per_seed": rows,
        "ari": {
            "median": float(np.median(ari)),
            "q25": float(np.quantile(ari, 0.25)),
            "q75": float(np.quantile(ari, 0.75)),
            "minimum": float(np.min(ari)),
            "maximum": float(np.max(ari)),
        },
        "nmi": {
            "median": float(np.median(nmi)),
            "q25": float(np.quantile(nmi, 0.25)),
            "q75": float(np.quantile(nmi, 0.75)),
            "minimum": float(np.min(nmi)),
            "maximum": float(np.max(nmi)),
        },
    }
