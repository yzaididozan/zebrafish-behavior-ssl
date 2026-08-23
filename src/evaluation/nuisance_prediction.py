"""Fixed nuisance-prediction models for confirmatory evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationProbeResult:
    balanced_accuracy: float
    macro_f1: float
    uniform_chance: float
    chance_ratio: float
    n_classes: int


@dataclass(frozen=True)
class RegressionProbeResult:
    r2: float
    mae: float


def _make_logistic_regression() -> LogisticRegression:
    """Create the frozen classification nuisance model.

    ``multi_class`` is intentionally omitted for compatibility across recent
    scikit-learn versions; multinomial behavior is selected automatically by
    compatible solvers for multiclass targets.
    """
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=20260822,
    )


def fit_classification_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    standardize: bool = True,
) -> tuple[object, ClassificationProbeResult]:
    """Fit the frozen nuisance classifier and evaluate it."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train)
    x_eval = np.asarray(x_eval, dtype=float)
    y_eval = np.asarray(y_eval)

    if standardize:
        scaler = StandardScaler()
        x_train_fit = scaler.fit_transform(x_train)
        x_eval_fit = scaler.transform(x_eval)
    else:
        scaler = None
        x_train_fit = x_train
        x_eval_fit = x_eval

    model = _make_logistic_regression()
    model.fit(x_train_fit, y_train)
    pred = model.predict(x_eval_fit)

    n_classes = int(np.unique(y_eval).size)
    if n_classes < 2:
        raise ValueError("y_eval must contain at least two classes.")

    balanced = float(balanced_accuracy_score(y_eval, pred))
    uniform_chance = 1.0 / n_classes

    result = ClassificationProbeResult(
        balanced_accuracy=balanced,
        macro_f1=float(f1_score(y_eval, pred, average="macro", zero_division=0)),
        uniform_chance=uniform_chance,
        chance_ratio=balanced / uniform_chance,
        n_classes=n_classes,
    )

    # Return scaler and model together without introducing a Pipeline whose
    # internals could be accidentally refit on evaluation data.
    return {"scaler": scaler, "model": model}, result


def fit_fish_identity_probe(
    x_train: np.ndarray,
    fish_train: np.ndarray,
    x_eval: np.ndarray,
    fish_eval: np.ndarray,
) -> tuple[object, ClassificationProbeResult]:
    """Convenience wrapper for fish-identity leakage."""
    train_classes = set(np.unique(fish_train).tolist())
    eval_classes = set(np.unique(fish_eval).tolist())
    missing = eval_classes - train_classes
    if missing:
        raise ValueError(
            "Fish-identity probe evaluation contains fish classes absent from "
            f"probe training: {sorted(missing)!r}"
        )

    return fit_classification_probe(x_train, fish_train, x_eval, fish_eval)


def fit_context_probe(
    x_train: np.ndarray,
    context_train: np.ndarray,
    x_eval: np.ndarray,
    context_eval: np.ndarray,
) -> tuple[object, ClassificationProbeResult]:
    """Convenience wrapper for context/session leakage."""
    return fit_classification_probe(x_train, context_train, x_eval, context_eval)


def fit_continuous_nuisance_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    alpha: float = 1.0,
) -> tuple[object, RegressionProbeResult]:
    """Fit frozen Ridge(alpha=1.0) to a continuous nuisance target."""
    scaler = StandardScaler()
    x_train_fit = scaler.fit_transform(np.asarray(x_train, dtype=float))
    x_eval_fit = scaler.transform(np.asarray(x_eval, dtype=float))

    model = Ridge(alpha=alpha)
    model.fit(x_train_fit, np.asarray(y_train, dtype=float))
    pred = model.predict(x_eval_fit)

    return {"scaler": scaler, "model": model}, RegressionProbeResult(
        r2=float(r2_score(y_eval, pred)),
        mae=float(mean_absolute_error(y_eval, pred)),
    )
