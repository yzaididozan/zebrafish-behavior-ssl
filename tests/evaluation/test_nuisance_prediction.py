import numpy as np
import pytest

from src.evaluation.nuisance_prediction import (
    fit_classification_probe,
    fit_context_probe,
    fit_continuous_nuisance_probe,
    fit_fish_identity_probe,
)


def test_classification_probe_learns_easy_binary_signal():
    rng = np.random.default_rng(0)

    x_train = np.vstack([
        rng.normal(loc=-2.0, scale=0.3, size=(100, 2)),
        rng.normal(loc=2.0, scale=0.3, size=(100, 2)),
    ])
    y_train = np.array([0] * 100 + [1] * 100)

    x_eval = np.vstack([
        rng.normal(loc=-2.0, scale=0.3, size=(40, 2)),
        rng.normal(loc=2.0, scale=0.3, size=(40, 2)),
    ])
    y_eval = np.array([0] * 40 + [1] * 40)

    _, result = fit_classification_probe(
        x_train,
        y_train,
        x_eval,
        y_eval,
    )

    assert result.balanced_accuracy > 0.95
    assert result.macro_f1 > 0.95
    assert result.uniform_chance == pytest.approx(0.5)
    assert result.chance_ratio > 1.9


def test_fish_identity_probe_rejects_unseen_eval_class():
    x_train = np.array([[0.0], [1.0], [0.1], [1.1]])
    fish_train = np.array(["f1", "f2", "f1", "f2"])

    x_eval = np.array([[0.2], [2.0]])
    fish_eval = np.array(["f1", "f3"])

    with pytest.raises(ValueError):
        fit_fish_identity_probe(
            x_train,
            fish_train,
            x_eval,
            fish_eval,
        )


def test_fish_identity_probe_runs_with_shared_classes():
    x_train = np.array([[0.0], [0.1], [1.0], [1.1]])
    fish_train = np.array(["f1", "f1", "f2", "f2"])

    x_eval = np.array([[0.05], [1.05]])
    fish_eval = np.array(["f1", "f2"])

    _, result = fit_fish_identity_probe(
        x_train,
        fish_train,
        x_eval,
        fish_eval,
    )

    assert 0.0 <= result.balanced_accuracy <= 1.0
    assert result.n_classes == 2


def test_context_probe_runs():
    x_train = np.array([[0.0], [0.1], [1.0], [1.1]])
    context_train = np.array([0, 0, 1, 1])

    x_eval = np.array([[0.05], [1.05]])
    context_eval = np.array([0, 1])

    _, result = fit_context_probe(
        x_train,
        context_train,
        x_eval,
        context_eval,
    )

    assert 0.0 <= result.balanced_accuracy <= 1.0


def test_continuous_nuisance_probe_recovers_linear_signal():
    rng = np.random.default_rng(1)

    x_train = rng.normal(size=(200, 3))
    y_train = 3.0 * x_train[:, 0] + x_train[:, 1]

    x_eval = rng.normal(size=(80, 3))
    y_eval = 3.0 * x_eval[:, 0] + x_eval[:, 1]

    _, result = fit_continuous_nuisance_probe(
        x_train,
        y_train,
        x_eval,
        y_eval,
        alpha=1.0,
    )

    assert result.r2 > 0.95
    assert result.mae < 0.25


def test_classification_probe_requires_at_least_two_eval_classes():
    x_train = np.array([[0.0], [1.0], [0.1], [1.1]])
    y_train = np.array([0, 1, 0, 1])

    x_eval = np.array([[0.0], [0.1]])
    y_eval = np.array([0, 0])

    with pytest.raises(ValueError):
        fit_classification_probe(
            x_train,
            y_train,
            x_eval,
            y_eval,
        )
