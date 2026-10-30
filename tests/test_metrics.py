"""Smoke-тесты для сбора и расчета метрик."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.registry import (
    METRICS_REGISTRY,
    calculate_metrics,
    list_metrics,
    register_metric,
)


@pytest.fixture
def binary_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.1, 0.9],
        [0.7, 0.3],
    ])
    return y_true, y_pred, y_prob


@pytest.fixture
def multiclass_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 2, 0, 2])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.7],
        [0.2, 0.3, 0.5],
        [0.9, 0.05, 0.05],
        [0.05, 0.15, 0.8],
    ])
    return y_true, y_pred, y_prob


def test_standard_metrics_registered() -> None:
    expected = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_micro",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    ]
    registered = list_metrics()
    for metric_name in expected:
        assert metric_name in registered, f"Метрика '{metric_name}' должна быть в реестре"


def test_calculate_only_requested_metrics(binary_sample: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y_true, y_pred, y_prob = binary_sample
    requested = ["f1_macro", "accuracy"]

    metrics = calculate_metrics(y_true, y_pred, y_prob, metric_names=requested)

    assert set(metrics.keys()) == set(requested)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_multiclass_metrics_calculation(multiclass_sample: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y_true, y_pred, y_prob = multiclass_sample
    all_metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert "roc_auc" in all_metrics
    assert "pr_auc" in all_metrics
    assert "f1_macro" in all_metrics
    assert not np.isnan(all_metrics["roc_auc"])
    assert not np.isnan(all_metrics["pr_auc"])


def test_custom_metric_decorator(binary_sample: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y_true, y_pred, y_prob = binary_sample

    @register_metric("custom_error_rate")
    def error_rate(yt: np.ndarray, yp: np.ndarray, ypr: np.ndarray | None = None) -> float:
        return float(np.mean(yt != yp))

    assert "custom_error_rate" in list_metrics()
    res = calculate_metrics(y_true, y_pred, metric_names=["custom_error_rate"])
    assert "custom_error_rate" in res
    assert res["custom_error_rate"] == pytest.approx(1.0 / 6.0)
