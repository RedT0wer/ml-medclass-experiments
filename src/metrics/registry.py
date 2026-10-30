"""Гибкий реестр метрик классификации с поддержкой бинарного и многоклассового случаев."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

MetricFunc = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]

METRICS_REGISTRY: Dict[str, MetricFunc] = {}


def register_metric(name: str) -> Callable[[MetricFunc], MetricFunc]:
    """
    Декоратор для регистрации функции расчета метрики.

    Функция должна принимать:
        y_true: истинные классы (np.ndarray)
        y_pred: предсказанные классы (np.ndarray)
        y_prob: вероятности классов (np.ndarray, опционально)
    и возвращать float.
    """

    def decorator(fn: MetricFunc) -> MetricFunc:
        key = name.lower().strip()
        METRICS_REGISTRY[key] = fn
        logger.debug("Зарегистрирована метрика '%s'", key)
        return fn

    return decorator


def list_metrics() -> List[str]:
    """Возвращает список зарегистрированных имен метрик."""
    return sorted(list(METRICS_REGISTRY.keys()))


def get_metric(name: str) -> MetricFunc:
    """Возвращает функцию для расчета метрики по имени."""
    key = name.lower().strip()
    if key not in METRICS_REGISTRY:
        available = ", ".join(list_metrics())
        raise KeyError(f"Метрика '{name}' не зарегистрирована. Доступные метрики: {available}")
    return METRICS_REGISTRY[key]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    metric_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Рассчитывает только запрошенные метрики (или все зарегистрированные, если metric_names=None).

    Args:
        y_true: Истинные метки классов.
        y_pred: Предсказанные классы.
        y_prob: Предсказанные вероятности классов (нужны для ROC AUC, PR AUC).
        metric_names: Список имен рассчитываемых метрик.

    Returns:
        Словарь {название_метрики: значение}.
    """
    targets = metric_names if metric_names is not None else list_metrics()
    results: Dict[str, float] = {}

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    y_pr = np.asarray(y_prob) if y_prob is not None else None

    for name in targets:
        key = name.lower().strip()
        if key not in METRICS_REGISTRY:
            available = ", ".join(list_metrics())
            raise KeyError(f"Запрошенная метрика '{name}' не найдена в реестре. Доступные: {available}")

        metric_fn = METRICS_REGISTRY[key]
        try:
            val = metric_fn(y_t, y_p, y_pr, **kwargs)
            results[key] = float(val)
        except Exception as exc:
            logger.warning("Ошибка расчета метрики '%s': %s", key, exc)
            results[key] = float("nan")

    return results


@register_metric("accuracy")
def metric_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    return float(accuracy_score(y_true, y_pred))


@register_metric("balanced_accuracy")
def metric_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


@register_metric("f1_macro")
def metric_f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("f1_micro")
def metric_f1_micro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


@register_metric("precision")
def metric_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("recall")
def metric_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


@register_metric("roc_auc")
def metric_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    if y_prob is None:
        raise ValueError("Для расчета roc_auc необходим массив вероятностей y_prob")

    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes == 2:
        prob = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
        return float(roc_auc_score(y_true, prob))

    # Многоклассовый случай OVR macro
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=classes))


@register_metric("pr_auc")
def metric_pr_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> float:
    if y_prob is None:
        raise ValueError("Для расчета pr_auc необходим массив вероятностей y_prob")

    classes = np.unique(y_true)
    n_classes = len(classes)

    if n_classes == 2:
        prob = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] >= 2 else y_prob.ravel()
        return float(average_precision_score(y_true, prob))

    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return float(average_precision_score(y_bin, y_prob, average="macro"))
