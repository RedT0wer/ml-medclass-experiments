"""Пакет гибкого сбора и расчета метрик."""

from .registry import (
    METRICS_REGISTRY,
    calculate_metrics,
    get_metric,
    list_metrics,
    register_metric,
)

__all__ = [
    "METRICS_REGISTRY",
    "register_metric",
    "calculate_metrics",
    "get_metric",
    "list_metrics",
]
