"""Реестр моделей с поддержкой Scikit-Learn API (duck typing: fit, predict, predict_proba)."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Duck-typing протокол для классификаторов с API Scikit-Learn."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassifierProtocol:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class ModelRegistration:
    """Запись о зарегистрированной модели."""

    name: str
    model_class: Type[ClassifierProtocol]
    default_params: Dict[str, Any] = field(default_factory=dict)


MODELS_REGISTRY: Dict[str, ModelRegistration] = {}


def register_model(
    name: str,
    default_params: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Декоратор для регистрации класса модели в реестре.

    Args:
        name: Уникальное имя модели (например, 'catboost', 'logreg').
        default_params: Дефолтные гиперпараметры для инициализации модели.

    Returns:
        Декорируемый класс.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        normalized_name = name.lower().strip()
        MODELS_REGISTRY[normalized_name] = ModelRegistration(
            name=normalized_name,
            model_class=cls,
            default_params=copy.deepcopy(default_params or {}),
        )
        logger.debug("Зарегистрирована модель '%s': %s", normalized_name, cls.__name__)
        return cls

    return decorator


def list_models() -> List[str]:
    """Возвращает список имен всех зарегистрированных моделей."""
    return sorted(list(MODELS_REGISTRY.keys()))


def get_model_registration(name: str) -> ModelRegistration:
    """Возвращает метаданные регистрации модели."""
    normalized_name = name.lower().strip()
    if normalized_name not in MODELS_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Модель '{name}' не найдена в реестре. Доступные модели: {available}")
    return MODELS_REGISTRY[normalized_name]


def get_model_class(name: str) -> Type[Any]:
    """Возвращает класс зарегистрированной модели."""
    return get_model_registration(name).model_class


def get_default_params(name: str) -> Dict[str, Any]:
    """Возвращает копию словаря дефолтных гиперпараметров модели."""
    return copy.deepcopy(get_model_registration(name).default_params)


def get_model(name: str, **kwargs: Any) -> ClassifierProtocol:
    """
    Создает и возвращает экземпляр модели с дефолтными гиперпараметрами,
    переопределенными переданными kwargs.

    Args:
        name: Имя модели в реестре.
        **kwargs: Параметры, переопределяющие дефолтные.

    Returns:
        Экземпляр модели (Duck typing: fit, predict, predict_proba).
    """
    reg = get_model_registration(name)
    params = copy.deepcopy(reg.default_params)
    params.update(kwargs)
    logger.debug("Создание модели '%s' с параметрами: %s", reg.name, params)
    return reg.model_class(**params)


register_model(
    name="logreg",
    default_params={
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    },
)(LogisticRegression)


register_model(
    name="random_forest",
    default_params={
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    },
)(RandomForestClassifier)


try:
    import lightgbm as lgb

    register_model(
        name="lightgbm",
        default_params={
            "n_estimators": 100,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        },
    )(lgb.LGBMClassifier)

except ImportError:
    logger.warning("Пакет lightgbm не установлен, модель 'lightgbm' недоступна.")


try:
    import xgboost as xgb

    register_model(
        name="xgboost",
        default_params={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        },
    )(xgb.XGBClassifier)

except ImportError:
    logger.warning("Пакет xgboost не установлен, модель 'xgboost' недоступна.")


try:
    import catboost as cb

    register_model(
        name="catboost",
        default_params={
            "iterations": 200,
            "depth": 5,
            "learning_rate": 0.05,
            "random_seed": 42,
            "verbose": 0,
            "thread_count": -1,
        },
    )(cb.CatBoostClassifier)

except ImportError:
    logger.warning("Пакет catboost не установлен, модель 'catboost' недоступна.")
