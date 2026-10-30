"""Реестр моделей с поддержкой Scikit-Learn API (duck typing: fit, predict, predict_proba)."""

from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


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


MODELS_REGISTRY: Dict[str, ModelRegistration] = {}


def register_model(name: str) -> Callable[[Type[Any]], Type[Any]]:
    """
    Декоратор для регистрации класса модели в реестре.

    Args:
        name: Уникальное имя модели (например, 'catboost', 'logreg').

    Returns:
        Декорируемый класс.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        normalized_name = name.lower().strip()
        MODELS_REGISTRY[normalized_name] = ModelRegistration(
            name=normalized_name,
            model_class=cls,
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


def load_models_config(
    config_path: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> Dict[str, Any]:
    """
    Загружает конфигурацию моделей из YAML-файла (configs/models.yaml).
    """
    global _CONFIG_CACHE

    if config_path is None and _CONFIG_CACHE is not None and not reload:
        return copy.deepcopy(_CONFIG_CACHE)

    target_path: Optional[Path] = None
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Файл конфигурации моделей не найден: {config_path}")
        target_path = p
    else:
        candidates = [
            Path("configs/models.yaml"),
            Path("../configs/models.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "models.yaml",
        ]
        for c in candidates:
            if c.exists():
                target_path = c
                break

    if target_path is not None and target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                cfg = data if isinstance(data, dict) else {}
                if config_path is None:
                    _CONFIG_CACHE = cfg
                return copy.deepcopy(cfg)
        except Exception as exc:
            logger.warning("Не удалось прочитать конфиг моделей %s: %s", target_path, exc)
            return {}

    return {}


def get_model_params(
    name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Возвращает параметры модели, загруженные из YAML-конфигурации (configs/models.yaml).
    """
    _ = get_model_registration(name)
    cfg = load_models_config(config_path=config_path)
    normalized_name = name.lower().strip()
    params = cfg.get(normalized_name, {})
    if isinstance(params, dict):
        return copy.deepcopy(params)
    return {}


get_default_params = get_model_params


def get_model(
    name: str,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> ClassifierProtocol:
    """
    Создает и возвращает экземпляр модели с параметрами из configs/models.yaml,
    переопределенными переданными kwargs.

    Args:
        name: Имя модели в реестре.
        config_path: Путь к YAML-конфигурации параметров.
        **kwargs: Параметры, переопределяющие конфигурационные.

    Returns:
        Экземпляр модели (Duck typing: fit, predict, predict_proba).
    """
    reg = get_model_registration(name)
    params = get_model_params(name, config_path=config_path)
    params.update(kwargs)

    sig = inspect.signature(reg.model_class.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        accepted_keys = set(sig.parameters.keys()) - {"self"}
        params = {k: v for k, v in params.items() if k in accepted_keys}

    logger.debug("Создание модели '%s' с параметрами: %s", reg.name, params)
    return reg.model_class(**params)


register_model(name="logreg")(LogisticRegression)
register_model(name="random_forest")(RandomForestClassifier)


try:
    import lightgbm as lgb

    register_model(name="lightgbm")(lgb.LGBMClassifier)

except ImportError:
    logger.warning("Пакет lightgbm не установлен, модель 'lightgbm' недоступна.")


try:
    import xgboost as xgb

    register_model(name="xgboost")(xgb.XGBClassifier)

except ImportError:
    logger.warning("Пакет xgboost не установлен, модель 'xgboost' недоступна.")


try:
    import catboost as cb

    register_model(name="catboost")(cb.CatBoostClassifier)

except ImportError:
    logger.warning("Пакет catboost не установлен, модель 'catboost' недоступна.")
