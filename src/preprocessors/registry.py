"""Реестр предобработчиков данных с поддержкой конфигураций из YAML."""

from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import yaml

from src.preprocessors.base import BasePreprocessor
from src.preprocessors.booster import BoosterPreprocessor
from src.preprocessors.onehot import OneHotPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorRegistration:
    """Запись о зарегистрированном препроцессоре."""

    name: str
    preprocessor_class: Type[BasePreprocessor]


PREPROCESSORS_REGISTRY: Dict[str, PreprocessorRegistration] = {}
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def register_preprocessor(
    name: str,
) -> Callable[[Type[BasePreprocessor]], Type[BasePreprocessor]]:
    """Декоратор для регистрации класса препроцессора в реестре."""

    def decorator(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
        normalized_name = name.lower().strip()
        PREPROCESSORS_REGISTRY[normalized_name] = PreprocessorRegistration(
            name=normalized_name,
            preprocessor_class=cls,
        )
        logger.debug("Зарегистрирован препроцессор '%s': %s", normalized_name, cls.__name__)
        return cls

    return decorator


def list_preprocessors() -> List[str]:
    """Возвращает отсортированный список канонических зарегистрированных препроцессоров."""
    canonical = [k for k in PREPROCESSORS_REGISTRY.keys() if not k.endswith("_preprocessor")]
    return sorted(list(set(canonical)))


def get_preprocessor_registration(name: str) -> PreprocessorRegistration:
    """Возвращает метаданные регистрации препроцессора."""
    normalized_name = name.lower().strip()
    if normalized_name not in PREPROCESSORS_REGISTRY:
        available = ", ".join(list_preprocessors())
        raise KeyError(f"Препроцессор '{name}' не найден в реестре. Доступные препроцессоры: {available}")
    return PREPROCESSORS_REGISTRY[normalized_name]


def get_preprocessor_class(name: str) -> Type[BasePreprocessor]:
    """Возвращает класс зарегистрированного препроцессора."""
    return get_preprocessor_registration(name).preprocessor_class


def load_preprocessors_config(
    config_path: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> Dict[str, Any]:
    """Загружает файл конфигурации препроцессоров configs/preprocessors.yaml."""
    global _CONFIG_CACHE

    if config_path is None and _CONFIG_CACHE is not None and not reload:
        return copy.deepcopy(_CONFIG_CACHE)

    target_path: Optional[Path] = None
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            target_path = p
    else:
        candidates = [
            Path("configs/preprocessors.yaml"),
            Path("../configs/preprocessors.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "preprocessors.yaml",
        ]
        for c in candidates:
            if c.exists():
                target_path = c
                break

    if target_path is not None and target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            if config_path is None:
                _CONFIG_CACHE = cfg
            return copy.deepcopy(cfg)

    return {}


load_model_preprocessors_config = load_preprocessors_config


def get_preprocessor_params(
    name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Возвращает параметры препроцессора из YAML-конфигурации."""
    _ = get_preprocessor_registration(name)
    config = load_preprocessors_config(config_path=config_path)
    clean_name = name.lower().strip()
    params = config.get("preprocessors", {}).get(clean_name, {})
    if isinstance(params, dict):
        return copy.deepcopy(params)
    return {}


get_default_params = get_preprocessor_params


def get_default_preprocessor_for_model(
    model_name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """Возвращает имя препроцессора, ассоциированного с указанной моделью."""
    config = load_preprocessors_config(config_path)
    model_mapping = config.get("model_preprocessors", {})
    clean_model = model_name.lower().strip()

    if clean_model in model_mapping:
        return str(model_mapping[clean_model]).lower().strip()

    # Fallback по ключевым паттернам алгоритма
    if any(k in clean_model for k in ["cat", "lgb", "xgb", "boost", "forest", "tree"]):
        return "booster"
    return "scaled_onehot"


def get_preprocessor(
    name_or_model: str,
    schema: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> BasePreprocessor:
    """
    Фабрика создания препроцессора по имени или модели.

    Иерархия параметров:
      1. Параметры из configs/preprocessors.yaml;
      2. Извлечение колонок и категорий из schema (если передана);
      3. Явные переопределения через kwargs.
    """
    clean_name = name_or_model.lower().strip()

    # Если имя не зарегистрировано, проверяем, не является ли это именем модели
    if clean_name not in PREPROCESSORS_REGISTRY:
        mapped_name = get_default_preprocessor_for_model(clean_name, config_path=config_path)
        if mapped_name in PREPROCESSORS_REGISTRY:
            clean_name = mapped_name

    reg = get_preprocessor_registration(clean_name)
    params = get_preprocessor_params(clean_name, config_path=config_path)

    # Извлечение параметров из схемы (features)
    if schema and "features" in schema:
        features_dict = schema["features"]
        cat_features: List[str] = []
        ord_features: List[str] = []
        ord_categories: Dict[str, List[str]] = {}

        for feat_name, feat_meta in features_dict.items():
            f_type = feat_meta.get("type", "categorical")
            cats = feat_meta.get("categories", [])
            if f_type == "ordinal":
                ord_features.append(feat_name)
                if cats:
                    ord_categories[feat_name] = [str(c) for c in cats]
            elif f_type == "categorical":
                cat_features.append(feat_name)

        if cat_features and "categorical_features" not in kwargs:
            params["categorical_features"] = cat_features
        if ord_features and "ordinal_features" not in kwargs:
            params["ordinal_features"] = ord_features
        if ord_categories and "ordinal_categories" not in kwargs:
            params["ordinal_categories"] = ord_categories

    params.update(kwargs)

    # Фильтруем параметры под сигнатуру __init__ если нет **kwargs
    sig = inspect.signature(reg.preprocessor_class.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        accepted_keys = set(sig.parameters.keys()) - {"self"}
        params = {k: v for k, v in params.items() if k in accepted_keys}

    logger.debug("Создание препроцессора '%s' с параметрами: %s", reg.name, params)
    return reg.preprocessor_class(**params)


create_preprocessor = get_preprocessor


# ---------------------------------------------------------------------------
# Регистрация стандартных препроцессоров
# ---------------------------------------------------------------------------
register_preprocessor("booster")(BoosterPreprocessor)
register_preprocessor("onehot")(OneHotPreprocessor)
register_preprocessor("scaled_onehot")(OneHotPreprocessor)
