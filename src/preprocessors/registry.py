"""Реестр предобработчиков данных с поддержкой конфигураций из YAML."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import yaml

from src.preprocessors.base import BasePreprocessor
from src.preprocessors.booster import BoosterPreprocessor
from src.preprocessors.filtered import FilteredPreprocessor
from src.preprocessors.onehot import OneHotPreprocessor
from src.preprocessors.onehot_scaled import ScaledOneHotPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorRegistration:
    """Запись о зарегистрированном препроцессоре."""

    name: str
    preprocessor_class: Type[BasePreprocessor]
    default_params: Dict[str, Any] = field(default_factory=dict)


PREPROCESSORS_REGISTRY: Dict[str, PreprocessorRegistration] = {}


def register_preprocessor(
    name: str,
    default_params: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[BasePreprocessor]], Type[BasePreprocessor]]:
    """Декоратор для регистрации класса препроцессора в реестре."""

    def decorator(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
        normalized_name = name.lower().strip()
        PREPROCESSORS_REGISTRY[normalized_name] = PreprocessorRegistration(
            name=normalized_name,
            preprocessor_class=cls,
            default_params=copy.deepcopy(default_params or {}),
        )
        logger.debug("Зарегистрирован препроцессор '%s': %s", normalized_name, cls.__name__)
        return cls

    return decorator


def list_preprocessors() -> List[str]:
    """Возвращает список имен всех зарегистрированных препроцессоров."""
    return sorted(list(PREPROCESSORS_REGISTRY.keys()))


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


def get_default_params(name: str) -> Dict[str, Any]:
    """Возвращает копию словаря дефолтных параметров препроцессора."""
    return copy.deepcopy(get_preprocessor_registration(name).default_params)


def load_preprocessors_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Загружает файл конфигурации препроцессоров configs/preprocessors.yaml."""
    if config_path is None:
        candidates = [
            Path("configs/preprocessors.yaml"),
            Path("../configs/preprocessors.yaml"),
            Path(__file__).resolve().parents[2] / "configs/preprocessors.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {}


load_model_preprocessors_config = load_preprocessors_config


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

    # Fallback по ключевым словам модели
    if any(k in clean_model for k in ["cat", "lgb", "xgb", "boost", "forest", "tree"]):
        return "booster"
    return "scaled_onehot"


def get_preprocessor(name: str, **kwargs: Any) -> BasePreprocessor:
    """Создает и возвращает экземпляр препроцессора по имени."""
    clean_name = name.lower().strip()

    # Если передано имя модели вместо препроцессора, разрешаем в соответствующий препроцессор
    if clean_name not in PREPROCESSORS_REGISTRY:
        mapped_name = get_default_preprocessor_for_model(clean_name)
        if mapped_name in PREPROCESSORS_REGISTRY:
            clean_name = mapped_name

    reg = get_preprocessor_registration(clean_name)

    # Загружаем переопределения параметров из preprocessors.yaml если есть
    config = load_preprocessors_config()
    cfg_params = config.get("preprocessors", {}).get(clean_name, {})

    params = copy.deepcopy(reg.default_params)
    params.update(cfg_params)
    params.update(kwargs)

    logger.debug("Создание препроцессора '%s' с параметрами: %s", reg.name, params)
    return reg.preprocessor_class(**params)


def create_preprocessor(
    name_or_model: str,
    schema: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BasePreprocessor:
    """Фабрика препроцессора: создает инстанс по имени модели или препроцессора."""
    clean_name = name_or_model.lower().strip()

    if clean_name in PREPROCESSORS_REGISTRY:
        prep_name = clean_name
    else:
        prep_name = get_default_preprocessor_for_model(clean_name)

    prep_params = copy.deepcopy(kwargs)

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
                ord_categories[feat_name] = [str(c) for c in cats]
            elif f_type == "categorical":
                cat_features.append(feat_name)

        if "categorical_features" not in prep_params and cat_features:
            prep_params["categorical_features"] = cat_features
        if "ordinal_features" not in prep_params and ord_features:
            prep_params["ordinal_features"] = ord_features
        if "ordinal_categories" not in prep_params and ord_categories:
            prep_params["ordinal_categories"] = ord_categories

    return get_preprocessor(prep_name, **prep_params)


# Регистрация стандартных препроцессоров
register_preprocessor("booster", default_params={"missing_value": "отсутствует"})(BoosterPreprocessor)
register_preprocessor("onehot", default_params={"missing_value": "отсутствует"})(OneHotPreprocessor)
register_preprocessor(
    "scaled_onehot",
    default_params={"missing_value": "отсутствует", "scale_ordinal_features": True},
)(ScaledOneHotPreprocessor)
register_preprocessor("filtered", default_params={})(FilteredPreprocessor)

# Алиасы для обратной совместимости конфигураций
register_preprocessor("catboost_preprocessor", default_params={"missing_value": "отсутствует"})(BoosterPreprocessor)
register_preprocessor("xgboost_preprocessor", default_params={"missing_value": "отсутствует"})(BoosterPreprocessor)
register_preprocessor("lightgbm_preprocessor", default_params={"missing_value": "отсутствует"})(BoosterPreprocessor)
register_preprocessor("random_forest_preprocessor", default_params={"missing_value": "отсутствует"})(BoosterPreprocessor)
register_preprocessor(
    "logreg_preprocessor",
    default_params={"missing_value": "отсутствует", "scale_ordinal_features": True},
)(ScaledOneHotPreprocessor)
register_preprocessor("onehot_preprocessor", default_params={"missing_value": "отсутствует"})(OneHotPreprocessor)
register_preprocessor(
    "scaled_onehot_preprocessor",
    default_params={"missing_value": "отсутствует", "scale_ordinal_features": True},
)(ScaledOneHotPreprocessor)
