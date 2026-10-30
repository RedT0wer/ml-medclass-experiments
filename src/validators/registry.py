"""Реестр валидаторов данных и кросс-валидационных разбиений."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import numpy as np
import yaml
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Duck-typing протокол для кросс-валидаторов (scikit-learn splitters)."""

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        ...

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        ...


@dataclass
class ValidatorRegistration:
    """Запись о зарегистрированном валидаторе."""

    name: str
    validator_class: Type[Any]
    default_params: Dict[str, Any] = field(default_factory=dict)


VALIDATORS_REGISTRY: Dict[str, ValidatorRegistration] = {}


def register_validator(
    name: str,
    default_params: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Декоратор для регистрации валидатора в реестре.

    Args:
        name: Уникальное имя валидатора (например, 'stratified_kfold').
        default_params: Дефолтные параметры для инициализации.

    Returns:
        Декорируемый класс.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        normalized_name = name.lower().strip()
        VALIDATORS_REGISTRY[normalized_name] = ValidatorRegistration(
            name=normalized_name,
            validator_class=cls,
            default_params=copy.deepcopy(default_params or {}),
        )
        logger.debug("Зарегистрирован валидатор '%s': %s", normalized_name, cls.__name__)
        return cls

    return decorator


def list_validators() -> List[str]:
    """Возвращает список имен всех зарегистрированных валидаторов."""
    return sorted(list(VALIDATORS_REGISTRY.keys()))


def get_validator_registration(name: str) -> ValidatorRegistration:
    """Возвращает метаданные регистрации валидатора."""
    normalized_name = name.lower().strip()
    if normalized_name not in VALIDATORS_REGISTRY:
        available = ", ".join(list_validators())
        raise KeyError(f"Валидатор '{name}' не найден в реестре. Доступные: {available}")
    return VALIDATORS_REGISTRY[normalized_name]


def get_validator_class(name: str) -> Type[Any]:
    """Возвращает класс зарегистрированного валидатора."""
    return get_validator_registration(name).validator_class


def load_validators_config(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Загружает параметры валидаторов из YAML конфигурации."""
    if config_path is None:
        candidates = [
            Path("configs/validators.yaml"),
            Path("../configs/validators.yaml"),
            Path(__file__).resolve().parents[2] / "configs/validators.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path is None or not Path(config_path).exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Не удалось прочитать конфиг валидаторов %s: %s", config_path, exc)
        return {}


def get_default_validator_name(
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """Возвращает имя валидатора по умолчанию (по умолчанию 'stratified_kfold')."""
    cfg = load_validators_config(config_path)
    return cfg.get("default", "stratified_kfold")


def get_default_params(
    name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Возвращает параметры по умолчанию для валидатора, совмещая код и YAML-конфиг."""
    reg = get_validator_registration(name)
    params = copy.deepcopy(reg.default_params)

    cfg = load_validators_config(config_path)
    normalized_name = name.lower().strip()
    if normalized_name in cfg and isinstance(cfg[normalized_name], dict):
        params.update(cfg[normalized_name])

    return params


class TrainTestSplitter:
    """Обертка для однократного разбиения выборки на train и test."""

    def __init__(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if y is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            yield from splitter.split(X, y)
        else:
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            yield from splitter.split(X)

    def get_n_splits(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        groups: Optional[Any] = None,
    ) -> int:
        return 1


def get_validator(
    name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """
    Создает и возвращает экземпляр валидатора.

    Args:
        name: Имя валидатора (если не указано, берется по умолчанию StratifiedKFold).
        config_path: Путь к YAML-конфигу параметров.
        **kwargs: Дополнительные параметры, переопределяющие дефолтные.

    Returns:
        Экземпляр валидатора со Scikit-Learn split() API.
    """
    if name is None or not str(name).strip():
        name = get_default_validator_name(config_path)

    reg = get_validator_registration(name)
    params = get_default_params(name, config_path=config_path)
    params.update(kwargs)
    logger.debug("Создание валидатора '%s' с параметрами: %s", reg.name, params)
    return reg.validator_class(**params)


# Регистрация стандартных валидаторов
register_validator(
    name="stratified_kfold",
    default_params={
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
)(StratifiedKFold)

register_validator(
    name="kfold",
    default_params={
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
)(KFold)

register_validator(
    name="stratified_shuffle_split",
    default_params={
        "n_splits": 5,
        "test_size": 0.2,
        "random_state": 42,
    },
)(StratifiedShuffleSplit)

register_validator(
    name="shuffle_split",
    default_params={
        "n_splits": 5,
        "test_size": 0.2,
        "random_state": 42,
    },
)(ShuffleSplit)

register_validator(
    name="time_series_split",
    default_params={
        "n_splits": 5,
    },
)(TimeSeriesSplit)

register_validator(
    name="train_test_split",
    default_params={
        "test_size": 0.2,
        "shuffle": True,
        "random_state": 42,
    },
)(TrainTestSplitter)

