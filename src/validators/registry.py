"""Реестр валидаторов данных и кросс-валидационных разбиений."""

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


VALIDATORS_REGISTRY: Dict[str, ValidatorRegistration] = {}


def register_validator(
    name: str,
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Декоратор для регистрации валидатора в реестре.

    Args:
        name: Уникальное имя валидатора (например, 'stratified_kfold').

    Returns:
        Декорируемый класс.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        normalized_name = name.lower().strip()
        VALIDATORS_REGISTRY[normalized_name] = ValidatorRegistration(
            name=normalized_name,
            validator_class=cls,
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
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Файл конфигурации валидаторов не найден: {config_path}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Не удалось прочитать конфиг валидаторов %s: %s", config_path, exc)
            return {}

    candidates = [
        Path("configs/validators.yaml"),
        Path("../configs/validators.yaml"),
        Path(__file__).resolve().parents[2] / "configs" / "validators.yaml",
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    return data if isinstance(data, dict) else {}
            except Exception as exc:
                logger.warning("Не удалось прочитать конфиг валидаторов %s: %s", c, exc)
                return {}

    return {}


def get_default_validator_name(
    config_path: Optional[Union[str, Path]] = None,
) -> str:
    """Возвращает имя валидатора по умолчанию из validators.yaml."""
    cfg = load_validators_config(config_path)
    default_name = cfg.get("default")
    if not default_name or not str(default_name).strip():
        raise KeyError("Валидатор по умолчанию ('default') не определен в конфигурации validators.yaml")
    return str(default_name).lower().strip()


def get_validator_params(
    name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Возвращает параметры валидатора, загруженные из YAML-конфигурации."""
    _ = get_validator_registration(name)
    cfg = load_validators_config(config_path)
    normalized_name = name.lower().strip()
    params = cfg.get(normalized_name, {})
    if isinstance(params, dict):
        return copy.deepcopy(params)
    return {}


get_default_params = get_validator_params


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
    name: Optional[Union[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Any:
    """
    Создает и возвращает экземпляр валидатора.

    Args:
        name: Имя валидатора (если не указано, берется default из validators.yaml).
        config_path: Путь к YAML-конфигу параметров.
        **kwargs: Дополнительные параметры, переопределяющие дефолтные.

    Returns:
        Экземпляр валидатора со Scikit-Learn split() API.

    Raises:
        KeyError: Если валидатор с указанным именем не зарегистрирован или если default не определен.
    """
    if name is not None and hasattr(name, "split") and not isinstance(name, str):
        return name

    if name is None or (isinstance(name, str) and not name.strip()):
        name = get_default_validator_name(config_path)

    val_name = str(name).lower().strip()
    reg = get_validator_registration(val_name)
    params = get_validator_params(val_name, config_path=config_path)
    params.update(kwargs)

    if params.get("shuffle") is False and "random_state" in params:
        params["random_state"] = None

    sig = inspect.signature(reg.validator_class.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_var_keyword:
        accepted_keys = set(sig.parameters.keys()) - {"self"}
        params = {k: v for k, v in params.items() if k in accepted_keys}

    logger.debug("Создание валидатора '%s' с параметрами: %s", reg.name, params)
    return reg.validator_class(**params)


# Регистрация стандартных валидаторов
register_validator(name="stratified_kfold")(StratifiedKFold)
register_validator(name="kfold")(KFold)
register_validator(name="stratified_shuffle_split")(StratifiedShuffleSplit)
register_validator(name="shuffle_split")(ShuffleSplit)
register_validator(name="time_series_split")(TimeSeriesSplit)
register_validator(name="train_test_split")(TrainTestSplitter)
