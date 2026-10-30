"""Тесты для модуля src.validators.registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from src.validators.registry import (
    VALIDATORS_REGISTRY,
    get_default_params,
    get_default_validator_name,
    get_validator,
    get_validator_class,
    get_validator_registration,
    list_validators,
    load_validators_config,
)


def test_list_validators() -> None:
    validators = list_validators()
    assert "stratified_kfold" in validators
    assert "kfold" in validators
    assert "stratified_shuffle_split" in validators
    assert "shuffle_split" in validators
    assert "train_test_split" in validators


def test_get_validator_by_name() -> None:
    # 1. Если передали название - ищем зарегистрированный и используем
    val_skf = get_validator("stratified_kfold")
    assert isinstance(val_skf, StratifiedKFold)
    assert val_skf.n_splits == 5
    assert val_skf.shuffle is True

    val_kf = get_validator("kfold")
    assert isinstance(val_kf, KFold)
    assert val_kf.n_splits == 5
    assert val_kf.shuffle is True

    val_sss = get_validator("stratified_shuffle_split")
    assert isinstance(val_sss, StratifiedShuffleSplit)
    assert val_sss.n_splits == 5
    assert val_sss.test_size == 0.2


def test_get_validator_default() -> None:
    # 2. Если не передали название - используем default из validators.yaml
    default_name = get_default_validator_name()
    assert default_name == "stratified_kfold"

    val_default = get_validator(None)
    assert isinstance(val_default, StratifiedKFold)

    val_default_empty_str = get_validator("")
    assert isinstance(val_default_empty_str, StratifiedKFold)


def test_get_validator_unknown_name_raises() -> None:
    # 3. Если передали название и его нет в реестре - ошибка
    with pytest.raises(KeyError, match="не найден в реестре"):
        get_validator("unknown_validator_name")


def test_default_not_defined_raises() -> None:
    # 4. Если default не определен - ошибка
    with tempfile.TemporaryDirectory() as tmpdir:
        # Создаем конфиг без поля 'default'
        no_default_yaml = Path(tmpdir) / "validators_no_default.yaml"
        with open(no_default_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"stratified_kfold": {"n_splits": 3}}, f)

        with pytest.raises(KeyError, match="default"):
            get_default_validator_name(config_path=no_default_yaml)

        with pytest.raises(KeyError, match="default"):
            get_validator(None, config_path=no_default_yaml)


def test_default_empty_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_yaml = Path(tmpdir) / "validators_empty.yaml"
        with open(empty_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"default": ""}, f)

        with pytest.raises(KeyError, match="default"):
            get_default_validator_name(config_path=empty_yaml)

        with pytest.raises(KeyError, match="default"):
            get_validator(None, config_path=empty_yaml)


def test_yaml_is_source_of_params() -> None:
    # Единственный источник параметров - validators.yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_yaml = Path(tmpdir) / "custom_validators.yaml"
        cfg_data = {
            "default": "kfold",
            "kfold": {
                "n_splits": 7,
                "shuffle": False,
            },
        }
        with open(custom_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_data, f)

        # Проверяем, что дефолт стал kfold с 7 фолдами
        val = get_validator(None, config_path=custom_yaml)
        assert isinstance(val, KFold)
        assert val.n_splits == 7
        assert val.shuffle is False


def test_splitter_passthrough() -> None:
    # Если уже передан объект сплиттера, возвращаем его как есть
    splitter = KFold(n_splits=3)
    res = get_validator(splitter)
    assert res is splitter

