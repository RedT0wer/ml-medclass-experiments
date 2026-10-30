"""Тесты для модуля src.tuning.tuner с поддержкой зарегистрированных валидаторов."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.model_selection import KFold, StratifiedKFold

from src.tuning.tuner import HyperparameterTuner, tune_model


@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, np.ndarray]:
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y = np.random.choice([0, 1], size=n)
    return df, y


def test_tuner_with_named_validator() -> None:
    # 1. Передаем имя зарегистрированного валидатора
    tuner = HyperparameterTuner(
        model="logreg",
        validator="kfold",
        param_grid={"C": [0.1, 1.0]},
    )
    assert tuner.validator_name == "kfold"
    assert isinstance(tuner.cv_splitter, KFold)


def test_tuner_with_default_validator() -> None:
    # 2. Не передаем имя - используется default из validators.yaml
    tuner = HyperparameterTuner(
        model="logreg",
        param_grid={"C": [0.1, 1.0]},
    )
    assert isinstance(tuner.cv_splitter, StratifiedKFold)


def test_tuner_with_unknown_validator_raises() -> None:
    # 3. Передали неизвестное имя - ошибка
    with pytest.raises(KeyError, match="не найден в реестре"):
        HyperparameterTuner(
            model="logreg",
            validator="non_existent_validator",
            param_grid={"C": [1.0]},
        )


def test_tuner_without_default_raises() -> None:
    # 4. Если default в validators.yaml не определен - ошибка
    with tempfile.TemporaryDirectory() as tmpdir:
        no_default_yaml = Path(tmpdir) / "no_default.yaml"
        with open(no_default_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"kfold": {"n_splits": 3}}, f)

        with pytest.raises(KeyError, match="default"):
            HyperparameterTuner(
                model="logreg",
                validators_config_path=no_default_yaml,
                param_grid={"C": [1.0]},
            )


def test_tuner_tune_execution(dummy_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = dummy_data
    # Тюнинг с явно указанным зарегистрированным валидатором
    res = tune_model(
        model="logreg",
        X=X,
        y=y,
        validator="kfold",
        param_grid={"C": [0.1, 1.0]},
    )
    assert res.best_score > 0
    assert "C" in res.best_params

    # Тюнинг с валидатором по умолчанию
    res_def = tune_model(
        model="logreg",
        X=X,
        y=y,
        param_grid={"C": [0.1, 1.0]},
    )
    assert res_def.best_score > 0

