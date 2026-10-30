"""Тесты для пакета src.preprocessors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessors.booster import BoosterPreprocessor
from src.preprocessors.onehot import OneHotPreprocessor
from src.preprocessors.registry import (
    create_preprocessor,
    get_default_preprocessor_for_model,
    get_preprocessor,
    list_preprocessors,
    register_preprocessor,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25, 45, 65, 30, 50],
        "gender": ["муж", "жен", "муж", "жен", "муж"],
        "severity": ["легкая", "средняя", "тяжелая", "легкая", "тяжелая"],
        "score": [1.2, np.nan, 3.4, 2.1, 4.0],
    })


def test_list_preprocessors() -> None:
    preps = list_preprocessors()
    assert "booster" in preps
    assert "onehot" in preps
    assert "scaled_onehot" in preps


def test_booster_preprocessor(sample_df: pd.DataFrame) -> None:
    prep = BoosterPreprocessor()
    X_trans = prep.fit_transform(sample_df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.dtype == np.float32
    assert X_trans.shape == (5, 4)
    assert len(prep.get_feature_names_out()) == 4
    assert len(prep.get_feature_names_in()) == 4


def test_onehot_preprocessor(sample_df: pd.DataFrame) -> None:
    prep = OneHotPreprocessor()
    X_trans = prep.fit_transform(sample_df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.dtype == np.float32
    # gender (2) + severity (3) + age (1) + score (1) = 7
    assert X_trans.shape[1] >= 4
    assert len(prep.get_feature_names_out()) == X_trans.shape[1]


def test_preprocessor_exclude_features(sample_df: pd.DataFrame) -> None:
    prep = BoosterPreprocessor(exclude_features=["score"])
    X_trans = prep.fit_transform(sample_df)

    assert X_trans.shape == (5, 3)
    assert "score" not in prep.get_feature_names_out()


def test_get_default_preprocessor_for_model() -> None:
    assert get_default_preprocessor_for_model("catboost") == "booster"
    assert get_default_preprocessor_for_model("logreg") == "scaled_onehot"
    assert get_default_preprocessor_for_model("xgboost") == "booster"


def test_create_preprocessor_with_schema() -> None:
    schema = {
        "features": {
            "gender": {"type": "categorical", "categories": ["муж", "жен"]},
            "severity": {"type": "ordinal", "categories": ["легкая", "средняя", "тяжелая"]},
        }
    }
    prep = create_preprocessor("scaled_onehot", schema=schema)
    assert isinstance(prep, OneHotPreprocessor)
    assert prep.categorical_features == ["gender"]
    assert prep.ordinal_features == ["severity"]


def test_preprocessor_from_schema_classmethod() -> None:
    schema = {
        "features": {
            "gender": {"type": "categorical", "categories": ["муж", "жен"]},
            "severity": {"type": "ordinal", "categories": ["легкая", "средняя", "тяжелая"]},
        }
    }
    prep = OneHotPreprocessor.from_schema(schema)
    assert isinstance(prep, OneHotPreprocessor)
    assert prep.categorical_features == ["gender"]
    assert prep.ordinal_features == ["severity"]
    assert prep.ordinal_categories == {"severity": ["легкая", "средняя", "тяжелая"]}


def test_get_preprocessor_from_model_name() -> None:
    prep_cb = get_preprocessor("catboost")
    assert isinstance(prep_cb, BoosterPreprocessor)

    prep_lr = get_preprocessor("logreg")
    assert isinstance(prep_lr, OneHotPreprocessor)


def test_onehot_scaling_toggle(sample_df: pd.DataFrame) -> None:
    prep_unscaled = OneHotPreprocessor(scale_numeric=False)
    res_unscaled = prep_unscaled.fit_transform(sample_df)

    prep_scaled = OneHotPreprocessor(scale_numeric=True)
    res_scaled = prep_scaled.fit_transform(sample_df)

    assert res_unscaled.shape == res_scaled.shape
    # В масштабированной версии значения числовой колонки age центрированы
    assert not np.array_equal(res_unscaled, res_scaled)


def test_pipeline_with_schema_integration(sample_df: pd.DataFrame) -> None:
    from src.pipeline.pipeline import create_pipeline

    schema = {
        "features": {
            "gender": {"type": "categorical", "categories": ["муж", "жен"]},
            "severity": {"type": "ordinal", "categories": ["легкая", "средняя", "тяжелая"]},
        }
    }
    y = np.array([0, 1, 0, 1, 0])
    pipe = create_pipeline(model="logreg", preprocessor="scaled_onehot", schema=schema)
    pipe.fit(sample_df, y)

    assert hasattr(pipe, "preprocessor_")
    assert pipe.preprocessor_.categorical_features == ["gender"]
    assert pipe.preprocessor_.ordinal_features == ["severity"]

    preds = pipe.predict(sample_df)
    assert len(preds) == len(sample_df)
