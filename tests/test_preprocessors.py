"""Тесты для пакета src.preprocessors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessors.booster import BoosterPreprocessor
from src.preprocessors.filtered import FilteredPreprocessor
from src.preprocessors.onehot import OneHotPreprocessor
from src.preprocessors.onehot_scaled import ScaledOneHotPreprocessor
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
    assert "filtered" in preps


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


def test_scaled_onehot_preprocessor(sample_df: pd.DataFrame) -> None:
    prep = ScaledOneHotPreprocessor()
    X_trans = prep.fit_transform(sample_df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.dtype == np.float32
    assert not np.isnan(X_trans).any()

def test_filtered_preprocessor(sample_df: pd.DataFrame) -> None:
    base = BoosterPreprocessor()
    filtered = FilteredPreprocessor(base_preprocessor=base, keep_indices=[0, 2])
    X_trans = filtered.fit_transform(sample_df)

    assert X_trans.shape == (5, 2)
    assert len(filtered.get_feature_names_out()) == 2


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
    assert isinstance(prep, ScaledOneHotPreprocessor)
    assert prep.categorical_features == ["gender"]
    assert prep.ordinal_features == ["severity"]
