"""Тесты для модуля Pipeline (src.pipeline)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.pipeline.pipeline import Pipeline, create_pipeline


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_num": np.random.randn(50),
        "feature_cat": np.random.choice(["A", "B", "C"], size=50),
        "feature_ord": np.random.choice(["low", "med", "high"], size=50),
    })
    y = np.random.choice([0, 1], size=50)
    return df, y


def test_pipeline_creation_and_fit(synthetic_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_data
    pipe = create_pipeline(model="logreg", preprocessor="scaled_onehot")
    pipe.fit(X, y)

    assert hasattr(pipe, "preprocessor_")
    assert hasattr(pipe, "model_")
    assert pipe.classes_.tolist() == [0, 1]

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_pipeline_auto_preprocessor(synthetic_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_data
    # catboost автоматически ассоциируется с booster
    pipe = create_pipeline(model="catboost")
    pipe.fit(X, y)

    assert pipe.preprocessor_.__class__.__name__ == "BoosterPreprocessor"
    preds = pipe.predict(X[:5])
    assert len(preds) == 5


def test_pipeline_evaluate(synthetic_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_data
    pipe = create_pipeline(model="random_forest")
    pipe.fit(X, y)

    metrics = pipe.evaluate(X, y, metrics=["accuracy", "roc_auc", "f1_macro"])
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_pipeline_evaluate_cv(synthetic_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_data
    pipe = create_pipeline(model="logreg")
    cv_res = pipe.evaluate_cv(X, y, cv=3, metrics=["accuracy", "f1_macro"])

    assert "accuracy" in cv_res
    assert "mean" in cv_res["accuracy"]
    assert "std" in cv_res["accuracy"]
    assert len(cv_res["accuracy"]["values"]) == 3


def test_pipeline_save_and_load(synthetic_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_data
    pipe = create_pipeline(model="logreg")
    pipe.fit(X, y)
    probs_orig = pipe.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        joblib_path = Path(tmpdir) / "pipeline.joblib"
        saved_path = pipe.save(joblib_path)
        assert saved_path.exists()

        loaded_pipe = Pipeline.load(saved_path)
        probs_loaded = loaded_pipe.predict_proba(X)

        np.testing.assert_allclose(probs_orig, probs_loaded, atol=1e-6)


def test_pipeline_sklearn_clone(synthetic_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_data
    pipe = create_pipeline(model="logreg", model_params={"C": 0.5})
    pipe_cloned = clone(pipe)

    assert pipe_cloned.model_params.get("C") == 0.5
    assert not hasattr(pipe_cloned, "model_")

    pipe_cloned.fit(X, y)
    assert hasattr(pipe_cloned, "model_")
