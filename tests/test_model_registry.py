"""Smoke-тесты для реестра моделей."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.registry import (
    MODELS_REGISTRY,
    get_default_params,
    get_model,
    list_models,
    register_model,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.randn(30, 4).astype(np.float32)
    y = np.random.choice([0, 1], size=30)
    return X, y


def test_models_registered() -> None:
    models = list_models()
    expected = ["catboost", "lightgbm", "logreg", "random_forest", "xgboost"]
    for exp in expected:
        assert exp in models, f"Модель '{exp}' должна быть в реестре"


@pytest.mark.parametrize("model_name", ["logreg", "random_forest", "lightgbm", "xgboost", "catboost"])
def test_duck_typing_and_fit(model_name: str, synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_data
    model = get_model(model_name)

    assert hasattr(model, "fit"), f"{model_name} должен иметь метод fit"
    assert hasattr(model, "predict"), f"{model_name} должен иметь метод predict"
    assert hasattr(model, "predict_proba"), f"{model_name} должен иметь метод predict_proba"

    model.fit(X, y)
    preds = model.predict(X[:5])
    probs = model.predict_proba(X[:5])

    assert len(preds) == 5
    assert probs.shape == (5, 2)


def test_custom_model_registration() -> None:
    from sklearn.dummy import DummyClassifier

    @register_model("dummy_clf", default_params={"strategy": "most_frequent"})
    class CustomDummy(DummyClassifier):
        pass

    assert "dummy_clf" in list_models()
    model = get_model("dummy_clf")
    assert isinstance(model, CustomDummy)
    assert model.strategy == "most_frequent"

    model2 = get_model("dummy_clf", strategy="prior")
    assert model2.strategy == "prior"


def test_get_default_params() -> None:
    params = get_default_params("logreg")
    assert isinstance(params, dict)
    assert "C" in params
