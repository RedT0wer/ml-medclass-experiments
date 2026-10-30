"""Smoke-тесты для экспорта моделей в Joblib и ONNX с валидацией через ONNX Runtime."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.export.exporter import export_and_validate, export_joblib, export_onnx, validate_onnx
from src.models.registry import get_model


@pytest.fixture
def train_and_test_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.randn(40, 4).astype(np.float32)
    y = np.random.choice([0, 1], size=40)
    return X[:30], y[:30], X[30:], y[30:]


def test_joblib_export(train_and_test_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    X_train, y_train, X_test, _ = train_and_test_data
    model = get_model("logreg")
    model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "model.joblib"
        res = export_joblib(model, out_path)
        assert res.exists()

        loaded_model = joblib.load(res)
        np.testing.assert_allclose(model.predict_proba(X_test), loaded_model.predict_proba(X_test))


@pytest.mark.parametrize("model_name", ["logreg", "random_forest", "lightgbm", "xgboost", "catboost"])
def test_onnx_export_and_validation(
    model_name: str,
    train_and_test_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, y_train, X_test, _ = train_and_test_data

    model = get_model(model_name)
    model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / f"{model_name}.onnx"
        joblib_file = Path(tmpdir) / f"{model_name}.joblib"

        artifacts = export_and_validate(
            model=model,
            test_batch=X_test,
            onnx_path=onnx_file,
            joblib_path=joblib_file,
            atol=1e-5,
        )

        assert artifacts["onnx"].exists()
        assert artifacts["joblib"].exists()

        is_valid = validate_onnx(artifacts["onnx"], model, X_test, atol=1e-5)
        assert is_valid is True
