"""Тесты для модуля экспорта Pipeline в Joblib и ONNX с автовалидацией."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.export.exporter import (
    export_and_validate,
    export_onnx,
    validate_onnx,
    verify_onnx_file,
)
from src.pipeline.pipeline import Pipeline, create_pipeline


@pytest.fixture
def train_and_test_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    np.random.seed(42)
    n_samples = 60
    df = pd.DataFrame({
        "num_feature": np.random.randn(n_samples).astype(np.float32),
        "cat_feature": np.random.choice(["cat_A", "cat_B", "cat_C"], size=n_samples),
        "ord_feature": np.random.choice(["step_1", "step_2", "step_3"], size=n_samples),
        "num_feature_2": np.random.rand(n_samples).astype(np.float32),
    })
    y = np.random.choice([0, 1], size=n_samples)
    return df.iloc[:40], y[:40], df.iloc[40:], y[40:]


def test_joblib_export(train_and_test_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    X_train, y_train, X_test, _ = train_and_test_data
    pipe = create_pipeline(model="logreg")
    pipe.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "pipeline.joblib"
        res = pipe.save(out_path)
        assert res.exists()

        loaded_pipe = Pipeline.load(res)
        assert isinstance(loaded_pipe, Pipeline)
        np.testing.assert_allclose(pipe.predict_proba(X_test), loaded_pipe.predict_proba(X_test))


@pytest.mark.parametrize("model_name", ["logreg", "random_forest", "lightgbm", "xgboost", "catboost"])
def test_onnx_export_and_validation(
    model_name: str,
    train_and_test_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
) -> None:
    X_train, y_train, X_test, _ = train_and_test_data

    pipe = create_pipeline(model=model_name)
    pipe.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / f"{model_name}.onnx"
        joblib_file = Path(tmpdir) / f"{model_name}.joblib"

        artifacts = export_and_validate(
            pipeline=pipe,
            test_batch=X_test,
            onnx_path=onnx_file,
            joblib_path=joblib_file,
            atol=1e-5,
        )

        assert artifacts["onnx"].exists()
        assert artifacts["joblib"].exists()

        is_valid = validate_onnx(artifacts["onnx"], pipe, X_test, atol=1e-5)
        assert is_valid is True

        rep = verify_onnx_file(onnx_file, joblib_path=joblib_file)
        assert rep["status"] == "OK"


def test_export_rejects_non_pipeline() -> None:
    class FakeEstimator:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "test.onnx"
        with pytest.raises(TypeError):
            export_joblib(FakeEstimator(), fake_path)  # type: ignore

        with pytest.raises(TypeError):
            export_onnx(FakeEstimator(), fake_path, sample_input=np.zeros((2, 2)))  # type: ignore
