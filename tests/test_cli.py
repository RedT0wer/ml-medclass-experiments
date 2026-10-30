"""Smoke-тесты для интерфейса командной строки cli.py на базе Pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cli import main
from src.pipeline.pipeline import Pipeline


@pytest.fixture
def temp_dataset_csv() -> str:
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 3.2, 4.0, 2.1, 1.9] * 3,
        "feature_2": [0.5, 1.2, 0.8, 2.1, 1.4, 0.9, 1.8, 2.5, 1.1, 1.0] * 3,
        "cat_feature": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 3,
        "diagnosis": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df.to_csv(tmp.name, sep=";", index=False)
    tmp.close()
    return tmp.name


def test_cli_list(capsys: pytest.CaptureFixture) -> None:
    main(["list"])
    captured = capsys.readouterr().out
    assert "catboost" in captured
    assert "roc_auc" in captured
    assert "Зарегистрированные модели" in captured
    assert "Зарегистрированные препроцессоры" in captured
    assert "Зарегистрированные валидаторы" in captured
    assert "stratified_kfold" in captured


def test_cli_prep_data() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_parquet = Path(tmpdir) / "test_out.parquet"
        main([
            "prep-data",
            "--input", "data/raw/algorithm.txt",
            "--features", "data/raw/features.csv",
            "--value-features", "data/raw/value_features.csv",
            "--output", str(out_parquet),
        ])
        assert out_parquet.exists()
        df = pd.read_parquet(out_parquet)
        assert len(df) > 0
        assert "diagnosis" in df.columns


def test_cli_tune(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        argv = [
            "tune",
            "--model", "logreg",
            "--dataset", temp_dataset_csv,
            "--validator", "stratified_kfold",
            "--search-type", "grid",
            "--output", tmpdir,
        ]
        main(argv)

        best_params_path = Path(tmpdir) / "best_params_logreg.json"
        pipeline_path = Path(tmpdir) / "best_params_logreg.joblib"
        assert best_params_path.exists()
        assert pipeline_path.exists()


def test_cli_tune_default_validator(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Без явного указания валидатора - должен использоваться default из validators.yaml
        argv = [
            "tune",
            "--model", "logreg",
            "--dataset", temp_dataset_csv,
            "--search-type", "grid",
            "--output", tmpdir,
        ]
        main(argv)

        best_params_path = Path(tmpdir) / "best_params_logreg.json"
        assert best_params_path.exists()


def test_cli_tune_unknown_validator_raises(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        argv = [
            "tune",
            "--model", "logreg",
            "--dataset", temp_dataset_csv,
            "--validator", "unknown_validator",
            "--output", tmpdir,
        ]
        with pytest.raises(KeyError):
            main(argv)


def test_cli_benchmark(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "leaderboard.csv"
        argv = [
            "benchmark",
            "--dataset", temp_dataset_csv,
            "--models", "logreg", "random_forest",
            "--metrics", "accuracy", "f1_macro",
            "--validator", "kfold",
            "--output", str(out_csv),
        ]
        main(argv)

        assert out_csv.exists()
        df_lb = pd.read_csv(out_csv)
        assert "Модель" in df_lb.columns
        assert any("accuracy" in c for c in df_lb.columns)
        assert len(df_lb) == 2


def test_cli_benchmark_default_validator(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "leaderboard.csv"
        # Без --validator - используется default
        argv = [
            "benchmark",
            "--dataset", temp_dataset_csv,
            "--models", "logreg",
            "--metrics", "accuracy",
            "--output", str(out_csv),
        ]
        main(argv)

        assert out_csv.exists()
        df_lb = pd.read_csv(out_csv)
        assert len(df_lb) == 1


def test_cli_benchmark_unknown_validator_raises(temp_dataset_csv: str) -> None:
    argv = [
        "benchmark",
        "--dataset", temp_dataset_csv,
        "--models", "logreg",
        "--validator", "non_existent_cv",
    ]
    with pytest.raises(KeyError):
        main(argv)


def test_cli_train_and_verify(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "pipeline.onnx"
        joblib_path = Path(tmpdir) / "pipeline.joblib"
        argv = [
            "train",
            "--model", "logreg",
            "--dataset", temp_dataset_csv,
            "--export-onnx", str(onnx_path),
            "--export-joblib", str(joblib_path),
        ]
        main(argv)

        assert onnx_path.exists()
        assert joblib_path.exists()

        verify_argv = [
            "verify-onnx",
            "--onnx", str(onnx_path),
            "--dataset", temp_dataset_csv,
            "--joblib", str(joblib_path),
        ]
        main(verify_argv)


def test_cli_train_with_model_params(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        joblib_path = Path(tmpdir) / "pipeline.joblib"
        custom_yaml = Path(tmpdir) / "custom_models.yaml"
        with open(custom_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"logreg": {"C": 0.25, "max_iter": 450}}, f)

        argv = [
            "train",
            "--model", "logreg",
            "--dataset", temp_dataset_csv,
            "--model-params", str(custom_yaml),
            "--export-joblib", str(joblib_path),
        ]
        main(argv)

        assert joblib_path.exists()
        loaded_pipe = Pipeline.load(joblib_path)
        assert loaded_pipe.model_.C == 0.25
        assert loaded_pipe.model_.max_iter == 450


def test_cli_train_with_model_params_underscore(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        joblib_path = Path(tmpdir) / "pipeline.joblib"
        custom_yaml = Path(tmpdir) / "custom_models.yaml"
        with open(custom_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"logreg": {"C": 0.35, "max_iter": 550}}, f)

        argv = [
            "train",
            "--model", "logreg",
            "--dataset", temp_dataset_csv,
            "--model_params", str(custom_yaml),
            "--export-joblib", str(joblib_path),
        ]
        main(argv)

        assert joblib_path.exists()
        loaded_pipe = Pipeline.load(joblib_path)
        assert loaded_pipe.model_.C == 0.35
        assert loaded_pipe.model_.max_iter == 550


def test_cli_benchmark_with_model_params(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "leaderboard.csv"
        custom_yaml = Path(tmpdir) / "custom_models.yaml"
        with open(custom_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"logreg": {"C": 0.5, "max_iter": 200}}, f)

        argv = [
            "benchmark",
            "--dataset", temp_dataset_csv,
            "--models", "logreg",
            "--model-params", str(custom_yaml),
            "--output", str(out_csv),
        ]
        main(argv)

        assert out_csv.exists()
        df_lb = pd.read_csv(out_csv)
        assert len(df_lb) == 1
        assert df_lb.iloc[0]["Модель"] == "logreg"


def test_cli_benchmark_with_model_params_underscore(temp_dataset_csv: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "leaderboard.csv"
        custom_yaml = Path(tmpdir) / "custom_models.yaml"
        with open(custom_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({"logreg": {"C": 0.8, "max_iter": 300}}, f)

        argv = [
            "benchmark",
            "--dataset", temp_dataset_csv,
            "--models", "logreg",
            "--model_params", str(custom_yaml),
            "--output", str(out_csv),
        ]
        main(argv)

        assert out_csv.exists()
        df_lb = pd.read_csv(out_csv)
        assert len(df_lb) == 1
        assert df_lb.iloc[0]["Модель"] == "logreg"

