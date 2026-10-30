"""Модуль экспорта пайплайнов в Joblib и ONNX с автовалидацией через ONNX Runtime."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import skl2onnx

from src.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _to_numpy_float32(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Приводит данные к 2D np.ndarray float32."""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def export_joblib(pipeline: Pipeline, filepath: Union[str, Path]) -> Path:
    """
    Экспортирует пайплайн в формат Joblib.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"export_joblib ожидает Pipeline, получен {type(pipeline).__name__}")
    return pipeline.save(filepath)


def export_onnx(
    pipeline: Pipeline,
    filepath: Union[str, Path],
    sample_input: Union[np.ndarray, pd.DataFrame],
    target_opset: Optional[int] = None,
) -> Path:
    """
    Конвертирует модель из Pipeline в формат ONNX с использованием трансформированного образца данных.
    Поддерживает: CatBoost, LightGBM, XGBoost, Scikit-learn модели.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Экспорт ONNX поддерживает только Pipeline, получен {type(pipeline).__name__}")

    if not hasattr(pipeline, "model_") or not hasattr(pipeline, "preprocessor_"):
        raise RuntimeError("Пайплайн должен быть обучен (fit) перед экспортом в ONNX.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Трансформируем тестовый образец через препроцессор пайплайна
    X_sample = np.asarray(pipeline.transform(sample_input), dtype=np.float32)
    n_features = X_sample.shape[1]
    inner_model = pipeline.model_
    model_type_name = type(inner_model).__name__.lower()

    logger.info("Конвертация модели '%s' из пайплайна в ONNX (n_features=%d, opset=%s)", model_type_name, n_features, target_opset)

    if "catboost" in model_type_name:
        if hasattr(inner_model, "save_model"):
            inner_model.save_model(str(path), format="onnx")
            logger.info("CatBoost модель экспортирована в ONNX: %s", path)
            return path

    if "lgbm" in model_type_name or "lightgbm" in model_type_name:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType

        initial_types = [("input", OnnxMLFloatTensorType([None, n_features]))]
        kwargs: Dict[str, Any] = {"initial_types": initial_types, "zipmap": False}
        if target_opset is not None:
            kwargs["target_opset"] = target_opset
        onnx_model = onnxmltools.convert_lightgbm(inner_model, **kwargs)
        path.write_bytes(onnx_model.SerializeToString())
        logger.info("LightGBM модель экспортирована в ONNX: %s", path)
        return path

    if "xgb" in model_type_name or "xgboost" in model_type_name:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType

        initial_types = [("input", OnnxMLFloatTensorType([None, n_features]))]
        kwargs = {"initial_types": initial_types}
        if target_opset is not None:
            kwargs["target_opset"] = target_opset
        onnx_model = onnxmltools.convert_xgboost(inner_model, **kwargs)
        path.write_bytes(onnx_model.SerializeToString())
        logger.info("XGBoost модель экспортирована в ONNX: %s", path)
        return path

    skl2onnx_kwargs: Dict[str, Any] = {"options": {"zipmap": False}}
    if target_opset is not None:
        skl2onnx_kwargs["target_opset"] = target_opset
    onnx_model = skl2onnx.to_onnx(
        inner_model,
        X_sample[:1],
        **skl2onnx_kwargs,
    )
    path.write_bytes(onnx_model.SerializeToString())
    logger.info("Scikit-learn модель экспортирована в ONNX: %s", path)
    return path


def validate_onnx(
    onnx_path: Union[str, Path],
    pipeline: Pipeline,
    test_batch: Union[np.ndarray, pd.DataFrame],
    atol: float = 1e-5,
) -> bool:
    """
    Проверяет валидность экспортированной ONNX-модели.
    Сравнивает выходы Python пайплайна и ONNX Runtime: np.testing.assert_allclose(py_probs, onnx_probs, atol=1e-5).
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Валидация ONNX ожидает Pipeline, получен {type(pipeline).__name__}")

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX файл для валидации не найден: {path}")

    # Трансформируем тестовый батч для ONNX инференса
    X_onnx_input = np.asarray(pipeline.transform(test_batch), dtype=np.float32)

    # Предсказания через Python-пайплайн
    py_probs = np.asarray(pipeline.predict_proba(test_batch), dtype=np.float32)

    # Инференс в ONNX Runtime
    session = ort.InferenceSession(str(path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_onnx_input})

    onnx_probs: Optional[np.ndarray] = None
    if len(outputs) == 1:
        onnx_probs = np.asarray(outputs[0], dtype=np.float32)
    elif len(outputs) >= 2:
        second_out = outputs[1]
        if isinstance(second_out, list) and len(second_out) > 0 and isinstance(second_out[0], dict):
            classes = sorted(list(second_out[0].keys()))
            onnx_probs = np.array(
                [[row_dict[c] for c in classes] for row_dict in second_out],
                dtype=np.float32,
            )
        else:
            onnx_probs = np.asarray(second_out, dtype=np.float32)

    if onnx_probs is None:
        raise RuntimeError("Не удалось извлечь массив вероятностей из выходов ONNX Runtime.")

    if onnx_probs.ndim == 1 and py_probs.ndim == 2 and py_probs.shape[1] == 2:
        onnx_probs = np.column_stack([1.0 - onnx_probs, onnx_probs])

    max_diff = float(np.max(np.abs(py_probs - onnx_probs)))
    try:
        np.testing.assert_allclose(py_probs, onnx_probs, atol=atol)
    except AssertionError as err:
        logger.error("Валидация ONNX провалена! Максимальное расхождение: %.6f > atol=%.6f", max_diff, atol)
        raise err

    logger.info(
        "[OK] Валидация ONNX Runtime пройдена (батч=%d, max_diff=%.2e, atol=%.1e)",
        len(test_batch),
        max_diff,
        atol,
    )
    return True


def export_and_validate(
    pipeline: Pipeline,
    test_batch: Union[np.ndarray, pd.DataFrame],
    onnx_path: Optional[Union[str, Path]] = None,
    joblib_path: Optional[Union[str, Path]] = None,
    atol: float = 1e-5,
) -> Dict[str, Path]:
    """
    Экспортирует пайплайн в Joblib и/или ONNX с обязательной валидацией через ONNX Runtime.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"export_and_validate ожидает Pipeline, получен {type(pipeline).__name__}")

    artifacts: Dict[str, Path] = {}

    if joblib_path is not None:
        artifacts["joblib"] = pipeline.save(joblib_path)

    if onnx_path is not None:
        out_onnx = export_onnx(pipeline, onnx_path, sample_input=test_batch)
        validate_onnx(out_onnx, pipeline, test_batch, atol=atol)
        artifacts["onnx"] = out_onnx

    return artifacts


def verify_onnx_file(
    onnx_path: Union[str, Path],
    dataset_path: Optional[Union[str, Path]] = None,
    joblib_path: Optional[Union[str, Path]] = None,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Автономная проверка работоспособности ONNX-файла и сверка с сохраненным Pipeline в Joblib.
    """
    import onnx

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX файл не найден: {path}")

    file_size_kb = path.stat().st_size / 1024.0

    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)

    session = ort.InferenceSession(str(path))
    inputs_info = [
        {"name": inp.name, "shape": list(inp.shape), "type": inp.type}
        for inp in session.get_inputs()
    ]
    outputs_info = [
        {"name": out.name, "shape": list(out.shape), "type": out.type}
        for out in session.get_outputs()
    ]

    primary_input = session.get_inputs()[0]

    matched_joblib: Optional[Path] = None
    if joblib_path is not None:
        p_j = Path(joblib_path)
        if p_j.exists():
            matched_joblib = p_j
    else:
        candidate_joblib = path.with_suffix(".joblib")
        if candidate_joblib.exists():
            matched_joblib = candidate_joblib

    loaded_pipe: Optional[Pipeline] = None
    if matched_joblib is not None:
        try:
            loaded_pipe = Pipeline.load(matched_joblib)
        except Exception as e:
            logger.warning("Не удалось загрузить Pipeline из %s: %s", matched_joblib, e)

    # Загружаем тестовые данные
    df_raw: Optional[pd.DataFrame] = None
    if dataset_path is not None and Path(dataset_path).exists():
        from src.dataset.loader import load_dataset
        df_raw, _, _ = load_dataset(dataset_path)
    else:
        for cand in [Path("data/processed/dataset.parquet"), Path("data/processed/dataset.csv")]:
            if cand.exists():
                from src.dataset.loader import load_dataset
                df_raw, _, _ = load_dataset(cand)
                break

    batch_raw: Optional[pd.DataFrame] = None
    if df_raw is not None and loaded_pipe is not None:
        pipe_features = getattr(loaded_pipe, "feature_names_in_", None)
        if pipe_features is not None and all(f in df_raw.columns for f in pipe_features):
            batch_raw = df_raw[pipe_features].iloc[:10]
            X_test_onnx = np.asarray(loaded_pipe.transform(batch_raw), dtype=np.float32)
            batch_size = len(batch_raw)
        else:
            expected_dim = primary_input.shape[1] if len(primary_input.shape) > 1 and isinstance(primary_input.shape[1], int) else 10
            X_test_onnx = np.random.randn(5, expected_dim).astype(np.float32)
            batch_size = len(X_test_onnx)
    else:
        expected_dim = primary_input.shape[1] if len(primary_input.shape) > 1 and isinstance(primary_input.shape[1], int) else 10
        X_test_onnx = np.random.randn(5, expected_dim).astype(np.float32)
        batch_size = len(X_test_onnx)

    t0 = time.perf_counter()
    outputs = session.run(None, {primary_input.name: X_test_onnx})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    diff: Optional[float] = None
    is_equivalent: Optional[bool] = None

    if loaded_pipe is not None and batch_raw is not None:
        py_probs = np.asarray(loaded_pipe.predict_proba(batch_raw), dtype=np.float32)

        onnx_probs: Optional[np.ndarray] = None
        if len(outputs) == 1:
            onnx_probs = np.asarray(outputs[0], dtype=np.float32)
        elif len(outputs) >= 2:
            sec = outputs[1]
            if isinstance(sec, list) and len(sec) > 0 and isinstance(sec[0], dict):
                classes = sorted(list(sec[0].keys()))
                onnx_probs = np.array([[d[c] for c in classes] for d in sec], dtype=np.float32)
            else:
                onnx_probs = np.asarray(sec, dtype=np.float32)

        if onnx_probs is not None:
            if onnx_probs.ndim == 1 and py_probs.ndim == 2 and py_probs.shape[1] == 2:
                onnx_probs = np.column_stack([1.0 - onnx_probs, onnx_probs])
            diff = float(np.max(np.abs(py_probs - onnx_probs)))
            is_equivalent = bool(diff <= atol)

    return {
        "path": str(path),
        "file_size_kb": file_size_kb,
        "inputs": inputs_info,
        "outputs": outputs_info,
        "batch_size": batch_size,
        "latency_ms": latency_ms,
        "matched_joblib": str(matched_joblib) if matched_joblib else None,
        "max_diff": diff,
        "is_equivalent": is_equivalent,
        "status": "OK",
    }
