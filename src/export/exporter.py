"""Модуль экспорта моделей в Joblib и ONNX с автовалидацией через ONNX Runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import skl2onnx

from src.models.registry import ClassifierProtocol

logger = logging.getLogger(__name__)


def export_joblib(model: Union[ClassifierProtocol, Any], filepath: Union[str, Path]) -> Path:
    """
    Экспортирует модель или пайплайн в формат Joblib.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Модель сохранена в Joblib: %s", path)
    return path


def _to_numpy_float32(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Приводит входные данные к 2D np.ndarray со float32."""
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def export_onnx(
    model: Union[ClassifierProtocol, Any],
    filepath: Union[str, Path],
    sample_input: Union[np.ndarray, pd.DataFrame],
    target_opset: Optional[int] = None,
) -> Path:
    """
    Конвертирует модель в формат ONNX.
    Поддерживает: CatBoost, LightGBM, XGBoost, Scikit-learn модели и пайплайны.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    X_sample = _to_numpy_float32(sample_input)
    n_features = X_sample.shape[1]
    model_type_name = type(model).__name__.lower()

    logger.info("Конвертация модели '%s' в ONNX (n_features=%d, opset=%s)", model_type_name, n_features, target_opset)

    if "catboost" in model_type_name:
        if hasattr(model, "save_model"):
            model.save_model(str(path), format="onnx")
            logger.info("CatBoost модель экспортирована в ONNX: %s", path)
            return path

    if "lgbm" in model_type_name or "lightgbm" in model_type_name:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMLFloatTensorType

        initial_types = [("input", OnnxMLFloatTensorType([None, n_features]))]
        kwargs: Dict[str, Any] = {"initial_types": initial_types, "zipmap": False}
        if target_opset is not None:
            kwargs["target_opset"] = target_opset
        onnx_model = onnxmltools.convert_lightgbm(model, **kwargs)
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
        onnx_model = onnxmltools.convert_xgboost(model, **kwargs)
        path.write_bytes(onnx_model.SerializeToString())
        logger.info("XGBoost модель экспортирована в ONNX: %s", path)
        return path

    skl2onnx_kwargs: Dict[str, Any] = {"options": {"zipmap": False}}
    if target_opset is not None:
        skl2onnx_kwargs["target_opset"] = target_opset
    onnx_model = skl2onnx.to_onnx(
        model,
        X_sample[:1],
        **skl2onnx_kwargs,
    )
    path.write_bytes(onnx_model.SerializeToString())
    logger.info("Scikit-learn модель экспортирована в ONNX: %s", path)
    return path


def validate_onnx(
    onnx_path: Union[str, Path],
    model: Union[ClassifierProtocol, Any],
    test_batch: Union[np.ndarray, pd.DataFrame],
    atol: float = 1e-5,
) -> bool:
    """
    Проверяет валидность экспортированной ONNX-модели.
    Сравнивает выходы Python-модели и ONNX Runtime: np.testing.assert_allclose(py_probs, onnx_probs, atol=1e-5).
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX файл для валидации не найден: {path}")

    X_test = _to_numpy_float32(test_batch)

    # Получаем предсказания из Python
    if not hasattr(model, "predict_proba"):
        raise AttributeError(f"Модель {type(model).__name__} не имеет метода predict_proba для валидации.")

    py_probs = np.asarray(model.predict_proba(X_test), dtype=np.float32)

    # Запускаем сессию ONNX Runtime
    session = ort.InferenceSession(str(path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_test})

    # Парсим вероятности из выходов ONNX
    onnx_probs: Optional[np.ndarray] = None

    if len(outputs) == 1:
        onnx_probs = np.asarray(outputs[0], dtype=np.float32)
    elif len(outputs) >= 2:
        second_out = outputs[1]
        # Проверяем, не zipmap ли (список словарей)
        if isinstance(second_out, list) and len(second_out) > 0 and isinstance(second_out[0], dict):
            # Конвертируем список словарей {0: p0, 1: p1} в 2D numpy массив
            classes = sorted(list(second_out[0].keys()))
            onnx_probs = np.array(
                [[row_dict[c] for c in classes] for row_dict in second_out],
                dtype=np.float32,
            )
        else:
            onnx_probs = np.asarray(second_out, dtype=np.float32)

    if onnx_probs is None:
        raise RuntimeError("Не удалось извлечь массив вероятностей из выходов ONNX Runtime.")

    # Если форма 1D при бинарной классификации (например, для одного класса)
    if onnx_probs.ndim == 1 and py_probs.ndim == 2 and py_probs.shape[1] == 2:
        onnx_probs = np.column_stack([1.0 - onnx_probs, onnx_probs])

    # Сравнение через assert_allclose
    max_diff = float(np.max(np.abs(py_probs - onnx_probs)))
    try:
        np.testing.assert_allclose(py_probs, onnx_probs, atol=atol)
    except AssertionError as err:
        logger.error("Валидация ONNX провалена! Максимальное расхождение: %.6f > atol=%.6f", max_diff, atol)
        raise err

    logger.info(
        "[OK] Валидация ONNX Runtime пройдена (батч=%d, max_diff=%.2e, atol=%.1e)",
        len(X_test),
        max_diff,
        atol,
    )
    return True


def export_and_validate(
    model: Union[ClassifierProtocol, Any],
    test_batch: Union[np.ndarray, pd.DataFrame],
    onnx_path: Optional[Union[str, Path]] = None,
    joblib_path: Optional[Union[str, Path]] = None,
    atol: float = 1e-5,
) -> Dict[str, Path]:
    """
    Экспортирует модель в Joblib и/или ONNX с обязательной валидацией через ONNX Runtime.
    """
    artifacts: Dict[str, Path] = {}

    if joblib_path is not None:
        artifacts["joblib"] = export_joblib(model, joblib_path)

    if onnx_path is not None:
        out_onnx = export_onnx(model, onnx_path, sample_input=test_batch)
        validate_onnx(out_onnx, model, test_batch, atol=atol)
        artifacts["onnx"] = out_onnx

    return artifacts


def verify_onnx_file(
    onnx_path: Union[str, Path],
    dataset_path: Optional[Union[str, Path]] = None,
    joblib_path: Optional[Union[str, Path]] = None,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Автономная проверка работоспособности уже экспортированного ONNX-файла.
    Проверяет структуру графа, типы входов/выходов, время инференса
    и (при наличии связанной Joblib-модели) совпадение вероятностей.
    """
    import time
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
    expected_dim = primary_input.shape[1] if len(primary_input.shape) > 1 and isinstance(primary_input.shape[1], int) else None

    X_test: np.ndarray
    if dataset_path is not None and Path(dataset_path).exists():
        from src.dataset.loader import load_dataset
        from src.tuning.tuner import prepare_features

        df_X, _, _ = load_dataset(dataset_path)
        X_proc, _ = prepare_features(df_X)
        X_test = X_proc[:10]
    else:
        candidate_paths = [
            Path("data/processed/dataset.parquet"),
            Path("data/processed/dataset.csv"),
        ]
        found = False
        for cand in candidate_paths:
            if cand.exists():
                from src.dataset.loader import load_dataset
                from src.tuning.tuner import prepare_features

                df_X, _, _ = load_dataset(cand)
                X_proc, _ = prepare_features(df_X)
                X_test = X_proc[:10]
                found = True
                break
        if not found:
            dim = expected_dim if expected_dim is not None else 10
            X_test = np.random.randn(5, dim).astype(np.float32)

    t0 = time.perf_counter()
    outputs = session.run(None, {primary_input.name: X_test})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    matched_joblib: Optional[Path] = None
    if joblib_path is not None:
        p_j = Path(joblib_path)
        if p_j.exists():
            matched_joblib = p_j
    else:
        candidate_joblib = path.with_suffix(".joblib")
        if candidate_joblib.exists():
            matched_joblib = candidate_joblib

    diff: Optional[float] = None
    is_equivalent: Optional[bool] = None

    if matched_joblib is not None:
        model_py = joblib.load(matched_joblib)
        if hasattr(model_py, "predict_proba"):
            py_probs = np.asarray(model_py.predict_proba(X_test), dtype=np.float32)

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
        "batch_size": len(X_test),
        "latency_ms": latency_ms,
        "matched_joblib": str(matched_joblib) if matched_joblib else None,
        "max_diff": diff,
        "is_equivalent": is_equivalent,
        "status": "OK",
    }
