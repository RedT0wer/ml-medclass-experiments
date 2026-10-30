"""Единый пайплайн классификации: предобработка данных (preprocessor) + модель (model)."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from src.metrics.registry import calculate_metrics
from src.models.registry import ClassifierProtocol, get_model
from src.preprocessors.base import BasePreprocessor
from src.preprocessors.registry import (
    create_preprocessor,
    get_default_preprocessor_for_model,
    get_preprocessor,
)

logger = logging.getLogger(__name__)


class Pipeline(BaseEstimator, ClassifierMixin):
    """
    Пайплайн машинного обучения, инкапсулирующий предобработчик признаков и модель классификации.

    Обеспечивает полный жизненный цикл:
      - fit(): раздельная настройка препроцессора и модели без утечки данных;
      - predict() / predict_proba(): сквозной инференс на сырых данных;
      - evaluate(): расчет метрик качества на выборке;
      - evaluate_cv(): кросс-валидация полного пайплайна;
      - save() / load(): сериализация и десериализация артефактов в Joblib;
      - export_onnx(): экспорт в ONNX.
    """

    def __init__(
        self,
        preprocessor: Union[BasePreprocessor, str] = "auto",
        model: Union[ClassifierProtocol, str] = "logreg",
        model_params: Optional[Dict[str, Any]] = None,
        preprocessor_params: Optional[Dict[str, Any]] = None,
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.model_params = copy.deepcopy(model_params) if model_params is not None else {}
        self.preprocessor_params = copy.deepcopy(preprocessor_params) if preprocessor_params is not None else {}

    def _init_components(self) -> tuple[BasePreprocessor, ClassifierProtocol]:
        """Инициализирует экземпляры препроцессора и модели из строковых имен или объектов."""
        # 1. Модель
        if isinstance(self.model, str):
            model_inst = get_model(self.model, **self.model_params)
            model_name = self.model.lower().strip()
        else:
            model_inst = copy.deepcopy(self.model)
            model_name = getattr(self.model, "__class__", type(self.model)).__name__.lower()

        # 2. Препроцессор
        if isinstance(self.preprocessor, str):
            prep_name = self.preprocessor.lower().strip()
            if prep_name == "auto":
                prep_name = get_default_preprocessor_for_model(model_name)
            prep_inst = get_preprocessor(prep_name, **self.preprocessor_params)
        elif self.preprocessor is None:
            prep_name = get_default_preprocessor_for_model(model_name)
            prep_inst = get_preprocessor(prep_name, **self.preprocessor_params)
        else:
            prep_inst = copy.deepcopy(self.preprocessor)

        return prep_inst, model_inst

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
        **fit_params: Any,
    ) -> Pipeline:
        """
        Обучает препроцессор на X, преобразует X, и обучает модель на полученных признаках.
        """
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        # Сохраняем исходные имена признаков
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            self.n_features_in_ = int(len(X.columns))
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = [f"f_{i}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
            self.n_features_in_ = int(len(self.feature_names_in_))
        else:
            self.feature_names_in_ = None
            self.n_features_in_ = None

        prep_inst, model_inst = self._init_components()

        logger.debug("Обучение препроцессора %s...", type(prep_inst).__name__)
        X_proc = prep_inst.fit_transform(X, y_arr)

        self.preprocessor_ = prep_inst
        self.feature_names_out_ = getattr(prep_inst, "feature_names_out_", None)

        logger.debug("Обучение модели %s на данных формы %s...", type(model_inst).__name__, X_proc.shape)
        model_inst.fit(X_proc, y_arr, **fit_params)
        self.model_ = model_inst

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Применяет обученный препроцессор к входным данным X."""
        check_is_fitted(self, ["preprocessor_", "model_"])
        return self.preprocessor_.transform(X)

    def _prepare_for_prediction(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Подготавливает X для передачи в модель.
        Если X уже трансформирован в numpy массив с количеством признаков модели,
        передает его напрямую, иначе трансформирует через препроцессор.
        """
        check_is_fitted(self, ["preprocessor_", "model_"])

        n_model_in = getattr(self.model_, "n_features_in_", None)
        if (
            isinstance(X, np.ndarray)
            and n_model_in is not None
            and X.ndim == 2
            and X.shape[1] == n_model_in
            and self.n_features_in_ is not None
            and self.n_features_in_ != n_model_in
        ):
            return np.asarray(X, dtype=np.float32)

        try:
            return self.transform(X)
        except Exception:
            return np.asarray(X, dtype=np.float32)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Вычисляет предсказания классов."""
        X_proc = self._prepare_for_prediction(X)
        return self.model_.predict(X_proc)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Вычисляет вероятности принадлежности к классам."""
        X_proc = self._prepare_for_prediction(X)
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(f"Модель {type(self.model_).__name__} не имеет метода predict_proba.")
        return self.model_.predict_proba(X_proc)

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
    ) -> float:
        """Возвращает точность (Accuracy) классификации."""
        y_arr = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_arr == y_pred))

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
        metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """
        Оценивает качество работы пайплайна на выборке (X, y) по набору метрик.
        """
        check_is_fitted(self, ["preprocessor_", "model_"])
        y_arr = np.asarray(y)
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X) if hasattr(self.model_, "predict_proba") else None

        return calculate_metrics(
            y_true=y_arr,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_names=metrics,
        )

    def evaluate_cv(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
        cv: Union[int, Any] = 5,
        metrics: Optional[Sequence[str]] = None,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Выполняет кросс-валидацию полного пайплайна без утечки данных.
        Возвращает средние значения и стандартные отклонения метрик по фолдам.
        """
        y_arr = np.asarray(y)
        requested_metrics = [m.lower().strip() for m in metrics] if metrics else ["accuracy", "f1_macro", "roc_auc"]

        if isinstance(cv, int):
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        elif hasattr(cv, "split"):
            cv_splitter = cv
        else:
            raise ValueError(f"Неподдерживаемый тип cv: {type(cv)}")

        fold_results: Dict[str, List[float]] = {m: [] for m in requested_metrics}

        for train_idx, val_idx in cv_splitter.split(X, y_arr):
            if isinstance(X, pd.DataFrame):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y_arr[train_idx], y_arr[val_idx]

            pipe = clone(self)
            pipe.fit(X_train_fold, y_train_fold)
            eval_res = pipe.evaluate(X_val_fold, y_val_fold, metrics=requested_metrics)

            for m in requested_metrics:
                fold_results[m].append(eval_res.get(m, float("nan")))

        summary: Dict[str, Any] = {}
        for m in requested_metrics:
            vals = fold_results[m]
            mean_v = float(np.nanmean(vals))
            std_v = float(np.nanstd(vals))
            summary[m] = {
                "mean": mean_v,
                "std": std_v,
                "values": vals,
                "formatted": f"{mean_v:.4f} ± {std_v:.3f}",
            }

        return summary

    def save(self, filepath: Union[str, Path]) -> Path:
        """Сохраняет обученный пайплайн в Joblib-файл."""
        check_is_fitted(self, ["preprocessor_", "model_"])
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Пайплайн сохранен: %s", path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Pipeline:
        """Загружает сохраненный пайплайн из Joblib-файла."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл пайплайна не найден: {path}")
        pipe = joblib.load(path)
        if not isinstance(pipe, Pipeline):
            raise TypeError(f"Загруженный объект не является Pipeline: {type(pipe)}")
        logger.info("Пайплайн успешно загружен из: %s", path)
        return pipe

    # ------------------------------------------------------------------ Scikit-Learn API

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            "preprocessor": self.preprocessor,
            "model": self.model,
            "model_params": copy.deepcopy(self.model_params),
            "preprocessor_params": copy.deepcopy(self.preprocessor_params),
        }
        if deep:
            # Предоставляем прямой доступ к гиперпараметрам модели и препроцессора
            for k, v in self.model_params.items():
                params[f"model__{k}"] = v
                params[k] = v
            for k, v in self.preprocessor_params.items():
                params[f"preprocessor__{k}"] = v
        return params

    def set_params(self, **params: Any) -> Pipeline:
        for key, value in params.items():
            if key in ("preprocessor", "model"):
                setattr(self, key, value)
            elif key == "model_params":
                self.model_params = copy.deepcopy(value)
            elif key == "preprocessor_params":
                self.preprocessor_params = copy.deepcopy(value)
            elif key.startswith("model__"):
                sub_key = key[len("model__") :]
                self.model_params[sub_key] = value
            elif key.startswith("preprocessor__"):
                sub_key = key[len("preprocessor__") :]
                self.preprocessor_params[sub_key] = value
            else:
                # Если параметр совпадает с параметром модели, обновляем model_params
                self.model_params[key] = value
        return self

    def __sklearn_clone__(self) -> Pipeline:
        return Pipeline(
            preprocessor=self.preprocessor,
            model=self.model,
            model_params=copy.deepcopy(self.model_params),
            preprocessor_params=copy.deepcopy(self.preprocessor_params),
        )


def create_pipeline(
    model: Union[str, ClassifierProtocol] = "logreg",
    preprocessor: Optional[Union[str, BasePreprocessor]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    preprocessor_params: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Pipeline:
    """
    Фабрика для создания Pipeline.
    """
    m_params = copy.deepcopy(model_params or {})
    m_params.update(kwargs)

    prep: Union[str, BasePreprocessor]
    if preprocessor is None:
        prep = "auto"
    elif isinstance(preprocessor, str):
        prep = preprocessor
    else:
        prep = preprocessor

    if schema is not None and isinstance(prep, str):
        prep_inst = create_preprocessor(
            name_or_model=prep if prep != "auto" else (model if isinstance(model, str) else "logreg"),
            schema=schema,
            **(preprocessor_params or {}),
        )
        return Pipeline(
            preprocessor=prep_inst,
            model=model,
            model_params=m_params,
        )

    return Pipeline(
        preprocessor=prep,
        model=model,
        model_params=m_params,
        preprocessor_params=preprocessor_params,
    )
