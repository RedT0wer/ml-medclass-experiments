"""Модуль оптимизации гиперпараметров моделей на базе Pipeline."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.pipeline.pipeline import Pipeline, create_pipeline
from src.preprocessors.base import BasePreprocessor
from src.validators.registry import get_validator

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Результаты подбора гиперпараметров пайплайна."""

    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    scoring_metric: str
    best_pipeline: Pipeline
    cv_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результаты в словарь для сериализации."""
        clean_params: Dict[str, Any] = {}
        for k, v in self.best_params.items():
            clean_k = k.replace("model__", "")
            if isinstance(v, (np.integer, int)):
                clean_params[clean_k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_params[clean_k] = float(v)
            else:
                clean_params[clean_k] = v

        return {
            "model_name": self.model_name,
            "best_score": float(self.best_score),
            "scoring_metric": self.scoring_metric,
            "best_params": clean_params,
        }

    def save(self, filepath: Union[str, Path], save_pipeline: bool = True) -> Path:
        """Сохраняет лучшие параметры в JSON и, опционально, обученный Pipeline в Joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Результаты тюнинга сохранены в %s", path)

        if save_pipeline and self.best_pipeline is not None:
            pipe_path = path.with_suffix(".joblib")
            self.best_pipeline.save(pipe_path)
            logger.info("Лучший Pipeline сохранен в %s", pipe_path)

        return path


def load_tuning_config(
    model_name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Any]]:
    """Загружает сетку гиперпараметров для модели из YAML-конфига."""
    if config_path is None:
        candidates = [
            Path("configs/tuning.yaml"),
            Path("../configs/tuning.yaml"),
            Path(__file__).resolve().parents[2] / "configs/tuning.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(f"Файл конфигурации тюнинга не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_key = model_name.lower().strip()
    if model_key not in data:
        raise KeyError(
            f"Сетка гиперпараметров для '{model_name}' отсутствует в {config_path}. "
            f"Доступные модели в конфиге: {list(data.keys())}"
        )

    grid = data[model_key]
    if not isinstance(grid, dict):
        raise ValueError(f"Сетка гиперпараметров в {config_path} для '{model_name}' должна быть словарем.")

    return grid


class HyperparameterTuner:
    """Класс подбора гиперпараметров на базе Pipeline."""

    def __init__(
        self,
        model: str,
        preprocessor: Optional[Union[str, BasePreprocessor]] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        config_path: Optional[Union[str, Path]] = None,
        search_type: str = "grid",
        n_iter: int = 10,
        scoring: str = "f1_macro",
        validator: Optional[Union[str, Any]] = None,
        cv: Optional[Union[int, str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        schema: Optional[Dict[str, Any]] = None,
        validators_config_path: Optional[Union[str, Path]] = None,
    ):
        self.model_name = model.lower().strip()
        self.preprocessor = preprocessor
        self.schema = copy.deepcopy(schema) if schema is not None else None
        self.validators_config_path = validators_config_path

        # Разрешение валидатора
        val_target = validator if validator is not None else cv
        if val_target is not None:
            if isinstance(val_target, str):
                self.validator_name = val_target
                self.cv_splitter = get_validator(val_target, config_path=self.validators_config_path)
            elif hasattr(val_target, "split"):
                self.validator_name = type(val_target).__name__
                self.cv_splitter = val_target
            elif isinstance(val_target, int):
                self.validator_name = None
                self.cv_splitter = get_validator(None, config_path=self.validators_config_path, n_splits=val_target)
            else:
                raise TypeError(f"Неподдерживаемый тип валидатора: {type(val_target)}")
        else:
            self.validator_name = None
            self.cv_splitter = get_validator(None, config_path=self.validators_config_path)

        self.validator = self.cv_splitter
        self.cv = self.cv_splitter

        if param_grid is not None:
            self.raw_grid = copy.deepcopy(param_grid)
        else:
            self.raw_grid = load_tuning_config(self.model_name, config_path)

        # Добавляем префикс model__ для Scikit-Learn GridSearchCV / RandomizedSearchCV
        self.param_grid: Dict[str, List[Any]] = {}
        for k, v in self.raw_grid.items():
            prefixed_key = k if k.startswith("model__") or k.startswith("preprocessor__") else f"model__{k}"
            self.param_grid[prefixed_key] = v

        self.search_type = search_type.lower().strip()
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

    def tune(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> TuningResult:
        """
        Запускает подбор гиперпараметров пайплайна (препроцессор + модель) через кросс-валидацию.
        """
        logger.info(
            "Старт тюнинга пайплайна для модели '%s' (метод=%s, scoring=%s, validator=%s, вариантов=%s)",
            self.model_name,
            self.search_type,
            self.scoring,
            self.validator_name or type(self.cv_splitter).__name__,
            len(self.param_grid),
        )

        base_pipe = create_pipeline(
            model=self.model_name,
            preprocessor=self.preprocessor,
            schema=self.schema,
        )

        cv_splitter = self.cv_splitter

        y_arr = np.asarray(y)

        if self.search_type == "random":
            searcher = RandomizedSearchCV(
                estimator=base_pipe,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=cv_splitter,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                error_score="raise",
            )
        elif self.search_type == "grid":
            searcher = GridSearchCV(
                estimator=base_pipe,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=cv_splitter,
                n_jobs=self.n_jobs,
                error_score="raise",
            )
        else:
            raise ValueError(f"Неизвестный тип поиска: {self.search_type}. Допустимы 'grid' и 'random'.")

        searcher.fit(X, y_arr)

        best_params_clean = {
            k.replace("model__", ""): v for k, v in searcher.best_params_.items()
        }

        logger.info(
            "Тюнинг завершен: лучший %s = %.4f при параметрах: %s",
            self.scoring,
            searcher.best_score_,
            best_params_clean,
        )

        return TuningResult(
            model_name=self.model_name,
            best_params=best_params_clean,
            best_score=float(searcher.best_score_),
            scoring_metric=self.scoring,
            best_pipeline=searcher.best_estimator_,
            cv_results=searcher.cv_results_,
        )


def prepare_features(
    X: Union[pd.DataFrame, np.ndarray],
    preprocessor: Optional[BasePreprocessor] = None,
) -> tuple[np.ndarray, BasePreprocessor]:
    """Предобрабатывает признаки с помощью препроцессора."""
    from src.preprocessors.registry import get_preprocessor

    prep = preprocessor or get_preprocessor("scaled_onehot")
    if not hasattr(prep, "feature_names_out_"):
        X_trans = prep.fit_transform(X)
    else:
        X_trans = prep.transform(X)
    return np.asarray(X_trans, dtype=np.float32), prep


def tune_model(
    model: str,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    preprocessor: Optional[Union[str, BasePreprocessor]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    config_path: Optional[Union[str, Path]] = None,
    search_type: str = "grid",
    n_iter: int = 10,
    scoring: str = "f1_macro",
    validator: Optional[Union[str, Any]] = None,
    cv: Optional[Union[int, str, Any]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    schema: Optional[Dict[str, Any]] = None,
    validators_config_path: Optional[Union[str, Path]] = None,
) -> TuningResult:
    """Удобная функция-фасад для подбора гиперпараметров Pipeline."""
    tuner = HyperparameterTuner(
        model=model,
        preprocessor=preprocessor,
        param_grid=param_grid,
        config_path=config_path,
        search_type=search_type,
        n_iter=n_iter,
        scoring=scoring,
        validator=validator,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        schema=schema,
        validators_config_path=validators_config_path,
    )
    return tuner.tune(X, y)

