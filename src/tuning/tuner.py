"""Модуль подбора гиперпараметров моделей классификации."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.registry import ClassifierProtocol, get_model

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Результаты подбора гиперпараметров."""

    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    scoring_metric: str
    best_estimator: ClassifierProtocol
    cv_results: Dict[str, Any] = field(default_factory=dict)
    preprocessor: Optional[ColumnTransformer] = None

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результаты в словарь для сериализации."""
        clean_params: Dict[str, Any] = {}
        for k, v in self.best_params.items():
            if isinstance(v, (np.integer, int)):
                clean_params[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_params[k] = float(v)
            else:
                clean_params[k] = v

        return {
            "model_name": self.model_name,
            "best_score": float(self.best_score),
            "scoring_metric": self.scoring_metric,
            "best_params": clean_params,
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Сохраняет лучшие параметры в JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Результаты тюнинга сохранены в %s", path)
        return path


def load_tuning_config(
    model_name: str,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Any]]:
    """
    Загружает сетку гиперпараметров для модели из YAML-конфига.
    """
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


def prepare_features(
    X: Union[pd.DataFrame, np.ndarray],
    preprocessor: Optional[ColumnTransformer] = None,
) -> Tuple[np.ndarray, Optional[ColumnTransformer]]:
    """
    Автоматически кодирует строковые/категориальные признаки для совместимости со всеми моделями.
    """
    if isinstance(X, pd.DataFrame):
        cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        if not cat_cols:
            return X.to_numpy(dtype=np.float32), None

        if preprocessor is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                    ("num", StandardScaler(), num_cols),
                ],
                remainder="passthrough",
            )
            X_trans = preprocessor.fit_transform(X)
        else:
            X_trans = preprocessor.transform(X)

        return np.asarray(X_trans, dtype=np.float32), preprocessor

    return np.asarray(X, dtype=np.float32), None


class HyperparameterTuner:
    """
    Класс подбора гиперпараметров моделей.
    """

    def __init__(
        self,
        model: Union[str, ClassifierProtocol],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        config_path: Optional[Union[str, Path]] = None,
        search_type: str = "grid",  # 'grid' или 'random'
        n_iter: int = 10,
        scoring: str = "f1_macro",
        cv: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        if isinstance(model, str):
            self.model_name = model.lower().strip()
            self.base_model = get_model(self.model_name)
        else:
            self.base_model = model
            self.model_name = getattr(model, "__class__", type(model)).__name__.lower()

        if param_grid is not None:
            self.param_grid = copy.deepcopy(param_grid)
        else:
            self.param_grid = load_tuning_config(self.model_name, config_path)

        self.search_type = search_type.lower()
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def tune(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
    ) -> TuningResult:
        """
        Запускает поиск гиперпараметров на предоставленных данных.
        """
        logger.info(
            "Старт тюнинга модели '%s' (метод=%s, scoring=%s, cv=%d, вариантов=%s)",
            self.model_name,
            self.search_type,
            self.scoring,
            self.cv,
            len(self.param_grid),
        )

        X_proc, preprocessor = prepare_features(X)
        y_arr = np.asarray(y)

        cv_splitter = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        if self.search_type == "random":
            searcher = RandomizedSearchCV(
                estimator=self.base_model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=cv_splitter,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                error_score="raise",
            )
        else:
            searcher = GridSearchCV(
                estimator=self.base_model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=cv_splitter,
                n_jobs=self.n_jobs,
                error_score="raise",
            )

        searcher.fit(X_proc, y_arr)

        logger.info(
            "Тюнинг завершен. Лучший скор (%s): %.4f. Лучшие параметры: %s",
            self.scoring,
            searcher.best_score_,
            searcher.best_params_,
        )

        return TuningResult(
            model_name=self.model_name,
            best_params=searcher.best_params_,
            best_score=float(searcher.best_score_),
            scoring_metric=self.scoring,
            best_estimator=searcher.best_estimator_,
            cv_results=searcher.cv_results_,
            preprocessor=preprocessor,
        )


def tune_model(
    model: Union[str, ClassifierProtocol],
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    config_path: Optional[Union[str, Path]] = None,
    search_type: str = "grid",
    n_iter: int = 10,
    scoring: str = "f1_macro",
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> TuningResult:
    """Удобная функция запуска тюнинга в одну строчку для ноутбуков и скриптов."""
    tuner = HyperparameterTuner(
        model=model,
        param_grid=param_grid,
        config_path=config_path,
        search_type=search_type,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return tuner.tune(X, y)
