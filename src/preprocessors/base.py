"""Базовый класс для предобработчиков признаков с Scikit-Learn совместимым API."""

from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Базовый класс для всех предобработчиков данных.

    Реализует Scikit-Learn совместимый интерфейс трансформера:
      - fit(X, y=None) настраивает предобработчик и возвращает self;
      - transform(X) возвращает 2D np.ndarray (float32);
      - fit_transform(X, y=None) совмещает fit и transform;
      - get_feature_names_in() возвращает имена признаков на входе;
      - get_feature_names_out() возвращает имена признаков на выходе;
      - exclude_features позволяет исключать ненужные признаки.
    """

    def __init__(self, exclude_features: Optional[Sequence[str]] = None):
        self.exclude_features = list(exclude_features) if exclude_features is not None else None

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> BasePreprocessor:
        """Обучает предобработчик на данных и возвращает self."""
        ...

    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Применяет трансформацию к данным и возвращает 2D np.ndarray float32."""
        ...

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """Обучает предобработчик и трансформирует данные."""
        return self.fit(X, y, **fit_params).transform(X)

    def _prepare_input_df(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """
        Преобразует входные данные в DataFrame и сохраняет feature_names_in_ / n_features_in_.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            feature_names = [str(c) for c in df.columns]
        elif isinstance(X, np.ndarray):
            n_cols = X.shape[1] if X.ndim > 1 else 1
            feature_names = [f"f_{i}" for i in range(n_cols)]
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = pd.DataFrame(X)
            feature_names = [str(c) for c in df.columns]

        if self.exclude_features:
            cols_to_drop = [c for c in self.exclude_features if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                feature_names = [c for c in feature_names if c not in cols_to_drop]

        return df

    def get_feature_names_in(self) -> np.ndarray:
        """
        Возвращает имена входных признаков.
        """
        check_is_fitted(self)
        if not hasattr(self, "feature_names_in_") or self.feature_names_in_ is None:
            raise AttributeError(
                f"{type(self).__name__}: входные имена признаков неизвестны."
            )
        return np.asarray(self.feature_names_in_, dtype=object)

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """
        Возвращает имена выходных признаков.
        """
        check_is_fitted(self)
        if not hasattr(self, "feature_names_out_") or self.feature_names_out_ is None:
            raise AttributeError(
                f"{type(self).__name__}: выходные имена признаков не заданы."
            )
        return np.asarray(self.feature_names_out_, dtype=object)
