"""Базовый класс для предобработчиков признаков с Scikit-Learn совместимым API."""

from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
      - exclude_features позволяет исключать ненужные признаки;
      - from_schema() фабрика для создания препроцессора из JSON-схемы датасета.
    """

    def __init__(
        self,
        missing_value: str = "отсутствует",
        exclude_features: Optional[Sequence[str]] = None,
    ):
        self.missing_value = str(missing_value)
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

    def _split_columns(
        self,
        df: pd.DataFrame,
        categorical_features: Optional[Sequence[str]] = None,
        ordinal_features: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Разделяет колонки DataFrame на категориальные, порядковые и числовые.
        """
        columns = list(df.columns)
        if categorical_features is not None:
            active_cat = [c for c in categorical_features if c in columns]
        else:
            active_cat = [
                c for c in columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and (ordinal_features is None or c not in ordinal_features)
            ]

        if ordinal_features is not None:
            active_ord = [c for c in ordinal_features if c in columns and c not in active_cat]
        else:
            active_ord = []

        active_num = [c for c in columns if c not in active_cat and c not in active_ord]
        return active_cat, active_ord, active_num

    def _fill_missing(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[Sequence[str]] = None,
        numeric_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Заполняет пропуски в категориальных (строкой missing_value) и числовых колонках.
        """
        df_filled = df.copy()
        cat_set = set(categorical_cols) if categorical_cols is not None else None
        num_set = set(numeric_cols) if numeric_cols is not None else None

        for col in df_filled.columns:
            is_cat = (col in cat_set) if cat_set is not None else (
                (col not in num_set) if num_set is not None else not pd.api.types.is_numeric_dtype(df_filled[col])
            )

            if is_cat:
                df_filled[col] = df_filled[col].fillna(self.missing_value)
                df_filled[col] = df_filled[col].replace(
                    ["", None, np.nan, "NaN", "nan", "None", "MISSING"],
                    self.missing_value,
                )
                df_filled[col] = df_filled[col].astype(str)
            else:
                df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce")
                if df_filled[col].isna().any():
                    median_val = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(0.0 if pd.isna(median_val) else median_val)

        return df_filled

    @classmethod
    def from_schema(
        cls,
        schema: Dict[str, Any],
        **kwargs: Any,
    ) -> BasePreprocessor:
        """
        Фабричный метод создания препроцессора из словаря схемы данных (.schema.json).
        """
        params = copy.deepcopy(kwargs)
        features_dict = schema.get("features", {})

        cat_features: List[str] = []
        ord_features: List[str] = []
        ord_categories: Dict[str, List[str]] = {}

        for feat_name, feat_meta in features_dict.items():
            f_type = feat_meta.get("type", "categorical")
            cats = feat_meta.get("categories", [])
            if f_type == "ordinal":
                ord_features.append(feat_name)
                if cats:
                    ord_categories[feat_name] = [str(c) for c in cats]
            elif f_type == "categorical":
                cat_features.append(feat_name)

        if "categorical_features" not in params and cat_features:
            params["categorical_features"] = cat_features
        if "ordinal_features" not in params and ord_features:
            params["ordinal_features"] = ord_features
        if "ordinal_categories" not in params and ord_categories:
            params["ordinal_categories"] = ord_categories

        return cls(**params)

    def get_feature_names_in(self) -> np.ndarray:
        """Возвращает имена входных признаков."""
        check_is_fitted(self)
        if not hasattr(self, "feature_names_in_") or self.feature_names_in_ is None:
            raise AttributeError(
                f"{type(self).__name__}: входные имена признаков неизвестны."
            )
        return np.asarray(self.feature_names_in_, dtype=object)

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Возвращает имена выходных признаков."""
        check_is_fitted(self)
        if not hasattr(self, "feature_names_out_") or self.feature_names_out_ is None:
            raise AttributeError(
                f"{type(self).__name__}: выходные имена признаков не заданы."
            )
        return np.asarray(self.feature_names_out_, dtype=object)
