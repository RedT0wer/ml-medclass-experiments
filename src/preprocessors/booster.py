"""Препроцессор признаков для градиентных бустингов (CatBoost, LightGBM, XGBoost)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from src.preprocessors.base import BasePreprocessor

logger = logging.getLogger(__name__)


class BoosterPreprocessor(BasePreprocessor):
    """
    Препроцессор для градиентных бустингов и древесных алгоритмов.

    Преобразует строковые и категориальные признаки в целые числа через OrdinalEncoder,
    обрабатывает пропущенные значения и возвращает 2D float32 массив для эффективного инференса.
    """

    def __init__(
        self,
        missing_value: str = "отсутствует",
        categorical_features: Optional[Sequence[str]] = None,
        ordinal_features: Optional[Sequence[str]] = None,
        ordinal_categories: Optional[Dict[str, List[str]]] = None,
        exclude_features: Optional[Sequence[str]] = None,
        unknown_value: int = -1,
        **kwargs: Any,
    ):
        super().__init__(missing_value=missing_value, exclude_features=exclude_features)
        self.categorical_features = list(categorical_features) if categorical_features is not None else None
        self.ordinal_features = list(ordinal_features) if ordinal_features is not None else None
        self.ordinal_categories = dict(ordinal_categories) if ordinal_categories is not None else {}
        self.unknown_value = int(unknown_value)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> BoosterPreprocessor:
        df = self._prepare_input_df(X)
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = int(len(df.columns))

        cat_cols, ord_cols, num_cols = self._split_columns(
            df,
            categorical_features=self.categorical_features,
            ordinal_features=self.ordinal_features,
        )
        self.categorical_cols_ = cat_cols + ord_cols
        self.numeric_cols_ = num_cols

        df_filled = self._fill_missing(df, categorical_cols=self.categorical_cols_, numeric_cols=self.numeric_cols_)

        if self.categorical_cols_:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=self.unknown_value,
                dtype=np.float32,
            )
            self.encoder_.fit(df_filled[self.categorical_cols_])
        else:
            self.encoder_ = None

        self.feature_names_out_ = list(df.columns)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["feature_names_in_", "feature_names_out_"])
        df = self._prepare_input_df(X)

        # Выравниваем порядок и наличие признаков
        for col in self.feature_names_in_:
            if col not in df.columns:
                df[col] = self.missing_value

        df = df[self.feature_names_in_]
        df_filled = self._fill_missing(df, categorical_cols=self.categorical_cols_, numeric_cols=self.numeric_cols_)

        out_cols = []
        for col in self.feature_names_in_:
            if col in self.categorical_cols_:
                cat_idx = self.categorical_cols_.index(col)
                col_enc = self.encoder_.transform(df_filled[self.categorical_cols_])[:, cat_idx]
                out_cols.append(col_enc.reshape(-1, 1))
            else:
                num_vals = pd.to_numeric(df_filled[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                out_cols.append(num_vals.reshape(-1, 1))

        if out_cols:
            return np.hstack(out_cols).astype(np.float32)
        return np.empty((len(df), 0), dtype=np.float32)
