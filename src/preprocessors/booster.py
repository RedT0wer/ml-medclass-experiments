"""Препроцессор признаков для градиентных бустингов (CatBoost, LightGBM, XGBoost)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

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
        exclude_features: Optional[Sequence[str]] = None,
        unknown_value: int = -1,
    ):
        super().__init__(exclude_features=exclude_features)
        self.missing_value = str(missing_value)
        self.unknown_value = int(unknown_value)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        for col in df_filled.columns:
            if not pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col] = df_filled[col].fillna(self.missing_value)
                df_filled[col] = df_filled[col].replace(
                    ["", None, np.nan, "NaN", "nan", "None", "MISSING"],
                    self.missing_value,
                )
                df_filled[col] = df_filled[col].astype(str)
            else:
                # Для числовых колонок заполняем медианой или 0
                if df_filled[col].isna().any():
                    median_val = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(0.0 if pd.isna(median_val) else median_val)
        return df_filled

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> BoosterPreprocessor:
        df = self._prepare_input_df(X)
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = int(len(df.columns))

        df_filled = self._fill_missing(df)

        self.categorical_cols_ = [
            c for c in df_filled.columns if not pd.api.types.is_numeric_dtype(df_filled[c])
        ]
        self.numeric_cols_ = [
            c for c in df_filled.columns if c not in self.categorical_cols_
        ]

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
        df_filled = self._fill_missing(df)

        # Обрабатываем колонки в исходном порядке
        out_cols = []
        for col in self.feature_names_in_:
            if col in self.categorical_cols_:
                cat_idx = self.categorical_cols_.index(col)
                # Извлекаем закодированный вектор
                col_enc = self.encoder_.transform(df_filled[self.categorical_cols_])[:, cat_idx]
                out_cols.append(col_enc.reshape(-1, 1))
            else:
                num_vals = pd.to_numeric(df_filled[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                out_cols.append(num_vals.reshape(-1, 1))

        if out_cols:
            result = np.hstack(out_cols).astype(np.float32)
        else:
            result = np.empty((len(df), 0), dtype=np.float32)

        return result
