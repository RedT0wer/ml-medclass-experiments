"""One-Hot препроцессор категориальных и порядковых признаков."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from src.preprocessors.base import BasePreprocessor

logger = logging.getLogger(__name__)


class OneHotPreprocessor(BasePreprocessor):
    """
    Препроцессор признаков с One-Hot кодированием категориальных колонок.

    Поддерживает:
      - автоматическое определение или явную спецификацию категориальных и порядковых признаков;
      - обработку отсутствующих категорий при инференсе;
      - сохранение имен признаков feature_names_in_ и feature_names_out_.
    """

    def __init__(
        self,
        categorical_features: Optional[Sequence[str]] = None,
        ordinal_features: Optional[Sequence[str]] = None,
        ordinal_categories: Optional[Dict[str, List[str]]] = None,
        missing_value: str = "отсутствует",
        exclude_features: Optional[Sequence[str]] = None,
    ):
        super().__init__(exclude_features=exclude_features)
        self.categorical_features = list(categorical_features) if categorical_features is not None else None
        self.ordinal_features = list(ordinal_features) if ordinal_features is not None else None
        self.ordinal_categories = dict(ordinal_categories) if ordinal_categories is not None else {}
        self.missing_value = str(missing_value)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        for col in df_filled.columns:
            if not pd.api.types.is_numeric_dtype(df_filled[col]) and col not in getattr(self, "active_num_", []):
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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> OneHotPreprocessor:
        df = self._prepare_input_df(X)
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = int(len(df.columns))

        df_filled = self._fill_missing(df)

        # Определение колонок
        if self.categorical_features is not None:
            active_cat = [c for c in self.categorical_features if c in df_filled.columns]
        else:
            active_cat = [
                c for c in df_filled.columns
                if not pd.api.types.is_numeric_dtype(df_filled[c])
                and (self.ordinal_features is None or c not in self.ordinal_features)
            ]

        if self.ordinal_features is not None:
            active_ord = [c for c in self.ordinal_features if c in df_filled.columns and c not in active_cat]
        else:
            active_ord = []

        active_num = [c for c in df_filled.columns if c not in active_cat and c not in active_ord]

        self.active_cat_ = active_cat
        self.active_ord_ = active_ord
        self.active_num_ = active_num

        transformers = []
        if active_cat:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                active_cat,
            ))

        if active_ord:
            categories_list = []
            for col in active_ord:
                if col in self.ordinal_categories:
                    categories_list.append([str(c) for c in self.ordinal_categories[col]])
                else:
                    categories_list.append(sorted(list(df_filled[col].unique())))
            transformers.append((
                "ord",
                OrdinalEncoder(
                    categories=categories_list,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                ),
                active_ord,
            ))

        if active_num:
            transformers.append((
                "num",
                "passthrough",
                active_num,
            ))

        if not transformers:
            # Если нет колонок, создаем фиктивный passthrough
            transformers.append(("empty", "passthrough", []))

        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer_.fit(df_filled, y)

        try:
            self.feature_names_out_ = [str(n) for n in self.transformer_.get_feature_names_out()]
        except Exception:
            self.feature_names_out_ = [f"f_out_{i}" for i in range(self.transform(df_filled).shape[1])]

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["transformer_", "feature_names_in_"])
        df = self._prepare_input_df(X)

        for col in self.feature_names_in_:
            if col not in df.columns:
                if col in getattr(self, "active_num_", []):
                    df[col] = np.nan
                else:
                    df[col] = self.missing_value

        df = df[self.feature_names_in_]
        df_filled = self._fill_missing(df)

        trans = self.transformer_.transform(df_filled)
        return np.asarray(trans, dtype=np.float32)
