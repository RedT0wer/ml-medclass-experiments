"""One-Hot препроцессор категориальных и порядковых признаков с опциональным масштабированием."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.preprocessors.base import BasePreprocessor

logger = logging.getLogger(__name__)


class OneHotPreprocessor(BasePreprocessor):
    """
    Препроцессор признаков с One-Hot кодированием категориальных колонок.

    Поддерживает:
      - автоматическое определение или явную спецификацию категориальных и порядковых признаков;
      - сохранение заданного порядка категорий для ordinal признаков;
      - опциональное масштабирование (StandardScaler) для числовых и порядковых признаков;
      - обработку пропущенных значений и отсутствующих категорий при инференсе;
      - сохранение имен признаков feature_names_in_ и feature_names_out_.
    """

    def __init__(
        self,
        categorical_features: Optional[Sequence[str]] = None,
        ordinal_features: Optional[Sequence[str]] = None,
        ordinal_categories: Optional[Dict[str, List[str]]] = None,
        missing_value: str = "отсутствует",
        scale_numeric: bool = False,
        scale_ordinal_features: bool = False,
        exclude_features: Optional[Sequence[str]] = None,
    ):
        super().__init__(missing_value=missing_value, exclude_features=exclude_features)
        self.categorical_features = list(categorical_features) if categorical_features is not None else None
        self.ordinal_features = list(ordinal_features) if ordinal_features is not None else None
        self.ordinal_categories = dict(ordinal_categories) if ordinal_categories is not None else {}
        self.scale_numeric = bool(scale_numeric)
        self.scale_ordinal_features = bool(scale_ordinal_features)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> OneHotPreprocessor:
        df = self._prepare_input_df(X)
        self.feature_names_in_ = list(df.columns)
        self.n_features_in_ = int(len(df.columns))

        active_cat, active_ord, active_num = self._split_columns(
            df,
            categorical_features=self.categorical_features,
            ordinal_features=self.ordinal_features,
        )
        self.active_cat_ = active_cat
        self.active_ord_ = active_ord
        self.active_num_ = active_num

        df_filled = self._fill_missing(
            df,
            categorical_cols=active_cat + active_ord,
            numeric_cols=active_num,
        )

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

            ord_encoder = OrdinalEncoder(
                categories=categories_list,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            if self.scale_ordinal_features:
                ord_pipe = SkPipeline([
                    ("encode", ord_encoder),
                    ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                    ("scale", StandardScaler()),
                ])
                transformers.append(("ord", ord_pipe, active_ord))
            else:
                transformers.append(("ord", ord_encoder, active_ord))

        if active_num:
            if self.scale_numeric:
                num_pipe = SkPipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
                transformers.append(("num", num_pipe, active_num))
            else:
                transformers.append(("num", "passthrough", active_num))

        if not transformers:
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
        df_filled = self._fill_missing(
            df,
            categorical_cols=self.active_cat_ + self.active_ord_,
            numeric_cols=self.active_num_,
        )

        trans = self.transformer_.transform(df_filled)
        return np.asarray(trans, dtype=np.float32)
