from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from .base import BasePreprocessor


class ScaledOneHotPreprocessor(BasePreprocessor):
    """
    One-hot препроцессор со встроенным масштабированием числовых
    (и, опционально, порядковых) признаков.

    - numeric (passthrough) всегда проходит через SimpleImputer + StandardScaler;
    - ordinal можно дополнительно масштабировать флагом scale_ordinal_features;
    - one-hot ВСЕГДА остаётся как есть (0/1, не масштабируется).
    """

    def __init__(
        self,
        categorical_features: List[str],
        ordinal_features: List[str],
        ordinal_categories: Dict[str, List[str]],
        missing_value: str = 'отсутствует',
        scale_ordinal_features: bool = False,
        exclude_features: Optional[List[str]] = None,
    ):
        super().__init__(exclude_features=exclude_features)
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features
        self.ordinal_categories = ordinal_categories
        self.missing_value = missing_value
        self.scale_ordinal_features = scale_ordinal_features


    def fit(self, X, y=None) -> 'ScaledOneHotPreprocessor':
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            self.input_columns_ = list(X.columns)
            df = X
        else:
            if hasattr(self, 'feature_names_in_'):
                del self.feature_names_in_
            self.input_columns_ = [f'f_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self.input_columns_)

        self.n_features_in_ = df.shape[1]

        active_cat = [f for f in self.categorical_features if f in df.columns]
        active_ord = [f for f in self.ordinal_features if f in df.columns]
        numeric = [c for c in df.columns if c not in active_cat + active_ord]

        transformers = []

        if active_cat:
            transformers.append((
                'cat',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                active_cat,
            ))

        if active_ord:
            ord_steps = []
            if self.scale_ordinal_features:
                ord_steps.append(('imputer', SimpleImputer(
                    strategy='constant', fill_value=-1,
                )))
                ord_steps.append(('scaler', StandardScaler()))

            ord_encoder = OrdinalEncoder(
                categories=self._categories_for(active_ord),
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )

            if ord_steps:
                ord_pipe = SkPipeline([('encode', ord_encoder)] + ord_steps)
            else:
                ord_pipe = ord_encoder

            transformers.append(('ord', ord_pipe, active_ord))

        if numeric:
            numeric_pipe = SkPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ])
            transformers.append(('num', numeric_pipe, numeric))

        self.transformer_ = ColumnTransformer(transformers=transformers)

        df_filled = self._fill_missing(df)
        self.transformer_.fit(df_filled, y)

        self.feature_names_out_ = list(self.transformer_.get_feature_names_out())
        return self


    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, 'transformer_')
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.input_columns_)
        return self.transformer_.transform(self._fill_missing(df))


    def get_feature_names_in(self) -> np.ndarray:
        return super().get_feature_names_in()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, 'transformer_')
        return self.transformer_.get_feature_names_out(input_features)


    def _categories_for(self, active_ord: List[str]):
        cats = [self.ordinal_categories.get(feat) for feat in active_ord]
        if all(c is not None for c in cats):
            return cats
        return 'auto'

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        for col in df_filled.columns:
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                continue
            df_filled[col] = df_filled[col].fillna(self.missing_value)
            df_filled[col] = df_filled[col].replace(
                ['', None, np.nan, 'NaN', 'nan'],
                self.missing_value,
            )
        return df_filled