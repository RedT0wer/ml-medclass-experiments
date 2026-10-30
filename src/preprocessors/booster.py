from typing import List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from .base import BasePreprocessor


class BoosterPreprocessor(BasePreprocessor):
    """
    Препроцессор для градиентных бустингов (CatBoost, XGBoost, LightGBM).
    Превращает все строковые значения в целые числа через OrdinalEncoder.
    """

    def __init__(
        self,
        missing_value: str = 'отсутствует',
        exclude_features: Optional[List[str]] = None,
    ):
        super().__init__(exclude_features=exclude_features)
        self.missing_value = missing_value

    def fit(self, X: Any, y=None) -> 'BoosterPreprocessor':
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            self.input_columns_ = list(X.columns)
        else:
            if hasattr(self, 'feature_names_in_'):
                del self.feature_names_in_
            self.input_columns_ = [f'f_{i}' for i in range(X.shape[1])]

        df = self._to_dataframe(X)
        self.n_features_in_ = df.shape[1]

        df_filled = self._fill_missing(df)

        self.encoder_ = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )
        self.encoder_.fit(df_filled)

        self.feature_names_out_ = list(df.columns)
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, 'encoder_')
        df = self._to_dataframe(X)
        return self.encoder_.transform(self._fill_missing(df)).astype(int)


    def get_feature_names_in(self) -> np.ndarray:
        return super().get_feature_names_in()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        # Для бустингов имена выхода совпадают с именами входа —
        # базовая реализация как раз это и делает.
        return super().get_feature_names_out(input_features)


    def _to_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            cols = (
                self.input_columns_
                if hasattr(self, 'input_columns_')
                else [f'f_{i}' for i in range(X.shape[1])]
            )
            return pd.DataFrame(X, columns=cols)
        return pd.DataFrame(X)

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        for col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(self.missing_value)
            df_filled[col] = df_filled[col].replace(
                ['', None, np.nan, 'NaN', 'nan'],
                self.missing_value,
            )
        return df_filled