import warnings
from typing import List, Optional, Dict
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from .base import BasePreprocessor


class OneHotPreprocessor(BasePreprocessor):
    def __init__(
        self,
        categorical_features: List[str],
        ordinal_features: List[str],
        ordinal_categories: Dict[str, List[str]],
        missing_value: str = 'отсутствует',
        exclude_features: Optional[List[str]] = None,
    ):
        super().__init__(exclude_features=exclude_features)
        self.categorical_features = categorical_features
        self.ordinal_features = ordinal_features
        self.ordinal_categories = ordinal_categories
        self.missing_value = missing_value

    @staticmethod
    def _to_string(X):
        if hasattr(X, 'astype'):
            return X.astype(str)
        return np.asarray(X).astype(str)

    def _resolve_columns(self, all_features: List[str]):
        """Оставляет только те признаки, что реально есть в X и не исключены."""
        present = set(all_features)
        exclude = set(self.exclude_features or [])

        cat_cols = [
            c for c in self.categorical_features
            if c in present and c not in exclude
        ]
        ord_cols = [
            c for c in self.ordinal_features
            if c in present and c not in exclude
        ]

        # Предупреждаем о признаках, объявленных в конфиге, но отсутствующих в X
        missing = [
            c for c in (self.categorical_features + self.ordinal_features)
            if c not in present and c not in exclude
        ]
        if missing:
            warnings.warn(
                f"{type(self).__name__}: признаки отсутствуют во входе X и "
                f"будут пропущены: {missing}",
                stacklevel=2,
            )

        return cat_cols, ord_cols

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        elif hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_

        X_str = self._to_string(X)

        if hasattr(X_str, 'columns'):
            all_features = list(X_str.columns)
        else:
            all_features = [f'f_{i}' for i in range(X_str.shape[1])]

        cat_cols, ord_cols = self._resolve_columns(all_features)

        transformers = []
        if cat_cols:
            transformers.append((
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                cat_cols,          # <- имена, а не индексы
            ))
        if ord_cols:
            ord_pipe = OrdinalEncoder(
                categories=[
                    [str(c) for c in self.ordinal_categories[c]]
                    for c in ord_cols
                ],
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )
            transformers.append(('ord', ord_pipe, ord_cols))

        if not transformers:
            raise ValueError(
                f"{type(self).__name__}: после фильтрации (present ∩ "
                f"(categorical ∪ ordinal)) \\ exclude_features не осталось "
                f"ни одного признака для обработки."
            )

        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False,
        )
        self.transformer_.fit(X_str, y)

        self.n_features_in_ = self.transformer_.n_features_in_
        self.feature_names_out_ = list(self.transformer_.get_feature_names_out())

        # Полезно для отладки / для того, чтобы узнать, что реально попало в модель
        self.used_features_ = cat_cols + ord_cols
        return self

    def transform(self, X):
        check_is_fitted(self, 'transformer_')
        return self.transformer_.transform(self._to_string(X))


    def get_feature_names_in(self) -> np.ndarray:
        return super().get_feature_names_in()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, 'transformer_')
        return self.transformer_.get_feature_names_out(input_features)