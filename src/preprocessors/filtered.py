"""Препроцессор-фильтр признаков для отбора колонок после базовой предобработки."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.preprocessors.base import BasePreprocessor

logger = logging.getLogger(__name__)


class FilteredPreprocessor(BaseEstimator, TransformerMixin):
    """
    Обертка над препроцессором, которая фильтрует выходные признаки по списку индексов или имен.
    """

    def __init__(
        self,
        base_preprocessor: BasePreprocessor,
        keep_indices: Optional[Sequence[int]] = None,
        keep_features: Optional[Sequence[str]] = None,
    ):
        self.base_preprocessor = base_preprocessor
        self.keep_indices = [int(i) for i in keep_indices] if keep_indices is not None else None
        self.keep_features = [str(f) for f in keep_features] if keep_features is not None else None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> FilteredPreprocessor:
        self.base_preprocessor_ = self.base_preprocessor
        self.base_preprocessor_.fit(X, y)

        self.feature_names_in_ = getattr(self.base_preprocessor_, "feature_names_in_", None)
        self.n_features_in_ = getattr(self.base_preprocessor_, "n_features_in_", None)

        base_out = getattr(self.base_preprocessor_, "feature_names_out_", None)

        if self.keep_features is not None and base_out is not None:
            name_to_idx = {name: idx for idx, name in enumerate(base_out)}
            self.selected_indices_ = [name_to_idx[name] for name in self.keep_features if name in name_to_idx]
            self.feature_names_out_ = [base_out[i] for i in self.selected_indices_]
        elif self.keep_indices is not None:
            self.selected_indices_ = self.keep_indices
            if base_out is not None:
                self.feature_names_out_ = [base_out[i] for i in self.selected_indices_ if i < len(base_out)]
            else:
                self.feature_names_out_ = [f"f_sel_{i}" for i in self.selected_indices_]
        else:
            self.selected_indices_ = None
            self.feature_names_out_ = base_out

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, ["base_preprocessor_"])
        trans = self.base_preprocessor_.transform(X)

        if self.selected_indices_ is not None:
            trans = trans[:, self.selected_indices_]

        return np.asarray(trans, dtype=np.float32)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
        **fit_params: Any,
    ) -> np.ndarray:
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names_in(self) -> np.ndarray:
        check_is_fitted(self)
        if not hasattr(self, "feature_names_in_") or self.feature_names_in_ is None:
            raise AttributeError("feature_names_in_ не определены.")
        return np.asarray(self.feature_names_in_, dtype=object)

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        check_is_fitted(self)
        if not hasattr(self, "feature_names_out_") or self.feature_names_out_ is None:
            raise AttributeError("feature_names_out_ не определены.")
        return np.asarray(self.feature_names_out_, dtype=object)

