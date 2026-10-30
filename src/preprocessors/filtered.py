from typing import List, Optional

import numpy as np


class FilteredPreprocessor:
    """
    Проксирует fit_transform/transform к base-препроцессору,
    затем срезает колонки по keep_indices.

    Все атрибуты base (feature_names_out_, input_columns_, classes_)
    доступны через обёртку.
    """

    def __init__(self, base_preprocessor, keep_indices: Optional[List[int]] = None):
        self.base = base_preprocessor
        self.keep_indices = list(keep_indices) if keep_indices is not None else None

        # Синхронизируем публичные атрибуты
        self._sync_attrs()


    def fit(self, X, y=None):
        if hasattr(self.base, "fit"):
            try:
                self.base.fit(X, y)
            except TypeError:
                self.base.fit(X)
        self._sync_attrs()
        return self

    def fit_transform(self, X, y=None):
        if hasattr(self.base, "fit_transform"):
            try:
                Xp = self.base.fit_transform(X, y)
            except TypeError:
                Xp = self.base.fit_transform(X)
        else:
            self.fit(X, y)
            Xp = self.transform(X)
        self._sync_attrs()
        return self._select(Xp)

    def transform(self, X, y=None):
        if hasattr(self.base, "transform"):
            try:
                Xp = self.base.transform(X)
            except TypeError:
                Xp = self.base.transform(X, None)
        else:
            raise AttributeError("base_preprocessor has no transform()")
        return self._select(Xp)


    def _select(self, Xp):
        if self.keep_indices is None:
            return Xp

        # DataFrame
        if hasattr(Xp, "iloc"):
            return Xp.iloc[:, self.keep_indices]

        # scipy sparse
        if hasattr(Xp, "toarray"):
            return Xp[:, self.keep_indices]

        # numpy
        arr = np.asarray(Xp)
        return arr[:, self.keep_indices]

    def _sync_attrs(self):
        for attr in (
            "feature_names_out_",
            "feature_names_in_",
            "input_columns_",
            "classes_",
            "missing_value",
        ):
            if hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))