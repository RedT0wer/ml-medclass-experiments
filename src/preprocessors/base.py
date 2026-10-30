from abc import abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BasePreprocessor(TransformerMixin, BaseEstimator):
    """
    Базовый класс для всех предобработчиков данных.

    Реализует sklearn-совместимый интерфейс трансформера:
      * параметры конструктора хранятся «как есть» (для get_params / clone);
      * fit(X, y=None) возвращает self и выставляет n_features_in_,
        feature_names_in_ (если X — DataFrame), feature_names_out_;
      * transform(X) возвращает np.ndarray;
      * реализованы оба «имя-метода»:
            - get_feature_names_in()  -> np.ndarray[str]
            - get_feature_names_out(input_features=None) -> np.ndarray[str]
    """

    def __init__(self, exclude_features: Optional[List[str]] = None):
        # ВАЖНО: сохраняем ровно то, что пришло в аргументе —
        # иначе BaseEstimator.get_params()/set_params()/clone() сломаются.
        self.exclude_features = exclude_features

    @abstractmethod
    def fit(self, X, y=None):
        """Обучает предобработчик на данных и возвращает self."""

    @abstractmethod
    def transform(self, X):
        """Применяет трансформацию к данным."""

    # get_feature_names_*

    def get_feature_names_in(self) -> np.ndarray:
        """
        Возвращает имена входных признаков (то, что было на входе fit).

        - Если модель обучалась на pandas.DataFrame, вернёт np.ndarray[str]
          с реальными именами колонок.
        - Если на numpy.ndarray — имена неизвестны; бросает AttributeError
          (как и обращение к sklearn-атрибуту feature_names_in_).
        - Если модель не обучена — бросает NotFittedError.
        """
        check_is_fitted(self)
        if not hasattr(self, 'feature_names_in_') or self.feature_names_in_ is None:
            raise AttributeError(
                f"{type(self).__name__}: входные имена признаков неизвестны "
                "(модель обучалась на массиве без имён колонок)."
            )
        return np.asarray(self.feature_names_in_, dtype=object)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Возвращает имена выходных признаков.

        input_features игнорируется в базовой реализации; подклассы,
        которые делегируют этот метод нижележащему sklearn-трансформеру,
        пробрасывают его туда (нужно для set_output(transform='pandas')).
        """
        check_is_fitted(self)
        if not hasattr(self, 'feature_names_out_') or self.feature_names_out_ is None:
            raise AttributeError(
                f"{type(self).__name__}: выходные имена признаков не заданы."
            )
        return np.asarray(self.feature_names_out_, dtype=object)