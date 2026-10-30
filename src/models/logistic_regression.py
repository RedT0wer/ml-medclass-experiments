from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    """
    Логистическая регрессия для мультиклассовой классификации.

    Поддерживает:
    - L1 (Lasso) и L2 (Ridge) регуляризацию
    - Автоматическое обнуление весов при L1
    - Балансировку классов
    - Различные solvers (lbfgs, saga, newton-cg)
    """

    def __init__(
            self,
            C: float = 1.0,
            solver: str = 'lbfgs',
            max_iter: int = 1000,
            class_weight: Optional[str] = None,
            random_state: int = 42
    ):
        """
        Args:
            C: Inverse of regularization strength (меньше = сильнее регуляризация)
            solver: 'lbfgs', 'saga', 'newton-cg', 'liblinear', 'sag'
                   Для L1 нужен 'saga' или 'liblinear'
            max_iter: Максимальное число итераций
            class_weight: 'balanced' или None
            random_state: Seed для воспроизводимости
        """
        super().__init__()
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            l1_ratio=1.0
        )

        self.coefficients_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'LogisticRegressionModel':
        """
        Обучает модель.

        Args:
            X: Обучающие данные
            y: Целевая переменная
            eval_set: Игнорируется (для совместимости с бустингами)
        """
        self.model.fit(X, y)

        self.coefficients_ = self.model.coef_.copy()
        self.intercept_ = self.model.intercept_.copy()
        self.n_features_in_ = X.shape[1]

        self.feature_importance = np.mean(np.abs(self.coefficients_), axis=0)

        self._build_training_report()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает предсказанные классы."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности для каждого класса."""
        return self.model.predict_proba(X)

    def _build_training_report(self) -> None:
        """Строит полиморфный отчет об обучении."""

        # Подсчет обнуленных весов (для L1 регуляризации)
        zeroed_weights_count = 0
        if self.coefficients_ is not None:
            # Считаем признаки, у которых ВСЕ веса по классам близки к нулю
            threshold = 1e-6
            for feature_idx in range(self.coefficients_.shape[1]):
                if np.all(np.abs(self.coefficients_[:, feature_idx]) < threshold):
                    zeroed_weights_count += 1

        # Определяем, сошлась ли модель
        converged = True
        if hasattr(self.model, 'n_iter_'):
            # n_iter_ может быть массивом (по одному на класс в ovr)
            n_iters = self.model.n_iter_
            if isinstance(n_iters, np.ndarray):
                max_iter_used = np.max(n_iters)
            else:
                max_iter_used = n_iters
            converged = max_iter_used < self.max_iter
        else:
            max_iter_used = None

        self._training_report = {
            'type': 'linear_model',
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features_in': self.n_features_in_,

            'linear_model_analysis': {
                'coefficients_matrix_shape': list(self.coefficients_.shape) if self.coefficients_ is not None else None,
                'coefficients': self.coefficients_.tolist() if self.coefficients_ is not None else None,
                'intercepts': self.intercept_.tolist() if self.intercept_ is not None else None,
                'n_iter_to_converge': int(max_iter_used) if max_iter_used is not None else None,
                'converged': bool(converged),
                'regularization_C': self.C,
                'solver': self.solver,
                'zeroed_weights_count': zeroed_weights_count,
                'class_weight': self.class_weight,
            }
        }

    def get_model_params(self) -> Dict[str, Any]:
        """Возвращает параметры модели для логирования."""
        return {
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }