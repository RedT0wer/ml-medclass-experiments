from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .base import BaseClassifier


class DecisionTreeModel(BaseClassifier):
    """
    Дерево решений для мультиклассовой классификации.

    Наследуется от BaseClassifier и создает полиморфный отчет обучения
    с блоком tree_analysis.
    """

    def __init__(
            self,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            criterion: str = 'gini',
            class_weight: Optional[str] = None,
            random_state: int = 42
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            class_weight=class_weight,
            random_state=random_state
        )

        # Дополнительные метрики для репорта
        self._tree_depth: Optional[int] = None
        self._n_leaves: Optional[int] = None
        self._n_features: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'DecisionTreeModel':
        """
        Обучает дерево решений.

        eval_set игнорируется (деревья не поддерживают early stopping).
        """
        self.model.fit(X, y)

        self._tree_depth = self.model.get_depth()
        self._n_leaves = self.model.get_n_leaves()
        self._n_features = X.shape[1]

        self.feature_importance = self.model.feature_importances_

        # Создаем полиморфный отчет обучения
        self._build_training_report()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает предсказанные классы."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности для каждого класса."""
        return self.model.predict_proba(X)

    def _build_training_report(self) -> None:
        """
        Создает полиморфный отчет обучения с обязательной базой
        и специфичным блоком tree_analysis.
        """
        self._training_report = {
            'type': 'decision_tree',
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features_in': self._n_features,

            'tree_analysis': {
                'max_depth_configured': self.max_depth,
                'actual_depth': self._tree_depth,
                'n_leaves': self._n_leaves,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'criterion': self.criterion,
                'class_weight': self.class_weight
            }
        }

    def get_model_params(self) -> Dict[str, Any]:
        """Возвращает параметры модели для логирования."""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'criterion': self.criterion,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }