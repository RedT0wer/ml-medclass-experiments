# models/svm.py
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.svm import SVC

from .base import BaseClassifier


class SVMModel(BaseClassifier):
    """
    Метод опорных векторов (SVM) для мультиклассовой классификации.

    ВАЖНО: модель НЕ масштабирует признаки самостоятельно.
    Ожидается, что X приходит уже обработанным препроцессором
    (например, ScaledOneHotPreprocessor для SVM/LogReg/SGD).
    Это даёт единый источник правды для масштабирования и
    корректный экспорт в ONNX (скейлер живёт в препроцессоре,
    а не теряется внутри модели).

    Особенности:
    - Поддерживает ядра: 'linear', 'rbf', 'poly', 'sigmoid'
    - Для вероятностей используется Platt scaling (probability=True)
    - Для несбалансированных классов — class_weight='balanced'

    Ключевые индикаторы качества:
    - support_vectors_ratio: чем ближе к 1.0, тем сильнее переобучение
    - n_support_vectors: абсолютное число опорных векторов
    """

    def __init__(
            self,
            C: float = 1.0,
            kernel: str = 'rbf',
            degree: int = 3,
            gamma: str = 'scale',
            coef0: float = 0.0,
            tol: float = 1e-3,
            max_iter: int = -1,
            class_weight: Optional[str] = 'balanced',
            decision_function_shape: str = 'ovr',
            probability: bool = True,
            random_state: int = 42,
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.probability = probability
        self.random_state = random_state

        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            class_weight=class_weight,
            decision_function_shape=decision_function_shape,
            probability=probability,
            random_state=random_state,
        )

        self._n_features: Optional[int] = None
        self._n_support_vectors: Optional[np.ndarray] = None
        self._support_vectors_ratio: Optional[float] = None


    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> 'SVMModel':
        """
        Обучает SVM на уже подготовленных признаках.

        Args:
            X: признаки, УЖЕ обработанные препроцессором
               (масштабирование — ответственность препроцессора)
            y: целевая переменная
            eval_set: игнорируется (SVM не поддерживает early stopping)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.model.fit(X, y)

        self._n_features = X.shape[1]
        self._n_support_vectors = self.model.n_support_

        n_samples = X.shape[0]
        total_sv = int(np.sum(self._n_support_vectors))
        self._support_vectors_ratio = total_sv / n_samples if n_samples > 0 else 0.0

        self._build_training_report()
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности через Platt scaling.
        Работает только если probability=True при инициализации.
        """
        if not self.probability:
            raise ValueError(
                "predict_proba доступен только при probability=True. "
                "Используйте decision_function вместо этого."
            )
        return self.model.predict_proba(np.asarray(X))


    def _build_training_report(self) -> None:
        report: Dict[str, Any] = {
            'type': 'svm',
            'feature_importance': None,
            'n_features_in': self._n_features,
            'svm_analysis': {
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'degree': self.degree if self.kernel == 'poly' else None,
                'coef0': self.coef0,
                'class_weight': self.class_weight,
                'decision_function_shape': self.decision_function_shape,
                'probability': self.probability,
                'n_classes': len(self.model.classes_),
                'classes': self.model.classes_.tolist(),
                'n_support_vectors': self._n_support_vectors.tolist(),
                'total_support_vectors': int(np.sum(self._n_support_vectors)),
                'support_vectors_ratio': float(self._support_vectors_ratio),
                'n_samples': int(
                    np.sum(self._n_support_vectors) / self._support_vectors_ratio
                ) if self._support_vectors_ratio and self._support_vectors_ratio > 0
                else None,
                'overfitting_indicator':
                    self._interpret_sv_ratio(self._support_vectors_ratio),
                '_gamma_actual': float(self.model._gamma)
                    if hasattr(self.model, '_gamma') else None,
                'converged': True,
                'max_iter': self.max_iter,
            },
        }

        # Для линейного ядра — коэффициенты
        if self.kernel == 'linear':
            coef = self.model.coef_
            intercept = self.model.intercept_

            report['svm_analysis']['coefficients'] = coef.tolist()
            report['svm_analysis']['intercepts'] = intercept.tolist()
            report['svm_analysis']['coefficients_shape'] = list(coef.shape)

            if coef.shape[0] == 1:
                importance = np.abs(coef[0])
            else:
                importance = np.mean(np.abs(coef), axis=0)

            report['feature_importance'] = importance.tolist()

        if hasattr(self.model, 'n_iter_'):
            n_iter = self.model.n_iter_
            report['svm_analysis']['n_iter'] = (
                n_iter.tolist() if hasattr(n_iter, '__iter__') else int(n_iter)
            )

        self._training_report = report

    @staticmethod
    def _interpret_sv_ratio(ratio: float) -> str:
        """
        Эвристика:
        - < 0.2: excellent_generalization
        - 0.2..0.5: normal
        - 0.5..0.8: possible_overfitting
        - >= 0.8: strong_overfitting
        """
        if ratio < 0.2:
            return 'excellent_generalization'
        elif ratio < 0.5:
            return 'normal'
        elif ratio < 0.8:
            return 'possible_overfitting'
        else:
            return 'strong_overfitting'


    def get_model_params(self) -> Dict[str, Any]:
        return {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'decision_function_shape': self.decision_function_shape,
            'probability': self.probability,
            'random_state': self.random_state,
        }