from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

from .base import BaseClassifier


class RidgeClassifierModel(BaseClassifier):
    """
    Ridge Classifier — Ridge-регрессия для классификации.

    Быстрый и стабильный, хорошо работает на маленьких выборках.
    НЕ имеет встроенного predict_proba — аппроксимируется через
    decision_function + softmax.

    Требует масштабирования признаков (StandardScaler внутри).
    """

    def __init__(
            self,
            alpha: float = 1.0,
            fit_intercept: bool = True,
            max_iter: Optional[int] = None,
            tol: float = 1e-3,
            class_weight: Optional[str] = None,
            random_state: Optional[int] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = RidgeClassifier(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state
        )

        # Скейлер для стабилизации обучения
        self.scaler = StandardScaler()

        self._n_features: Optional[int] = None
        self._n_iter: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'RidgeClassifierModel':
        """
        Обучает модель. eval_set игнорируется.
        Данные масштабируются через StandardScaler.
        """
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self._n_features = X_scaled.shape[1]

        # Безопасная обработка n_iter_
        if hasattr(self.model, 'n_iter_') and self.model.n_iter_ is not None:
            n_iter = self.model.n_iter_
            if isinstance(n_iter, np.ndarray):
                self._n_iter = int(np.max(n_iter))
            elif isinstance(n_iter, (int, float)):
                self._n_iter = int(n_iter)
            else:
                self._n_iter = None
        else:
            self._n_iter = None

        # Важность признаков
        if self.model.coef_.ndim == 1:
            self.feature_importance = np.abs(self.model.coef_)
        else:
            self.feature_importance = np.mean(np.abs(self.model.coef_), axis=0)

        self._build_training_report()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Аппроксимация вероятностей через decision_function + softmax.

        RidgeClassifier не имеет встроенного predict_proba,
        но decision_function возвращает "уверенность" модели,
        которую можно преобразовать в вероятности через softmax.
        """
        X_scaled = self.scaler.transform(X)
        decision = self.model.decision_function(X_scaled)

        if decision.ndim == 1:
            # Бинарный случай: decision shape (n_samples,)
            # Преобразуем в (n_samples, 2)
            decision = np.vstack([-decision, decision]).T

        # Softmax для нормализации в вероятности
        proba = softmax(decision, axis=1)
        return proba

    def _build_training_report(self) -> None:
        # Получаем коэффициенты и интерсепты
        coef = self.model.coef_
        intercept = self.model.intercept_

        if coef.ndim == 1:
            coef_shape = (1, len(coef))
        else:
            coef_shape = coef.shape

        self._training_report = {
            'type': 'ridge',
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features_in': self._n_features,

            'ridge_analysis': {
                'alpha': self.alpha,
                'fit_intercept': self.fit_intercept,
                'class_weight': self.class_weight,
                'n_classes': len(self.model.classes_),
                'classes': self.model.classes_.tolist(),
                'coefficients_matrix_shape': list(coef_shape),
                'coefficients': coef.tolist() if coef.ndim == 1 else coef.tolist(),
                'intercepts': intercept.tolist() if isinstance(intercept, np.ndarray) else [float(intercept)],
                'n_iter': self._n_iter,
                'converged': self._n_iter is not None and (self.max_iter is None or self._n_iter < self.max_iter),
                'feature_importance_mean': float(
                    np.mean(self.feature_importance)) if self.feature_importance is not None else None,
                'feature_importance_std': float(
                    np.std(self.feature_importance)) if self.feature_importance is not None else None
            }
        }

    def get_model_params(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }


class SGDClassifierModel(BaseClassifier):
    """
    Stochastic Gradient Descent Classifier.

    Поддерживает различные функции потерь:
    - 'log_loss' (логистическая регрессия) — имеет predict_proba
    - 'modified_huber' — имеет predict_proba (аппроксимация)
    - 'hinge' (SVM) — НЕ имеет predict_proba (аппроксимация через decision_function)

    Требует масштабирования признаков (StandardScaler внутри).
    Хорошо работает на больших данных, но может быть нестабилен на маленьких.
    """

    def __init__(
            self,
            loss: str = 'log_loss',
            penalty: str = 'l2',
            alpha: float = 0.0001,
            l1_ratio: float = 0.15,
            max_iter: int = 1000,
            tol: float = 1e-3,
            class_weight: Optional[str] = None,
            random_state: Optional[int] = None,
            learning_rate: str = 'optimal',
            eta0: float = 0.01,
            early_stopping: bool = False,
            validation_fraction: float = 0.1,
            n_iter_no_change: int = 5
    ):
        super().__init__()
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change
        )

        # Скейлер для стабилизации обучения
        self.scaler = StandardScaler()

        self._n_features: Optional[int] = None
        self._n_iter: Optional[int] = None

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'SGDClassifierModel':
        """
        Обучает модель. eval_set игнорируется (SGD использует internal validation).
        Данные масштабируются через StandardScaler.
        """
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self._n_features = X_scaled.shape[1]

        # Количество итераций
        if hasattr(self.model, 'n_iter_'):
            n_iter = self.model.n_iter_
            if isinstance(n_iter, np.ndarray):
                self._n_iter = int(np.max(n_iter))
            else:
                self._n_iter = int(n_iter)
        else:
            self._n_iter = None

        # Важность признаков
        if self.model.coef_.ndim == 1:
            self.feature_importance = np.abs(self.model.coef_)
        else:
            self.feature_importance = np.mean(np.abs(self.model.coef_), axis=0)

        self._build_training_report()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает вероятности.

        Для loss='log_loss' и 'modified_huber' — встроенный predict_proba.
        Для loss='hinge' — аппроксимация через decision_function + softmax.
        """
        X_scaled = self.scaler.transform(X)

        if self.loss in ['log_loss', 'modified_huber']:
            return self.model.predict_proba(X_scaled)
        else:
            # Для hinge и других — аппроксимация
            decision = self.model.decision_function(X_scaled)

            if decision.ndim == 1:
                decision = np.vstack([-decision, decision]).T

            proba = softmax(decision, axis=1)
            return proba

    def _build_training_report(self) -> None:
        coef = self.model.coef_
        intercept = self.model.intercept_

        if coef.ndim == 1:
            coef_shape = (1, len(coef))
        else:
            coef_shape = coef.shape

        # Подсчёт обнулённых весов (для L1-регуляризации)
        zeroed_weights = 0
        if self.penalty in ['l1', 'elasticnet']:
            threshold = 1e-6
            if coef.ndim == 1:
                zeroed_weights = int(np.sum(np.abs(coef) < threshold))
            else:
                # Признак обнулён, если ВСЕ его веса по классам близки к нулю
                for feature_idx in range(coef.shape[1]):
                    if np.all(np.abs(coef[:, feature_idx]) < threshold):
                        zeroed_weights += 1

        self._training_report = {
            'type': 'sgd',
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'n_features_in': self._n_features,

            'sgd_analysis': {
                'loss': self.loss,
                'penalty': self.penalty,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio if self.penalty == 'elasticnet' else None,
                'learning_rate': self.learning_rate,
                'class_weight': self.class_weight,
                'early_stopping': self.early_stopping,
                'n_classes': len(self.model.classes_),
                'classes': self.model.classes_.tolist(),
                'coefficients_matrix_shape': list(coef_shape),
                'coefficients': coef.tolist() if coef.ndim == 1 else coef.tolist(),
                'intercepts': intercept.tolist() if isinstance(intercept, np.ndarray) else [float(intercept)],
                'n_iter': self._n_iter,
                'converged': self._n_iter is not None and self._n_iter < self.max_iter,
                'zeroed_weights_count': zeroed_weights,
                'feature_importance_mean': float(
                    np.mean(self.feature_importance)) if self.feature_importance is not None else None,
                'feature_importance_std': float(
                    np.std(self.feature_importance)) if self.feature_importance is not None else None
            }
        }

    def get_model_params(self) -> Dict[str, Any]:
        return {
            'loss': self.loss,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change
        }