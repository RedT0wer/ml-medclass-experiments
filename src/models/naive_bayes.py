from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler

from .base import BaseClassifier


class ComplementNBModel(BaseClassifier):
    """
    Complement Naive Bayes — специально для несбалансированных данных.

    Работает лучше MultinomialNB, когда классы несбалансированы.
    Требует неотрицательных признаков (использует MinMaxScaler внутри).

    Наследует ScalerPipelineExportMixin, чтобы ONNX/TFLite-экспорт
    корректно включал MinMaxScaler в граф.
    """

    def __init__(
            self,
            alpha: float = 1.0,
            fit_prior: bool = True,
            norm: bool = False
    ):
        super().__init__()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.norm = norm

        self.model = ComplementNB(
            alpha=alpha,
            fit_prior=fit_prior,
            norm=norm,
        )

        # Скейлер для гарантии неотрицательности
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self._n_features: Optional[int] = None
        self.feature_importance: Optional[np.ndarray] = None


    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> 'ComplementNBModel':
        """
        Обучает модель. eval_set игнорируется.
        Данные масштабируются в [0, 1] для гарантии неотрицательности.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._n_features = X_scaled.shape[1]

        # Важность признаков = |feature_log_prob_|, усреднённая по классам
        self.feature_importance = np.mean(
            np.abs(self.model.feature_log_prob_), axis=0
        )

        self._build_training_report()
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(np.asarray(X))
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(np.asarray(X))
        return self.model.predict_proba(X_scaled)


    def _build_training_report(self) -> None:
        class_prior = getattr(self.model, 'class_prior_', None)

        self._training_report = {
            'type': 'complement_nb',
            'feature_importance': (
                self.feature_importance.tolist()
                if self.feature_importance is not None else None
            ),
            'n_features_in': self._n_features,
            'complement_nb_analysis': {
                'alpha': self.alpha,
                'fit_prior': self.fit_prior,
                'norm': self.norm,
                'n_classes': len(self.model.classes_),
                'classes': self.model.classes_.tolist(),
                'class_prior': (
                    class_prior.tolist()
                    if class_prior is not None else []
                ),
                'feature_log_prob_shape': list(self.model.feature_log_prob_.shape),
                'feature_importance_mean': (
                    float(np.mean(self.feature_importance))
                    if self.feature_importance is not None else None
                ),
                'feature_importance_std': (
                    float(np.std(self.feature_importance))
                    if self.feature_importance is not None else None
                ),
            }
        }


    def get_model_params(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'fit_prior': self.fit_prior,
            'norm': self.norm,
        }