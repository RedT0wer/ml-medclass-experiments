import time
import warnings
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import (
    cross_validate, train_test_split, learning_curve, cross_val_predict,
)
from sklearn.metrics import (
    precision_score, recall_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, make_scorer,
    accuracy_score, balanced_accuracy_score, f1_score,
)
from sklearn.preprocessing import label_binarize

from .pipeline import Pipeline
from .validators.base import BaseValidator


# SCORERS

def _auroc_macro(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: Optional[List] = None,
) -> float:
    labels = (
        list(class_labels) if class_labels is not None
        else list(np.unique(y_true))
    )
    if len(labels) == 2:
        return float(roc_auc_score(y_true, y_score[:, 1]))
    return float(roc_auc_score(
        y_true, y_score, multi_class='ovr', average='macro', labels=labels,
    ))


def _auprc_macro(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: Optional[List] = None,
) -> float:
    labels = (
        list(class_labels) if class_labels is not None
        else list(np.unique(y_true))
    )
    if len(labels) == 2:
        return float(average_precision_score(y_true, y_score[:, 1]))
    y_bin = label_binarize(y_true, classes=labels)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return float(average_precision_score(y_bin, y_score, average='macro'))


# EVALUATOR

class ModelEvaluator:
    """
    Оценивает Pipeline по схеме holdout + CV.

    Схема evaluate():

      1. train_test_split(стратифицированный, stratify=y) → X_train, X_test.
      2. cross_validate на X_train с self.validator → метрики CV (mean/std
         по фолдам).
      3. Финальный fit на X_train.
      4. Holdout: финальная модель оценивается на X_test (блок result['holdout']).
      5. (опционально, compute_detailed=True):
         - OOF на X_train с self.validator — тот же CV-разрез, что и п.2;
         - ROC/PR и confusion matrix по OOF;
         - learning curves на X_train.

    Согласованность:
      - CV-метрики и OOF-детали считаются на одной подвыборке (X_train)
        и с одной CV-схемой (self.validator).
      - Holdout считается на отложенной X_test, которую модель не видела.
      - Везде используется единый порядок классов (class_labels),
        вероятности выравниваются через _align_proba.

    Примечание: cross_val_predict требует, чтобы каждый объект попадал
    в тестовую часть ровно один раз. Для Repeated*-валидаторов OOF-блок
    будет пропущен (learning curves считаются всегда).
    """

    METRICS = [
        'accuracy',
        'balanced_accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'f1_weighted',
        'roc_auc_macro',
        'auprc_macro',
        'mcc',
        'cohen_kappa',
    ]

    def __init__(
        self,
        pipeline_factory: Callable[[], Pipeline],
        validator: BaseValidator,
    ):
        self.pipeline_factory = pipeline_factory
        self.validator = validator


    def evaluate(
        self,
        X: Any,
        y: Any,
        class_labels: Optional[List[int]] = None,
        compute_detailed: bool = True,
    ) -> Dict[str, Any]:
        y = np.asarray(y)
        if class_labels is None:
            class_labels = sorted(list(np.unique(y)))
        class_labels = list(class_labels)

        split_info = self.validator.get_split_info()
        test_size = float(split_info.get('test_size', 0.2))
        random_state = int(split_info.get('random_state', 42))

        _, class_counts = np.unique(y, return_counts=True)
        min_class = int(class_counts.min())

        requested_splits = int(self.validator.get_n_splits())
        if requested_splits > min_class:
            warnings.warn(
                f"n_splits={requested_splits} превышает минимальный размер "
                f"класса ({min_class}); используется n_splits={min_class}."
            )
        n_splits = max(2, min(requested_splits, min_class))

        indices = np.arange(len(y))
        (
            X_train, X_test, y_train, y_test, idx_train, idx_test,
        ) = train_test_split(
            X, y, indices,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        scoring = self._build_scoring()
        t_start = time.time()
        cv_results = cross_validate(
            self.pipeline_factory(),
            X_train,
            y_train,
            cv=self.validator,
            scoring=scoring,
            return_train_score=False,
            error_score='raise',
        )
        total_seconds = time.time() - t_start

        metrics: Dict[str, Dict[str, Any]] = {}
        for name in self.METRICS:
            values = list(np.asarray(cv_results[f'test_{name}']).astype(float))
            metrics[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }

        total_folds = int(np.asarray(cv_results['fit_time']).size)
        cv_timing = {
            'fit_seconds': float(np.sum(cv_results['fit_time'])),
            'score_seconds': float(np.sum(cv_results['score_time'])),
            'total_seconds': float(total_seconds),
            'n_splits_per_repeat': n_splits,
            'total_folds': total_folds,
        }

        t_fit = time.time()
        final_pipeline = self.pipeline_factory().fit(X_train, y_train)
        fit_full_seconds = time.time() - t_fit
        training_data = final_pipeline.get_training_history()

        t_pred = time.time()
        y_pred_test = final_pipeline.predict(X_test)
        y_proba_test_raw = final_pipeline.predict_proba(X_test)
        y_proba_test = self._align_proba(
            y_proba_test_raw, final_pipeline.classes_, class_labels,
        )
        predict_test_seconds = time.time() - t_pred

        holdout = self._build_holdout_block(
            y_true=y_test,
            y_pred=y_pred_test,
            y_proba=y_proba_test,
            class_labels=class_labels,
        )

        try:
            prep_params = dict(
                final_pipeline.preprocessor.get_params(deep=False)
            )
        except Exception:
            prep_params = {}

        result: Dict[str, Any] = {
            'config': {
                'preprocessor_params': prep_params,
                'validator_info': split_info,
                'feature_names_in': getattr(
                    final_pipeline.preprocessor_,
                    'feature_names_in_',
                    None,
                ),
                'feature_names_out': getattr(
                    final_pipeline.preprocessor_,
                    'feature_names_out_',
                    None,
                ),
                'class_labels': class_labels,
                'split': {
                    'test_size': test_size,
                    'random_state': random_state,
                    'n_train': int(len(y_train)),
                    'n_test': int(len(y_test)),
                    'class_distribution_train': self._class_distribution(y_train),
                    'class_distribution_test': self._class_distribution(y_test),
                },
                'cv_scheme': {
                    'validator_type': type(self.validator).__name__,
                    'requested_n_splits': requested_splits,
                    'n_splits_per_repeat': n_splits,
                    'total_folds': total_folds,
                    'random_state': random_state,
                    'note': (
                        'CV-метрики: cross_validate на X_train с self.validator. '
                        'OOF/детали: cross_val_predict на X_train с тем же '
                        'self.validator (если он совместим с cross_val_predict; '
                        'для Repeated* OOF пропускается). '
                        'Holdout: финальная модель на X_train, оценка на X_test.'
                    ),
                },
            },
            'metrics': metrics,   # CV на X_train
            'holdout': holdout,   # оценка финальной модели на X_test
            'training_data': training_data,
            'timing': {
                **cv_timing,
                'fit_full_seconds': float(fit_full_seconds),
                'predict_test_seconds': float(predict_test_seconds),
            },
            # Финальная обученная модель — доступна для export_onnx и т.п.
            'pipeline': final_pipeline,
        }

        if compute_detailed:
            result['detailed'] = self._compute_detailed(
                X=X_train,
                y=y_train,
                class_labels=class_labels,
            )

        return result


    @staticmethod
    def _class_distribution(y: np.ndarray) -> Dict[str, int]:
        values, counts = np.unique(np.asarray(y), return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts)}


    def _build_scoring(self) -> Dict[str, Any]:
        return {
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'precision_macro': make_scorer(
                precision_score, average='macro', zero_division=0
            ),
            'recall_macro': make_scorer(
                recall_score, average='macro', zero_division=0
            ),
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
            'roc_auc_macro': make_scorer(
                _auroc_macro, response_method='predict_proba'
            ),
            'auprc_macro': make_scorer(
                _auprc_macro, response_method='predict_proba'
            ),
            'mcc': make_scorer(matthews_corrcoef),
            'cohen_kappa': make_scorer(cohen_kappa_score),
        }


    def _build_holdout_block(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_labels: List[int],
    ) -> Dict[str, Any]:
        """
        Метрики и артефакты на отложенной X_test.

        ВАЖНО: holdout-метрики — это точечные значения на одной выборке
        (без mean/std), в отличие от CV. Они не обязаны совпадать с
        metrics[*]['mean'] (это среднее по фолдам CV на X_train),
        но их можно сравнивать как «ожидаемо близко».

        Сырые предсказания (y_true / y_pred / y_proba) больше не сохраняются
        в JSON — они полностью восстанавливаются из сохранённого Pipeline.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_proba = np.asarray(y_proba)

        metric_values = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(
                y_true, y_pred, average='macro', zero_division=0,
            )),
            'recall_macro': float(recall_score(
                y_true, y_pred, average='macro', zero_division=0,
            )),
            'f1_macro': float(f1_score(
                y_true, y_pred, average='macro', zero_division=0,
            )),
            'f1_weighted': float(f1_score(
                y_true, y_pred, average='weighted', zero_division=0,
            )),
            'roc_auc_macro': float(_auroc_macro(y_true, y_proba, class_labels)),
            'auprc_macro': float(_auprc_macro(y_true, y_proba, class_labels)),
            'mcc': float(matthews_corrcoef(y_true, y_pred)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        }

        return {
            'metrics': metric_values,
            'confusion_matrix': confusion_matrix(
                y_true, y_pred, labels=class_labels,
            ).tolist(),
            'roc_curves': self._roc(y_true, y_proba, class_labels),
            'pr_curves': self._pr(y_true, y_proba, class_labels),
        }


    def _compute_detailed(
        self,
        X: Any,
        y: np.ndarray,
        class_labels: List[int],
    ) -> Dict[str, Any]:
        """
        OOF-предсказания и кривые на X_train с использованием self.validator.

        Используется тот же валидатор, что и в cross_validate — это
        гарантирует согласованность CV-метрик и OOF-артефактов.

        Ограничение sklearn: cross_val_predict требует, чтобы каждый объект
        попадал в тестовую часть ровно один раз. Для Repeated* валидаторов
        OOF-блок пропускается (learning curves считаются всегда).
        """
        y = np.asarray(y)
        cv = self.validator

        detailed: Dict[str, Any] = {
            'validator': type(cv).__name__,
        }

        try:
            y_pred_oof = cross_val_predict(
                self.pipeline_factory(), X, y, cv=cv,
                method='predict', n_jobs=1,
            )
            y_proba_raw = cross_val_predict(
                self.pipeline_factory(), X, y, cv=cv,
                method='predict_proba', n_jobs=1,
            )
        except Exception as e:
            warnings.warn(
                f"cross_val_predict несовместим с валидатором "
                f"{type(cv).__name__}: {e}. "
                f"OOF / ROC / PR / confusion_matrix пропущены."
            )
            detailed['oof_error'] = str(e)
        else:
            # classes_ эквивалентны np.unique(y): Pipeline.fit выставляет
            # self.classes_ = np.unique(y), поэтому не нужно делать лишний fit.
            classes_ = np.unique(y)
            y_proba = self._align_proba(y_proba_raw, classes_, class_labels)

            detailed['confusion_matrix'] = confusion_matrix(
                y, y_pred_oof, labels=class_labels,
            ).tolist()
            detailed['roc_curves'] = self._roc(y, y_proba, class_labels)
            detailed['pr_curves'] = self._pr(y, y_proba, class_labels)

        detailed['learning_curves'] = self._compute_learning_curves(
            X_train=X,
            y_train=y,
            cv=cv,
            scoring='f1_macro',
        )

        return detailed


    @staticmethod
    def _align_proba(
        y_proba: np.ndarray,
        classes_: np.ndarray,
        class_labels: List[int],
    ) -> np.ndarray:
        """Приводит столбцы predict_proba к порядку class_labels."""
        y_proba = np.asarray(y_proba)
        classes_ = np.asarray(classes_)
        n = len(class_labels)
        out = np.zeros((y_proba.shape[0], n), dtype=float)
        for j, c in enumerate(class_labels):
            idx = np.where(classes_ == c)[0]
            if len(idx) == 1:
                out[:, j] = y_proba[:, idx[0]]
        return out

    @staticmethod
    def _roc(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_labels: List[int],
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        class_labels = list(class_labels)

        if len(class_labels) == 2:
            pos_label = class_labels[1]
            y_bin = (y_true == pos_label).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
            return {
                'binary': {
                    'pos_label': int(pos_label),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(auc(fpr, tpr)),
                }
            }

        y_bin = label_binarize(y_true, classes=class_labels)
        per_class: Dict[str, Any] = {}
        auc_values: List[float] = []
        for i, label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = float(auc(fpr, tpr))
            per_class[str(label)] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc,
            }
            auc_values.append(roc_auc)

        return {
            'per_class': per_class,
            'macro_auc': float(np.mean(auc_values)),
        }

    @staticmethod
    def _pr(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_labels: List[int],
    ) -> Dict[str, Any]:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        class_labels = list(class_labels)

        if len(class_labels) == 2:
            pos_label = class_labels[1]
            y_bin = (y_true == pos_label).astype(int)
            precision, recall, _ = precision_recall_curve(
                y_bin, y_proba[:, 1]
            )
            ap = float(average_precision_score(y_bin, y_proba[:, 1]))
            return {
                'binary': {
                    'pos_label': int(pos_label),
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'average_precision': ap,
                }
            }

        y_bin = label_binarize(y_true, classes=class_labels)
        per_class: Dict[str, Any] = {}
        ap_values: List[float] = []
        for i, label in enumerate(class_labels):
            precision, recall, _ = precision_recall_curve(
                y_bin[:, i], y_proba[:, i]
            )
            ap = float(average_precision_score(y_bin[:, i], y_proba[:, i]))
            per_class[str(label)] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'average_precision': ap,
            }
            ap_values.append(ap)

        return {
            'per_class': per_class,
            'macro_ap': float(np.mean(ap_values)),
        }


    def _compute_learning_curves(
        self,
        X_train: Any,
        y_train: np.ndarray,
        cv: Any,
        scoring: str = 'f1_macro',
        n_points: int = 8,
    ) -> Dict[str, Any]:
        n_samples = len(y_train)

        n_splits = (
            cv.get_n_splits()
            if hasattr(cv, 'get_n_splits')
            else getattr(cv, 'n_splits', 5)
        )
        min_size = max(n_splits, 5)
        if n_samples < min_size * 2:
            return {'error': 'too few samples for learning curves'}

        train_sizes = np.linspace(
            max(min_size / n_samples, 0.2), 1.0, n_points
        )

        try:
            sizes, train_scores, val_scores = learning_curve(
                self.pipeline_factory(),
                X_train,
                y_train,
                train_sizes=train_sizes,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score='raise',
            )
        except Exception as e:
            return {'error': str(e)}

        return {
            'scoring': scoring,
            'train_sizes': sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist(),
        }