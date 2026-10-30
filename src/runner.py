from typing import Any, Callable, Dict, List, Optional
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluator import ModelEvaluator
from .pipeline import Pipeline, _summarize_training
from .tuner import GridSearchTuner
from .validators.base import BaseValidator
from .preprocessors.base import BasePreprocessor
from .models.base import BaseClassifier


class ExperimentRunner:
    """
    Оркестратор экспериментов.

    Каждый эксперимент сохраняется в собственной папке:
        results/<name>/
            result.json          # CV + holdout метрики, config, timing
            best_detailed.json   # (только gridsearch) detailed-блок
            pipeline.joblib      # обученный Pipeline (лучший для gridsearch)

    JSON-отчёты не содержат полных массивов (per-fold values, точки ROC/PR,
    полная training_history) — только агрегаты. Полные данные доступны
    в памяти через self.results[name] и в pipeline.joblib / *.pkl.

    Пример:
        runner.run_gridsearch(
            name='RandomForest',
            preprocessor=onehot_prep,
            model=lambda **p: RandomForestModel(**p),
            validator=validator,
            param_grid={...},
            X=X, y=y,
        )
    """

    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict[str, Any]] = {}


    def _exp_dir(self, name: str) -> Path:
        """Папка эксперимента: results/<name>/ (создаётся при необходимости)."""
        d = self.results_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


    @staticmethod
    def _as_model_factory(model) -> Callable[..., BaseClassifier]:
        """
        Нормализует `model` в фабрику, вызываемую как factory(**kwargs).

        Поддерживаются:
          - callable-фабрика:  lambda **p: RandomForestModel(**p)
          - класс модели:      RandomForestModel
          - готовый инстанс:   RandomForestModel(n_estimators=100)
                               (kwargs запрещены — используется как есть)
        """
        if isinstance(model, type):
            return model
        if callable(model) and not hasattr(model, 'fit'):
            return model

        def _factory(**kw):
            if kw:
                raise ValueError(
                    f"Модель передана инстансом, переопределение "
                    f"параметров невозможно: {kw}"
                )
            return model
        return _factory

    @staticmethod
    def make_pipeline_factory(
        preprocessor: BasePreprocessor,
        model_factory: Callable[..., BaseClassifier],
    ) -> Callable[..., Pipeline]:
        """
        Возвращает фабрику Pipeline: Pipeline(prep, model_factory(**kwargs)).

        Единая точка для run_single / run_gridsearch. `model_factory` уже
        нормализован через _as_model_factory().
        """
        return lambda **kw: Pipeline(preprocessor, model_factory(**kw))


    def run_single(
        self,
        name: str,
        preprocessor: BasePreprocessor,
        model,
        validator: BaseValidator,
        X: Any,
        y: Any,
        class_labels: Optional[List[int]] = None,
        save: bool = True,
        save_pipeline: bool = True,
    ) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"Запуск эксперимента: {name}")
        print(f"{'=' * 60}")

        model_factory = self._as_model_factory(model)
        pipeline_factory = self.make_pipeline_factory(preprocessor, model_factory)

        evaluator = ModelEvaluator(pipeline_factory, validator)
        result = evaluator.evaluate(
            X, y,
            class_labels=class_labels,
            compute_detailed=True,
        )

        self.results[name] = result

        if save:
            self.save_detailed(name)
            if save_pipeline:
                self.save_pipeline(name)

        return result


    def run_gridsearch(
        self,
        name: str,
        preprocessor: BasePreprocessor,
        model,
        validator: BaseValidator,
        param_grid: Dict[str, List[Any]],
        X: Any,
        y: Any,
        class_labels: Optional[List[int]] = None,
        scoring_metric: str = 'f1_macro',
        save: bool = True,
        save_pipeline: bool = True,
    ) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"Запуск GridSearch: {name}")
        print(f"Параметров для перебора: {param_grid}")
        print(f"{'=' * 60}")

        model_factory = self._as_model_factory(model)
        pipeline_factory = self.make_pipeline_factory(preprocessor, model_factory)

        tuner = GridSearchTuner(
            pipeline_factory=pipeline_factory,
            validator=validator,
            param_grid=param_grid,
            scoring_metric=scoring_metric,
        )

        result = tuner.tune(X, y, class_labels=class_labels)
        self.results[name] = result

        if save:
            self.save_result(name)
            self.save_detailed(name)
            if save_pipeline:
                self.save_pipeline(name)

        return result


    def get_pipeline(self, name: str) -> Pipeline:
        """
        Возвращает обученный Pipeline:
        - run_single      → финальный Pipeline
        - run_gridsearch  → лучшая комбинация
        """
        if name not in self.results:
            raise ValueError(f"Результат '{name}' не найден")

        result = self.results[name]
        if 'best_result' in result:
            result = result['best_result']

        pipeline = result.get('pipeline')
        if pipeline is None:
            raise ValueError(
                f"Для '{name}' нет обученного Pipeline "
                f"(возможно, результат был загружен из JSON)"
            )
        return pipeline


    def save_pipeline(self, name: str, compress: int = 3) -> str:
        """
        Сохраняет обученный Pipeline (лучший для gridsearch) в
        results/<name>/pipeline.joblib.
        """
        if name not in self.results:
            raise ValueError(f"Результат '{name}' не найден")

        pipeline = self.get_pipeline(name)
        filepath = self._exp_dir(name) / "pipeline.joblib"
        joblib.dump(pipeline, filepath, compress=compress)
        print(f"Pipeline сохранён: {filepath}")
        return str(filepath)

    @staticmethod
    def load_pipeline(path: str | Path) -> Pipeline:
        """Загружает Pipeline, сохранённый через save_pipeline()."""
        pipeline = joblib.load(path)
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"Файл '{path}' содержит объект типа "
                f"{type(pipeline).__name__}, ожидался Pipeline"
            )
        return pipeline


    def save_result(self, name: str, format: str = 'json') -> str:
        if name not in self.results:
            raise ValueError(f"Результат '{name}' не найден")

        result = self._strip_pipelines(self.results[name])
        if 'best_result' in result:
            result_to_save = {k: v for k, v in result.items() if k != 'best_result'}
        else:
            result_to_save = result

        result_to_save = self._slim_result(result_to_save)

        exp_dir = self._exp_dir(name)

        if format == 'json':
            filepath = exp_dir / "result.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(result_to_save), f,
                        indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Результат сохранён: {filepath}")
        return str(filepath)


    def save_detailed(self, name: str, format: str = 'json') -> str:
        """
        Сохраняет ПОЛНЫЙ результат без «слима»:
        - metrics[*].values (per-fold),
        - полные ROC/PR-кривые (fpr/tpr/precision/recall),
        - confusion_matrix, learning_curves, training_data.

        Для run_gridsearch  → best_result   → results/<name>/best_detailed.json
        Для run_single      → result        → results/<name>/detailed.json
        """
        if name not in self.results:
            raise ValueError(f"Результат '{name}' не найден")

        result = self.results[name]
        if 'best_result' in result:
            full = self._strip_pipelines(result['best_result'])
            filename = 'best_detailed.json'
        else:
            full = self._strip_pipelines(result)
            filename = 'detailed.json'

        exp_dir = self._exp_dir(name)

        if format == 'json':
            filepath = exp_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(full), f,
                        indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Detailed сохранён: {filepath}")
        return str(filepath)


    @staticmethod
    def _strip_pipelines(obj: Any) -> Any:
        if isinstance(obj, Pipeline):
            return None
        if isinstance(obj, dict):
            return {
                k: ExperimentRunner._strip_pipelines(v)
                for k, v in obj.items()
                if not isinstance(v, Pipeline)
            }
        if isinstance(obj, list):
            return [ExperimentRunner._strip_pipelines(item) for item in obj]
        return obj

    def _make_serializable(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, (np.int32, np.int64, np.int_)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        if isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        return obj


    @staticmethod
    def _slim_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Оставляем только mean/std — per-fold values в JSON не нужны."""
        return {
            name: {'mean': v.get('mean'), 'std': v.get('std')}
            for name, v in metrics.items()
        }

    @staticmethod
    def _slim_roc(roc: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(roc, dict):
            return roc
        if 'binary' in roc:
            b = roc['binary']
            return {'binary': {
                'pos_label': b.get('pos_label'),
                'auc': b.get('auc'),
            }}
        out: Dict[str, Any] = {}
        if 'macro_auc' in roc:
            out['macro_auc'] = roc['macro_auc']
        if 'per_class' in roc:
            out['per_class'] = {
                k: {'auc': v.get('auc')}
                for k, v in roc['per_class'].items()
            }
        return out

    @staticmethod
    def _slim_pr(pr: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(pr, dict):
            return pr
        if 'binary' in pr:
            b = pr['binary']
            return {'binary': {
                'pos_label': b.get('pos_label'),
                'average_precision': b.get('average_precision'),
            }}
        out: Dict[str, Any] = {}
        if 'macro_ap' in pr:
            out['macro_ap'] = pr['macro_ap']
        if 'per_class' in pr:
            out['per_class'] = {
                k: {'average_precision': v.get('average_precision')}
                for k, v in pr['per_class'].items()
            }
        return out

    @staticmethod
    def _slim_curves(obj: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(obj)
        if 'roc_curves' in out:
            out['roc_curves'] = ExperimentRunner._slim_roc(out['roc_curves'])
        if 'pr_curves' in out:
            out['pr_curves'] = ExperimentRunner._slim_pr(out['pr_curves'])
        return out

    @staticmethod
    def _slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Убирает из result тяжёлые поля перед записью в JSON:
          - metrics: только mean/std (без per-fold values);
          - roc_curves / pr_curves: только AUC/AP (без массивов точек);
          - training_data: длинные списки → {length, head, tail, truncated};
          - для gridsearch — каждый элемент all_results приводится
            к виду {params, metrics, holdout} (без full_result, config,
            timing, training_data и пр.).
        """
        out = dict(result)
        if 'metrics' in out:
            out['metrics'] = ExperimentRunner._slim_metrics(out['metrics'])
        if 'holdout' in out:
            out['holdout'] = ExperimentRunner._slim_curves(out['holdout'])
        if 'detailed' in out:
            out['detailed'] = ExperimentRunner._slim_curves(out['detailed'])
        if 'training_data' in out:
            out['training_data'] = _summarize_training(out['training_data'])
        if 'all_results' in out:
            new_items = []
            for item in out['all_results']:
                if (
                    isinstance(item, dict)
                    and 'metrics' in item
                    and 'holdout' in item
                ):
                    new_items.append({
                        'params': item.get('params'),
                        'metrics': ExperimentRunner._slim_metrics(item['metrics']),
                        'holdout': ExperimentRunner._slim_curves(item['holdout']),
                    })
                else:
                    # уже слим или неожиданный формат — пропускаем как есть
                    new_items.append(item)
            out['all_results'] = new_items
        return out