from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Optional

from .evaluator import ModelEvaluator
from .pipeline import Pipeline
from .validators.base import BaseValidator


class GridSearchTuner:
    """
    GridSearch через cross_validate.

    Для каждой комбинации гиперпараметров считаются базовые метрики.
    Детальные артефакты (ROC/PR-кривые, learning curves, confusion matrix,
    OOF-предсказания) считаются ТОЛЬКО для лучшей комбинации — ровно один
    дополнительный прогон.

    `pipeline_factory` — callable, принимающий гиперпараметры модели как
    **kwargs и возвращающий свежий Pipeline. Например:

        pipeline_factory=lambda **kw: Pipeline(my_prep, XGBClassifier(**kw))
    """

    def __init__(
        self,
        pipeline_factory: Callable[..., Pipeline],
        validator: BaseValidator,
        param_grid: Dict[str, List[Any]],
        scoring_metric: str = 'f1_macro',
    ):
        self.pipeline_factory = pipeline_factory
        self.validator = validator
        self.param_grid = param_grid
        self.scoring_metric = scoring_metric


    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def _factory_for(self, params: Dict[str, Any]) -> Callable[[], Pipeline]:
        """Zero-arg фабрика для конкретной комбинации гиперпараметров."""
        return partial(self.pipeline_factory, **params)


    def tune(
        self,
        X: Any,
        y: Any,
        class_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        combinations = self._generate_param_combinations()

        all_results: List[Dict[str, Any]] = []
        best_score = -float('inf')
        best_params: Optional[Dict[str, Any]] = None

        for i, params in enumerate(combinations):
            print(f"  [GridSearch] Комбинация {i + 1}/{len(combinations)}: {params}")

            evaluator = ModelEvaluator(self._factory_for(params), self.validator)
            result = evaluator.evaluate(
                X, y,
                class_labels=class_labels,
                compute_detailed=False,
            )
            score = float(result['metrics'][self.scoring_metric]['mean'])

            all_results.append({
                'params': params,
                'metrics': result['metrics'],
                'holdout': result['holdout'],
            })

            if score > best_score:
                best_score = score
                best_params = params

        print(
            f"  [GridSearch] Детальный анализ для лучшей комбинации: {best_params}"
        )
        evaluator = ModelEvaluator(self._factory_for(best_params), self.validator)
        best_result = evaluator.evaluate(
            X, y,
            class_labels=class_labels,
            compute_detailed=True,
        )
        best_result['params'] = best_params
        best_result['scoring_value'] = float(
            best_result['metrics'][self.scoring_metric]['mean']
        )

        return {
            'tuning_strategy': {
                'method': 'GridSearch',
                'param_grid': self.param_grid,
                'scoring_metric': self.scoring_metric,
                'n_combinations': len(combinations),
            },
            'all_results': all_results,
            'best_params': best_params,
            'best_result': best_result,
        }