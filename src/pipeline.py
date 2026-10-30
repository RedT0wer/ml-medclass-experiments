from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import copy
import json

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .preprocessors.base import BasePreprocessor
from .models.base import BaseClassifier


class Pipeline(BaseEstimator, ClassifierMixin):
    """Единый фасад: препроцессор + модель.

    Совместим со sklearn API. После fit() доступны:
      - preprocessor_       : обученный препроцессор
      - model_              : обученная модель
      - classes_            : уникальные метки
      - n_features_in_      : число входных признаков (после отбора!)
      - feature_names_in_   : np.ndarray имён входных колонок (если известны)
      - feature_names_out_  : имена колонок после препроцессинга (если есть)
    """

    def __init__(self, preprocessor: BasePreprocessor, model: BaseClassifier):
        self.preprocessor = preprocessor
        self.model = model


    def __sklearn_clone__(self) -> 'Pipeline':
        return Pipeline(self.preprocessor, self.model)


    def fit(self, X, y, eval_set=None) -> 'Pipeline':
        y = np.asarray(y)

        self.preprocessor_ = copy.deepcopy(self.preprocessor)
        X_proc = self.preprocessor_.fit_transform(X, y)

        eval_set_t = None
        if eval_set is not None:
            eval_set_t = [
                (self.preprocessor_.transform(Xe), np.asarray(ye))
                for Xe, ye in eval_set
            ]

        self.model_ = copy.deepcopy(self.model)
        self.model_.fit(X_proc, y, eval_set=eval_set_t)

        self.classes_ = np.unique(y)

        # Если до fit() пользователь выкинул часть признаков из X,
        # feature_names_in_ отразит уже отобранный набор.
        prep_names = getattr(self.preprocessor_, 'feature_names_in_', None)
        if prep_names is None and hasattr(X, 'columns'):
            prep_names = list(X.columns)
        if prep_names is not None:
            self.feature_names_in_ = np.asarray(
                [str(n) for n in prep_names], dtype=object
            )
            self.n_features_in_ = int(len(prep_names))
        else:
            self.feature_names_in_ = None
            self.n_features_in_ = getattr(
                self.preprocessor_, 'n_features_in_', None
            )

        self.feature_names_out_ = self._extract_feature_names_out()
        return self


    def _transform(self, X) -> np.ndarray:
        if getattr(self, 'preprocessor_', None) is None:
            raise RuntimeError("Pipeline is not fitted")
        return self.preprocessor_.transform(X)

    def predict(self, X) -> np.ndarray:
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model_.predict_proba(self._transform(X))


    def get_training_history(self) -> Dict[str, Any]:
        if hasattr(self, 'model_'):
            return self.model_.get_training_history()
        return {'type': 'unknown'}

    def get_model_params(self) -> Dict[str, Any]:
        if hasattr(self, 'model_'):
            return self.model_.get_model_params()
        return {}


    def _as_sklearn_pipeline(self):
        from sklearn.pipeline import Pipeline as SkPipeline

        if getattr(self, 'preprocessor_', None) is None:
            raise RuntimeError("Pipeline must be fitted before export")

        native = getattr(self.preprocessor_, 'transformer_', None)
        prep = native if native is not None else self.preprocessor_

        inner = getattr(self.model_, 'model', self.model_)
        return SkPipeline([('prep', prep), ('clf', inner)])

    def _get_input_feature_names(self) -> Optional[List[str]]:
        """Имена входных колонок, увиденных препроцессором при fit."""
        names = getattr(self, 'feature_names_in_', None)
        if names is None:
            names = getattr(self.preprocessor_, 'feature_names_in_', None)
        if names is None:
            return None
        return [str(n) for n in names]

    def _extract_feature_names_out(self) -> Optional[List[str]]:
        if not hasattr(self, 'preprocessor_'):
            return None
        if not hasattr(self.preprocessor_, 'get_feature_names_out'):
            return None
        try:
            names = self.preprocessor_.get_feature_names_out()
            return list(names) if names is not None else None
        except Exception:
            return None

    def export_onnx(
        self,
        path: Union[str, Path],
        n_features: Optional[int] = None,
        input_type=None,
        *,
        meta_path: Optional[Union[str, Path]] = None,
        feature_schema: Optional[Dict[str, Any]] = None,
        write_meta: bool = True,
    ) -> Path:
        """Экспорт всего пайплайна в один ONNX-граф.

        Входы ONNX строятся по именам колонок, увиденных препроцессором
        при fit() — иначе ColumnTransformer не найдёт колонки по именам.

        По умолчанию рядом с .onnx пишется meta.json — без него
        потребитель не знает ни порядок колонок, ни классы.
        """
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import StringTensorType

        path = Path(path)
        # Если путь без расширения (например "_export") — трактуем как каталог.
        if path.suffix == '':
            path = path / 'model.onnx'
        path.parent.mkdir(parents=True, exist_ok=True)

        feature_names = self._get_input_feature_names()

        # Сверяем явно переданный n_features с тем, что реально было при fit.
        if (
            n_features is not None
            and feature_names is not None
            and len(feature_names) != n_features
        ):
            raise ValueError(
                f"n_features={n_features} не совпадает с числом признаков, "
                f"на которых обучался пайплайн ({len(feature_names)}: "
                f"{feature_names}). Возможно, при обучении были отобраны "
                f"признаки — уберите параметр n_features."
            )

        if input_type is not None:
            # Совместимость со старым API: можно передать один TensorType
            # либо готовый список (name, TensorType).
            if (
                isinstance(input_type, (list, tuple))
                and input_type
                and isinstance(input_type[0], tuple)
            ):
                input_types = list(input_type)
            else:
                input_types = [('input', input_type)]
        elif feature_names:
            # Ключевое исправление: по-колоночные входы с именами,
            # которые видел ColumnTransformer при fit.
            input_types = [
                (name, StringTensorType([None, 1])) for name in feature_names
            ]
        else:
            if n_features is None:
                n_features = getattr(self, 'n_features_in_', None)
            if n_features is None:
                raise ValueError(
                    "Не удалось определить n_features. Передайте его явно "
                    "или убедитесь, что preprocessor выставляет n_features_in_."
                )
            input_types = [('input', StringTensorType([None, n_features]))]

        sklearn_pipe = self._as_sklearn_pipeline()
        onnx_model = to_onnx(
            sklearn_pipe,
            initial_types=input_types,
            options={'zipmap': False},
        )

        path.write_bytes(onnx_model.SerializeToString())

        if write_meta:
            if meta_path is None:
                meta_path = path.parent / 'meta.json'
            self.export_meta(
                Path(meta_path),
                feature_schema=feature_schema,
                model_inputs=_describe_inputs(input_types),
            )

        return path

    def export_meta(
        self,
        path: Union[str, Path],
        feature_schema: Optional[Dict[str, Any]] = None,
        model_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Сохраняет meta.json с минимально необходимой схемой использования.

        Содержит:
          - model_inputs: имена/типы/формы входов ONNX (в порядке,
            ожидаемом моделью). Этого достаточно, чтобы корректно
            подать данные в модель.
          - feature_names_in: имена признаков, увиденных при обучении.
          - classes: порядок выходных классов.
        """
        if getattr(self, 'preprocessor_', None) is None:
            raise RuntimeError("Pipeline must be fitted before export")

        feature_names_in = self._get_input_feature_names()
        feature_names_out = getattr(self, 'feature_names_out_', None)
        if feature_names_out is None:
            feature_names_out = self._extract_feature_names_out()

        meta: Dict[str, Any] = {
            'format': 'onnx',
            'model': type(self.model_).__name__,
            'model_params': self.get_model_params(),
            'n_features_in': getattr(self, 'n_features_in_', None),
            'feature_names_in': feature_names_in,
            'feature_names_out': feature_names_out,
            'classes': (
                self.classes_.tolist() if hasattr(self, 'classes_') else None
            ),
        }
        if model_inputs is not None:
            meta['model_inputs'] = model_inputs
        if feature_schema is not None:
            meta['input_schema'] = feature_schema

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default),
            encoding='utf-8',
        )
        return path

    def export(
        self,
        out_dir: Union[str, Path],
        feature_schema: Optional[Dict[str, Any]] = None,
        formats: Tuple[str, ...] = ('onnx',),
        n_features: Optional[int] = None,
        input_type=None,
    ) -> Dict[str, Path]:
        """Экспорт во все запрошенные форматы. meta.json пишется всегда."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Path] = {}
        meta_path = out_dir / 'meta.json'

        if 'onnx' in formats:
            artifacts['onnx'] = self.export_onnx(
                out_dir / 'model.onnx',
                n_features=n_features,
                input_type=input_type,
                meta_path=meta_path,
                feature_schema=feature_schema,
                write_meta=True,
            )
        else:
            # meta.json — обязателен, даже если onnx не запрашивали
            self.export_meta(meta_path, feature_schema=feature_schema)

        artifacts['meta'] = meta_path
        return artifacts




def _describe_inputs(input_types) -> List[Dict[str, Any]]:
    """Человекочитаемое описание initial_types для meta.json."""
    dtype_map = {
        'StringTensorType': 'string',
        'FloatTensorType': 'float',
        'DoubleTensorType': 'double',
        'Int64TensorType': 'int64',
        'Int32TensorType': 'int32',
        'BooleanTensorType': 'bool',
    }
    out: List[Dict[str, Any]] = []
    for name, t in input_types:
        cls = type(t).__name__
        shape = getattr(t, 'shape', None)
        out.append({
            'name': str(name),
            'shape': list(shape) if shape is not None else None,
            'dtype': dtype_map.get(cls, cls),
        })
    return out


def _safe_params(obj) -> Dict[str, Any]:
    try:
        return dict(obj.get_params(deep=False))
    except Exception:
        return {}


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _summarize_training(report, max_list_len: int = 20):
    """Сжимает training_history для export_meta / JSON-отчётов."""
    if isinstance(report, np.ndarray):
        return _summarize_training(report.tolist(), max_list_len)
    if isinstance(report, dict):
        return {k: _summarize_training(v, max_list_len) for k, v in report.items()}
    if isinstance(report, (list, tuple)):
        n = len(report)
        if n <= max_list_len:
            return [_summarize_training(x, max_list_len) for x in report]
        return {
            'length': n,
            'head': [_summarize_training(x, max_list_len) for x in report[:3]],
            'tail': [_summarize_training(x, max_list_len) for x in report[-3:]],
            'truncated': True,
        }
    if isinstance(report, (np.integer,)):
        return int(report)
    if isinstance(report, (np.floating,)):
        return float(report)
    return report