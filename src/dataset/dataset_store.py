import json
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import pandas as pd
from pandas.api.types import CategoricalDtype

from .schema_config import (
    DEFAULT_MISSING_VALUE,
)

class DatasetStore:
    """Сохранение/загрузка DataFrame вместе с JSON-схемой."""

    @staticmethod
    def save(
            df: pd.DataFrame,
            filepath: str,
            schema: Dict[str, Any],
            schema_path: str,
            **to_csv_kwargs
    ) -> None:
        df.to_csv(filepath, index=False, **to_csv_kwargs)
        Path(schema_path).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.exists():
            return p
        parent_p = Path("..") / path_str
        if parent_p.exists():
            return parent_p
        # Also check relative to workspace root if possible
        repo_root = Path(__file__).resolve().parents[2]
        repo_p = repo_root / path_str
        if repo_p.exists():
            return repo_p
        return p

    @staticmethod
    def load(
            filepath: str,
            schema_path: str,
            include_missing: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        include_missing:
            * True  — оставляем маркер пропуска в categories. Значения
              "MISSING" останутся отдельной категорией.
            * False — убираем маркер из categories. Тогда read_csv
              автоматически превратит "MISSING" в NaN (поведение
              CategoricalDtype: значения вне categories → NaN).
        """
        resolved_schema_path = DatasetStore._resolve_path(schema_path)
        resolved_filepath = DatasetStore._resolve_path(filepath)

        schema = json.loads(resolved_schema_path.read_text(encoding='utf-8'))
        missing_value = schema.get('missing_value', DEFAULT_MISSING_VALUE)

        dtypes = {}
        for col, spec in schema['features'].items():
            ftype = spec.get('type')
            cats = spec.get('categories')
            if not cats:
                continue
            if not include_missing:
                cats = [c for c in cats if c != missing_value]
            if ftype == 'ordinal':
                dtypes[col] = CategoricalDtype(categories=cats, ordered=True)
            elif ftype == 'categorical':
                dtypes[col] = CategoricalDtype(categories=cats, ordered=False)

        df = pd.read_csv(resolved_filepath, dtype=dtypes, delimiter=";")
        return df, schema

    @staticmethod
    def get_feature_info(
            schema: Dict[str, Any],
            include_missing: bool = True,
            excluded_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        include_missing:
            * True  — маркер пропуска остаётся в ordinal_categories,
              а missing_value возвращается потребителю.
            * False — маркер вычищается из ordinal_categories,
              а missing_value = None.
        """
        excluded_features = set(excluded_features or [])
        missing_value = schema.get('missing_value', DEFAULT_MISSING_VALUE)

        categorical_features: List[str] = []
        ordinal_features: List[str] = []
        ordinal_categories: Dict[str, List[str]] = {}

        features: Dict[str, Dict[str, Any]] = schema.get('features', {})

        for feature_name, spec in features.items():
            if feature_name in excluded_features:
                continue

            feature_type = spec.get('type')

            if feature_type == 'categorical':
                categorical_features.append(feature_name)
            elif feature_type == 'ordinal':
                ordinal_features.append(feature_name)
                cats = list(spec.get('categories', []))
                if not include_missing:
                    cats = [c for c in cats if c != missing_value]
                ordinal_categories[feature_name] = cats

        return {
            'categorical_features': categorical_features,
            'ordinal_features': ordinal_features,
            'ordinal_categories': ordinal_categories,
            'all_features': [
                name for name in features.keys()
                if name not in excluded_features
            ],
            'missing_value': missing_value if include_missing else None,
        }