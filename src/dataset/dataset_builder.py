from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

DEFAULT_MISSING_VALUE = "MISSING"


@dataclass(frozen=True)
class MissingValueConfig:
    """Единая правда о представлении пропусков в CSV и в JSON-схеме."""

    value: str = DEFAULT_MISSING_VALUE
    include_in_categories: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("MissingValueConfig.value должен быть непустой строкой")

    def cell_value(self) -> Optional[str]:
        """Что записывать в ячейку CSV при пропуске (None -> NaN/пусто)."""
        return self.value if self.include_in_categories else None

    def categories_suffix(self) -> List[str]:
        """Хвост, добавляемый к categories в схеме."""
        return [self.value] if self.include_in_categories else []



class DatasetBuilder:

    def __init__(
            self,
            features_file: str,
            value_features_file: str,
            data_file: str,
            target_name: str = 'target',
            missing_config: Optional[MissingValueConfig] = None,
    ):
        self.features_file = Path(features_file)
        self.value_features_file = Path(value_features_file)
        self.data_file = Path(data_file)
        self.target_name = target_name

        # единая правда о пропусках
        self.missing_config = missing_config or MissingValueConfig()

        self.features_df: Optional[pd.DataFrame] = None
        self.value_features_df: Optional[pd.DataFrame] = None
        self.raw_data: Optional[List[List[int]]] = None
        self.feature_ids: Optional[List[int]] = None

        self._value_mapping: Optional[Dict[int, Dict[int, str]]] = None
        self._feature_names: Optional[Dict[int, str]] = None
        self._feature_types: Optional[Dict[int, str]] = None


    def load_features(self) -> pd.DataFrame:
        if self.features_df is None:
            self.features_df = pd.read_csv(self.features_file, sep=';', encoding='utf-8')
            self.features_df.columns = self.features_df.columns.str.strip()
        return self.features_df

    def load_value_features(self) -> pd.DataFrame:
        if self.value_features_df is None:
            self.value_features_df = pd.read_csv(self.value_features_file, sep=';', encoding='utf-8')
            self.value_features_df = self.value_features_df.loc[
                :, ~self.value_features_df.columns.str.contains('^Unnamed')
            ]
            self.value_features_df.columns = self.value_features_df.columns.str.strip()
            self.value_features_df['Name'] = self.value_features_df['Name'].str.strip()
        return self.value_features_df

    def load_data(self) -> Tuple[List[int], List[List[int]]]:
        if self.raw_data is None:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            first_line = lines[0].strip().split()
            self.feature_ids = [int(x) for x in first_line[1:-1]]

            data_rows = []
            for line in lines[1:]:
                values = line.strip().split()
                if len(values) > 2:
                    data_rows.append([int(x) for x in values])

            self.raw_data = data_rows
        return self.feature_ids, self.raw_data


    def _create_value_mapping(self) -> Dict[int, Dict[int, str]]:
        if self._value_mapping is None:
            self.load_value_features()
            mapping: Dict[int, Dict[int, str]] = {}
            for _, row in self.value_features_df.iterrows():
                param_id = int(row['paramID'])
                val_id = int(row['valID'])
                mapping.setdefault(param_id, {})[val_id] = row['Name']
            self._value_mapping = mapping
        return self._value_mapping

    def _get_feature_names(self) -> Dict[int, str]:
        if self._feature_names is None:
            self.load_features()
            self._feature_names = {
                int(row['ID_Char']): row['Name']
                for _, row in self.features_df.iterrows()
            }
        return self._feature_names

    def _get_feature_types(self) -> Dict[int, str]:
        if self._feature_types is None:
            self.load_features()
            self._feature_types = {
                int(row['ID_Char']): row['Type']
                for _, row in self.features_df.iterrows()
            }
        return self._feature_types


    def create_dataframe(
            self,
            excluded_features: Optional[List[str]] = None,
            use_value_labels: bool = True
    ) -> pd.DataFrame:
        self.load_features()
        self.load_value_features()
        self.load_data()

        feature_names = self._get_feature_names()
        value_mapping = self._create_value_mapping()
        excluded_features = set(excluded_features or [])

        missing_cell = self.missing_config.cell_value()  # <- единая точка

        rows = []
        for data_row in self.raw_data:
            target = data_row[-1]
            feature_values = data_row[1:-1]

            row_dict = {}
            for i, feature_id in enumerate(self.feature_ids):
                feature_name = feature_names.get(feature_id, f'feature_{feature_id}')
                if feature_name in excluded_features:
                    continue
                if i >= len(feature_values):
                    continue

                raw_value = feature_values[i]
                is_valid = (
                        raw_value != -1
                        and feature_id in value_mapping
                        and raw_value in value_mapping[feature_id]
                )
                if not is_valid:
                    row_dict[feature_name] = missing_cell
                elif use_value_labels:
                    row_dict[feature_name] = value_mapping[feature_id][raw_value]
                else:
                    row_dict[feature_name] = raw_value

            row_dict[self.target_name] = target
            rows.append(row_dict)

        return pd.DataFrame(rows)


    def _build_schema(
            self,
            df: pd.DataFrame,
            excluded_features: Optional[List[str]]
    ) -> Dict[str, Any]:
        feature_names = self._get_feature_names()
        feature_types = self._get_feature_types()
        value_mapping = self._create_value_mapping()
        excluded_features = set(excluded_features or [])

        missing_suffix = self.missing_config.categories_suffix()  # <- единая точка

        features: Dict[str, Dict[str, Any]] = {}

        for feature_id, feature_type in feature_types.items():
            feature_name = feature_names.get(feature_id, f'feature_{feature_id}')
            if feature_name in excluded_features or feature_name not in df.columns:
                continue

            if feature_type in ('ран.', 'кат.'):
                cats = [name for _, name in sorted(value_mapping.get(feature_id, {}).items())]
                cats.extend(missing_suffix)
                features[feature_name] = {
                    'type': 'ordinal' if feature_type == 'ран.' else 'categorical',
                    'categories': cats,
                }

        return {
            'target': self.target_name,
            'features': features,
            'missing_value': self.missing_config.value,
            'missing_value_in_categories': self.missing_config.include_in_categories,
        }


    def save_dataset(
            self,
            filepath: str = "dataset.csv",
            schema_path: Optional[str] = None,
            excluded_features: Optional[List[str]] = None,
            use_value_labels: bool = True,
            **kwargs
    ) -> None:
        if not use_value_labels:
            raise ValueError(
                "Схему можно строить только для CSV с текстовыми метками "
                "(use_value_labels=True)."
            )

        if schema_path is None:
            schema_path = str(Path(filepath).with_suffix('.schema.json'))

        df = self.create_dataframe(
            excluded_features=excluded_features,
            use_value_labels=use_value_labels
        )
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False, **kwargs)

        schema = self._build_schema(df, excluded_features)
        Path(schema_path).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )