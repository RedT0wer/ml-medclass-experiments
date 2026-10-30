"""Модуль универсальной загрузки подготовленных датасетов."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_path: str | Path,
    target_column: str | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Загружает сформированный датасет (.csv или .parquet) и разделяет его на X и y.

    Args:
        dataset_path: Путь к файлу датасета.
        target_column: Название целевой переменной. Если не указано,
                       пытается взять из .schema.json или найти колонки 'diagnosis' / 'target'.

    Returns:
        X: DataFrame с признаками.
        y: Массив меток классов, закодированных в 0, 1, ..., N-1.
        label_encoder: Обученный LabelEncoder для восстановления оригинальных меток.
    """
    path = Path(dataset_path)
    if not path.exists():
        # Проверяем относительно корня репозитория или родительской папки
        parent_p = Path("..") / path
        if parent_p.exists():
            path = parent_p
        else:
            repo_root = Path(__file__).resolve().parents[2]
            repo_p = repo_root / path
            if repo_p.exists():
                path = repo_p
            else:
                raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")

    logger.info("Загрузка датасета из: %s", path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        # Пробуем определить разделитель (; или ,)
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}. Ожидается .csv или .parquet")

    # Определение целевой переменной
    if target_column is None:
        # Проверим сопутствующую схему
        schema_path = path.with_suffix(".schema.json")
        if schema_path.exists():
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                target_column = schema.get("target")
            except Exception as e:
                logger.warning("Не удалось прочитать целевую переменную из схемы %s: %s", schema_path, e)

        if target_column is None or target_column not in df.columns:
            for candidate in ["diagnosis", "target", "label"]:
                if candidate in df.columns:
                    target_column = candidate
                    break

        if target_column is None:
            # По умолчанию берем последнюю колонку
            target_column = df.columns[-1]
            logger.info("Целевая колонка выбрана по умолчанию: '%s'", target_column)

    if target_column not in df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' не найдена в датасете. Колонки: {df.columns.tolist()}")

    X = df.drop(columns=[target_column]).copy()
    y_raw = df[target_column].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    logger.info(
        "Датасет успешно загружен: X.shape=%s, классов=%d, метки=%s -> %s",
        X.shape,
        len(label_encoder.classes_),
        label_encoder.classes_.tolist(),
        np.unique(y).tolist(),
    )
    return X, y, label_encoder
