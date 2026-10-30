# ml-medclass-experiments

Классификация структурированных клинических медицинских данных (дифференциальная диагностика острого панкреатита: отёчный панкреатит / панкреонекроз). Репозиторий содержит исследовательские Jupyter Notebooks, библиотеку модулей (`src/`) и интерфейс командной строки (`cli.py`) для автоматизации обучения, подбора гиперпараметров, кросс-валидации и экспорта моделей в форматы ONNX и Joblib.

## Структура репозитория

```
.
├── notebooks/                     # Исследовательские ноутбуки (EDA, гипотезы, отбор признаков)
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_grid_search.ipynb
│   ├── 04_learning.ipynb
│   ├── 05_correlated_traits.ipynb
│   ├── 06_top_feature.ipynb
│   ├── 07_comparison.ipynb
│   └── 08_export_import.ipynb
├── configs/
│   └── tuning.yaml                # Сетки гиперпараметров для моделей
├── data/
│   ├── raw/                       # Исходные файлы (algorithm.txt, features.csv, value_features.csv)
│   └── processed/                 # Очищенные датасеты (dataset.parquet, dataset.csv, schema.json)
├── results/                       # Артефакты: leaderboard.csv, best_params_*.json, .onnx, .joblib
├── src/                           # Библиотека модулей
│   ├── dataset/                   # Загрузка и сборка датасетов (loader.py, dataset_builder.py)
│   ├── models/                    # Реестр моделей на duck typing (registry.py)
│   ├── metrics/                   # Реестр метрик (registry.py)
│   ├── tuning/                    # Подбор гиперпараметров (tuner.py)
│   ├── export/                    # Экспорт и валидация через ONNX Runtime (exporter.py)
│   ├── preprocessors/             # Препроцессоры признаков
│   └── validators/                # Стратегии кросс-валидации
├── tests/                         # Набор smoke-тестов на pytest
├── cli.py                         # Консольный интерфейс
├── requirements.txt               # Зависимости проекта
└── pytest.ini                     # Конфигурация pytest
```

## Рабочий процесс

1. **Исследования в Jupyter Notebooks (`notebooks/`)**:
   Разведочный анализ данных, проверка статистических гипотез и отбор признаков выполняются в ноутбуках. Отдельный пакет преобразования признаков отсутствует; исследователь формирует выборки непосредственно в интерактивной среде.

2. **Сохранение данных (`data/processed/`)**:
   Очищенные и сформированные выборки сохраняются в `data/processed/<name>.parquet` или `data/processed/<name>.csv`. Сопутствующая схема категорий сохраняется в `.schema.json`.

3. **Автоматизация через CLI и библиотеку (`cli.py`, `src/`)**:
   Пайплайны принимают путь к сформированному датасету через параметр `--dataset`. Модули выполняют кросс-валидацию, подбор гиперпараметров и экспорт моделей с автоматической валидацией эквивалентности выходов в ONNX Runtime.

## Установка и запуск

```bash
# Клонирование и переход в каталог проекта
cd ml-medclass-experiments

# Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Проверка установленных компонентов
python cli.py list
```

## Использование CLI (`cli.py`)

Все команды поддерживают значения по умолчанию: при запуске без аргументов используется `data/processed/dataset.parquet` (или `dataset.csv`), модель `catboost` и стандартный набор метрик.

### 1. Просмотр доступных моделей и метрик (`list`)
Выводит зарегистрированные модели, их классы и параметры по умолчанию, а также список метрик:
```bash
python cli.py list
```

### 2. Сборка датасета из сырых файлов (`prep-data`)
Формирует чистый датасет и JSON-схему из сырых данных:
```bash
# Сборка с параметрами по умолчанию:
python cli.py prep-data

# Сборка с указанием путей:
python cli.py prep-data --input data/raw/algorithm.txt --output data/processed/dataset.parquet
```

### 3. Обучение и экспорт модели (`train`)
Обучает классификатор, выводит метрики на отложенной выборке, сохраняет артефакты в Joblib и ONNX, и проверяет эквивалентность предсказаний в ONNX Runtime (`np.testing.assert_allclose(py_probs, onnx_probs, atol=1e-5)`):
```bash
# Обучение модели по умолчанию:
python cli.py train

# Обучение с параметрами из файла тюнинга:
python cli.py train --model logreg --params-file results/best_params_logreg.json

# Обучение с явным указанием путей экспорта:
python cli.py train --model catboost --export-onnx results/catboost.onnx --export-joblib results/catboost.joblib
```

### 4. Бенчмарк моделей (`benchmark`)
Запускает стратифицированную кросс-валидацию моделей, рассчитывает запрошенные метрики и сохраняет таблицу результатов:
```bash
# Бенчмарк всех моделей по умолчанию:
python cli.py benchmark

# Бенчмарк выбранных моделей:
python cli.py benchmark \
  --models catboost xgboost lightgbm logreg random_forest \
  --metrics roc_auc f1_macro accuracy recall precision \
  --cv 5 \
  --output results/leaderboard.csv
```

### 5. Подбор гиперпараметров (`tune`)
Выполняет подбор параметров по сетке из `configs/tuning.yaml` через `GridSearchCV` или `RandomizedSearchCV`:
```bash
# Grid search для CatBoost:
python cli.py tune --model catboost

# Random search для Logistic Regression:
python cli.py tune --model logreg --search-type random --n-iter 20 --cv 5 --output results/
```
Результаты сохраняются в `results/best_params_<model>.json`.

### 6. Проверка ONNX модели (`verify-onnx`)
Выполняет проверку графа ONNX, типов и размерностей входов/выходов, замеряет задержку инференса и сверяет предсказания с соответствующей Joblib-моделью:
```bash
python cli.py verify-onnx --onnx results/catboost.onnx
```

## Руководство разработчика

### Регистрация новой модели
Модели не требуют наследования от тяжелых абстрактных классов. Используется duck typing Scikit-Learn API (`fit`, `predict`, `predict_proba`). Регистрация выполняется декоратором `@register_model`:

```python
from sklearn.neighbors import KNeighborsClassifier
from src.models.registry import register_model

@register_model(
    name="knn",
    default_params={"n_neighbors": 5, "weights": "distance"}
)
class RegisteredKNN(KNeighborsClassifier):
    pass
```

После регистрации модель доступна во всех командах CLI (`train`, `benchmark`, `tune`) и через функцию `get_model("knn")`.

### Регистрация новой метрики
Функции расчета метрик регистрируются через декоратор `@register_metric`:

```python
import numpy as np
from src.metrics.registry import register_metric

@register_metric("error_rate")
def compute_error_rate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> float:
    return float(np.mean(y_true != y_pred))
```

После регистрации метрика доступна в аргументе `--metrics` командной строки и в функции `calculate_metrics(..., metric_names=["error_rate"])`.

## Тестирование

Запуск тестов:
```bash
pytest -v tests/
```

Набор тестов проверяет:
- Реестр моделей и Scikit-Learn duck typing.
- Вычисление стандартных и кастомных метрик в бинарном и многоклассовом режимах.
- Экспорт в Joblib и ONNX с автовалидацией в ONNX Runtime (Scikit-Learn, LightGBM, XGBoost, CatBoost).
- Все команды интерфейса командной строки (`list`, `prep-data`, `tune`, `benchmark`, `train`, `verify-onnx`).