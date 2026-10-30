# ml-medclass-experiments

Классификация структурированных клинических медицинских данных (дифференциальная диагностика острого панкреатита: отёчный панкреатит / панкреонекроз).

Репозиторий представляет собой модульный ML-фреймворк на базе архитектуры единых пайплайнов (`Pipeline`), объединяющих предобработку признаков, модели классификации, гибкую систему валидации данных и экспорт моделей в форматы **Joblib** и **ONNX** с автовалидацией точности инференса в **ONNX Runtime**.

---

## Основные возможности

- **Архитектура единого Pipeline (`src/pipeline/`)**:
  - Полная интеграция препроцессора и модели классификатора в единый объект.
  - Совместимость со Scikit-Learn API (`fit`, `predict`, `predict_proba`, `clone`, `get_params`, `set_params`).
  - Честная кросс-валидация без утечки данных (`evaluate_cv`): препроцессор обучается строго на обучающих фолдах.
  - Атомарная сериализация и загрузка полного пайплайна в один `.joblib` файл (`pipe.save()`, `Pipeline.load()`).
- **Реестр препроцессоров (`src/preprocessors/`)**:
  - `booster`: специализированная предобработка для градиентных бустингов и решающих деревьев (CatBoost, LightGBM, XGBoost, Random Forest).
  - `onehot` / `scaled_onehot`: универсальная предобработка со Scikit-Learn OneHotEncoder, опциональным масштабированием числовых (`StandardScaler`) и порядковых признаков.
  - Автоматическая поддержка JSON-схемы данных (`dataset.schema.json`): извлечение категориальных, порядковых признаков и явных порядков категорий.
- **Реестр валидаторов (`src/validators/`)**:
  - Единый источник конфигурации — `configs/validators.yaml`.
  - Поддержка стратегий: `stratified_kfold`, `kfold`, `stratified_shuffle_split`, `shuffle_split`, `time_series_split`, `train_test_split`.
  - Выбор стратегии по названию с автоматическим fallback на `default` из конфигурации.
- **Реестр моделей на Duck Typing (`src/models/`)**:
  - Поддержка алгоритмов: CatBoost, LightGBM, XGBoost, Logistic Regression, Random Forest.
  - Легковесная регистрация через декоратор `@register_model`.
- **Реестр метрик (`src/metrics/`)**:
  - Расчет стандартных и кастомных метрик: `roc_auc`, `pr_auc`, `accuracy`, `f1_macro`, `f1_micro`, `balanced_accuracy`, `precision`, `recall`.
  - Поддержка как вероятностных метрик, так и метрик на предсказанных классах.
- **Подбор гиперпараметров (`src/tuning/`)**:
  - Оптимизация параметров полного пайплайна через `GridSearchCV` и `RandomizedSearchCV`.
  - Загрузка сеток параметров из `configs/tuning.yaml`.
  - Использование зарегистрированных валидаторов по названию или по умолчанию.
  - Сохранение результатов в формате JSON и обученного пайплайна в `.joblib`.
- **Экспорт и автовалидация ONNX (`src/export/`)**:
  - Конвертация моделей в ONNX (CatBoost Native ONNX, ONNXMLTools для LightGBM/XGBoost, Sklearn-ONNX для Scikit-Learn).
  - Автоматическая проверка эквивалентности вероятностей в ONNX Runtime с контролем расхождения (`atol=1e-5`).
  - Утилита независимой проверки графа и бенчмаркинга задержки инференса (`verify-onnx`).
- **Консольный интерфейс CLI (`cli.py`)**:
  - Управление полным циклом экспериментов: подготовка данных, обзор реестров, обучение, бенчмарк, тюнинг и валидация ONNX.

---

## Структура репозитория

```
.
├── configs/                       # Конфигурационные файлы (YAML)
│   ├── models.yaml                # Параметры моделей
│   ├── preprocessors.yaml         # Привязка препроцессоров к моделям и их настройки
│   ├── tuning.yaml                # Сетки гиперпараметров для Grid/Random Search
│   └── validators.yaml            # Единственный источник стратегий кросс-валидации и default
├── data/
│   ├── raw/                       # Исходные файлы (algorithm.txt, features.csv, value_features.csv)
│   └── processed/                 # Сформированные датасеты (dataset.parquet, dataset.csv, dataset.schema.json)
├── _results/                      # Выходные артефакты (leaderboard.csv, best_params_*.json, .onnx, .joblib)
├── src/                           # Библиотека модулей фреймворка
│   ├── dataset/                   # Загрузка данных и генерация схемы (loader.py, dataset_builder.py)
│   ├── export/                    # Экспорт моделей в ONNX/Joblib и верификация (exporter.py)
│   ├── metrics/                   # Реестр метрик качества классификации (registry.py)
│   ├── models/                    # Реестр классификаторов (registry.py)
│   ├── pipeline/                  # Базовый класс Pipeline и фабрика create_pipeline (pipeline.py)
│   ├── preprocessors/             # Препроцессоры признаков и их реестр (base.py, booster.py, onehot.py, registry.py)
│   ├── tuning/                    # Модуль подбора гиперпараметров (tuner.py)
│   └── validators/                # Реестр и фабрика валидаторов (registry.py)
├── tests/                         # Автоматические тесты (pytest)
│   ├── test_cli.py                # Тесты консольного интерфейса
│   ├── test_export.py             # Тесты экспорта и валидации ONNX
│   ├── test_metrics.py            # Тесты реестра метрик
│   ├── test_model_registry.py     # Тесты реестра моделей
│   ├── test_pipeline.py           # Тесты функциональности Pipeline
│   ├── test_preprocessors.py      # Тесты препроцессоров и работы со схемой
│   ├── test_tuner.py              # Тесты подбора гиперпараметров
│   └── test_validators.py         # Тесты реестра валидаторов и YAML-конфигурации
├── cli.py                         # Консольный интерфейс командной строки
├── requirements.txt               # Зависимости Python
└── pytest.ini                     # Настройки pytest
```

---

## Установка и настройка

```bash
# Клонирование репозитория
git clone <url-репозитория>
cd ml-medclass-experiments

# Создание и активация виртуального окружения
python -m venv .venv

# Для Linux/macOS:
source .venv/bin/activate
# Для Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Установка зависимостей
pip install -r requirements.txt

# Проверка работоспособности компонентов
python cli.py list
```

---

## Конфигурационные файлы (`configs/`)

1. **`configs/validators.yaml`** — единственный источник конфигурации валидаторов.
   ```yaml
   default: stratified_kfold       # Валидатор по умолчанию

   stratified_kfold:
     n_splits: 5
     shuffle: true
     random_state: 42

   kfold:
     n_splits: 5
     shuffle: true
     random_state: 42
   ```
2. **`configs/preprocessors.yaml`** — сопоставление препроцессоров моделям и их параметры.
   ```yaml
   model_preprocessors:
     catboost: booster
     logreg: scaled_onehot

   preprocessors:
     scaled_onehot:
       missing_value: "отсутствует"
       scale_numeric: true
       scale_ordinal_features: true
   ```
3. **`configs/tuning.yaml`** — сетки параметров для `GridSearchCV` и `RandomizedSearchCV`.
   ```yaml
   logreg:
    C: [0.01, 0.1, 1.0, 10.0]
    solver: ["lbfgs", "saga"]
  
    random_forest:
      n_estimators: [50, 100, 200]
      max_depth: [null, 5, 10]
      min_samples_split: [2, 5]
   ```
4. **`configs/models.yaml`** — параметры для моделей.
   ```yaml
   catboost:
    iterations: 200
    depth: 5
    learning_rate: 0.05
    random_seed: 42
    verbose: 0
    thread_count: -1
  
    xgboost:
      n_estimators: 100
      max_depth: 5
      learning_rate: 0.1
      random_state: 42
      eval_metric: "mlogloss"
      n_jobs: -1
   ```

---

## Консольный интерфейс CLI (`cli.py`)

Все команды поддерживают автоматические значения по умолчанию: если путь к датасету не передан, используется `data/processed/dataset.parquet` (или `dataset.csv`).

### 1. Просмотр реестров (`list`)
Выводит форматированные таблицы всех доступных моделей, препроцессоров, метрик и валидаторов с их параметрами из `.yaml`:
```bash
python cli.py list
```

### 2. Сборка датасета из сырых файлов (`prep-data`)
Выполняет парсинг сырых файлов признаков и алгоритма, формирует чистую выборку и сопутствующую JSON-схему:
```bash
# Использование путей по умолчанию:
python cli.py prep-data

# С явным указанием путей:
python cli.py prep-data \
  --input data/raw/algorithm.txt \
  --features data/raw/features.csv \
  --value-features data/raw/value_features.csv \
  --output data/processed/dataset.parquet
```

### 3. Обучение и экспорт Pipeline (`train`)
Обучает полный пайплайн (препроцессор + классификатор), вычисляет метрики на валидационной выборке, экспортирует артефакты в Joblib и ONNX, и автоматически валидирует идентичность предсказаний в ONNX Runtime:
```bash
# Обучение модели по умолчанию (CatBoost):
python cli.py train

# Обучение с кастомным файлом параметров моделей (или JSON тюнинга):
python cli.py train --model logreg --model-params configs/models.yaml
python cli.py train --model logreg --model-params _results/best_params_logreg.json

# Обучение с явным указанием путей сохранения артефактов:
python cli.py train \
  --model catboost \
  --export-onnx _results/catboost.onnx \
  --export-joblib _results/catboost.joblib
```

### 4. Бенчмарк моделей (`benchmark`)
Запускает кросс-валидацию моделей через `Pipeline.evaluate_cv()` с расчетом средних значений и стандартных отклонений метрик.
Поддерживает выбор зарегистрированного валидатора через флаг `--validator`:
```bash
# Бенчмарк с валидатором по умолчанию (из validators.yaml):
python cli.py benchmark

# Бенчмарк выбранных моделей с указанием стратегии валидации и конфигурации параметров:
python cli.py benchmark \
  --models catboost xgboost logreg random_forest \
  --model-params configs/models.yaml \
  --metrics roc_auc f1_macro accuracy recall precision \
  --validator stratified_kfold \
  --output _results/leaderboard.csv
```

### 5. Подбор гиперпараметров (`tune`)
Выполняет оптимизацию параметров пайплайна (Grid Search или Random Search) по сетке из `configs/tuning.yaml`.
Валидатор выбирается через `--validator` (если не передан — используется `default` из `configs/validators.yaml`):
```bash
# Grid Search для CatBoost с валидатором по умолчанию:
python cli.py tune --model catboost

# Random Search для Logistic Regression с явным валидатором:
python cli.py tune \
  --model logreg \
  --validator kfold \
  --search-type random \
  --n-iter 20 \
  --scoring f1_macro \
  --output _results/
```
Результаты сохраняются в `_results/best_params_<model>.json`, а обученный пайплайн — в `_results/best_params_<model>.joblib`.

### 6. Проверка ONNX модели (`verify-onnx`)
Автономная проверка структуры графа ONNX, входов/выходов, замер средней латентности инференса и сверка предсказаний с соответствующим Joblib-пайплайном:
```bash
python cli.py verify-onnx --onnx _results/catboost.onnx
```

---

## Использование Python API

### 1. Создание и обучение Pipeline
```python
from src.dataset.loader import load_dataset, load_schema
from src.pipeline.pipeline import create_pipeline

# Загрузка данных и схемы
X, y, _ = load_dataset("data/processed/dataset.parquet")
schema = load_schema("data/processed/dataset.parquet")

# Создание Pipeline (препроцессор подбирается автоматически или задается явно)
pipe = create_pipeline(model="logreg", preprocessor="scaled_onehot", schema=schema)
pipe.fit(X, y)

# Инференс
y_pred = pipe.predict(X)
y_prob = pipe.predict_proba(X)

# Сохранение и загрузка
pipe.save("_results/logreg.joblib")
```

### 2. Кросс-валидация без утечки данных
```python
# Валидатор передается строковым именем из configs/validators.yaml или None (default)
cv_results = pipe.evaluate_cv(
    X=X,
    y=y,
    validator="stratified_kfold",
    metrics=["roc_auc", "f1_macro", "accuracy"]
)
print("F1-Macro:", cv_results["f1_macro"]["formatted"])
```

### 3. Подбор гиперпараметров в коде
```python
from src.tuning.tuner import tune_model

result = tune_model(
    model="catboost",
    X=X,
    y=y,
    validator="stratified_kfold",
    search_type="grid",
    scoring="f1_macro"
)
print("Лучший скор:", result.best_score)
print("Лучшие параметры:", result.best_params)
```

---

## Руководство разработчика: расширение системы

### Регистрация новой модели
Модели поддерживают Scikit-Learn duck typing (`fit`, `predict`, `predict_proba`) и регистрируются декоратором `@register_model`:
```python
from sklearn.neighbors import KNeighborsClassifier
from src.models.registry import register_model

@register_model(name="knn")
class RegisteredKNN(KNeighborsClassifier):
    pass
```
Параметры модели задаются в `configs/models.yaml`:
```yaml
knn:
  n_neighbors: 5
  weights: "distance"
```

### Регистрация нового препроцессора
Препроцессоры наследуются от `BasePreprocessor` и регистрируются через `@register_preprocessor`:
```python
from src.preprocessors.base import BasePreprocessor
from src.preprocessors.registry import register_preprocessor

@register_preprocessor("custom_prep")
class CustomPreprocessor(BasePreprocessor):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
```
Параметры препроцессора задаются в `configs/preprocessors.yaml`:
```yaml
preprocessors:
  custom_prep:
    scale: true
```

### Регистрация нового валидатора
Валидаторы реализуют протокол со Scikit-Learn `split(X, y)` API и регистрируются декоратором `@register_validator`:
```python
from sklearn.model_selection import RepeatedStratifiedKFold
from src.validators.registry import register_validator

@register_validator(name="repeated_stratified_kfold")
class RegisteredRepeatedSKF(RepeatedStratifiedKFold):
    pass
```
Параметры валидатора задаются в `configs/validators.yaml`:
```yaml
repeated_stratified_kfold:
  n_splits: 5
  n_repeats: 2
  random_state: 42
```
После регистрации валидатор можно настраивать в `configs/validators.yaml` и вызывать по имени во всех командах CLI и модулях.

### Регистрация новой метрики
```python
import numpy as np
from src.metrics.registry import register_metric

@register_metric("error_rate")
def compute_error_rate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> float:
    return float(np.mean(y_true != y_pred))
```

---

## Тестирование

Запуск полного набора автотестов:
```bash
pytest -v
```

Тестовый набор (57 тестов) проверяет:
- Реестр моделей и интерфейс duck typing.
- Препроцессоры признаков (`booster`, `onehot`, `scaled_onehot`), масштабирование и извлечение метаданных схемы.
- Единый `Pipeline`: обучение, сериализацию Joblib, Scikit-Learn `clone` и честную кросс-валидацию.
- Реестр валидаторов: корректность выборки по названию, fallback на `default`, ошибки при отсутствии `default` и приоритет `validators.yaml`.
- Тюнинг гиперпараметров: работу с именованными и дефолтными валидаторами, `GridSearchCV` и `RandomizedSearchCV`.
- Экспорт и валидацию эквивалентности выходов в ONNX Runtime для всех поддерживаемых семейств классификаторов.
- Все команды интерфейса командной строки (`list`, `prep-data`, `train`, `benchmark`, `tune`, `verify-onnx`).
