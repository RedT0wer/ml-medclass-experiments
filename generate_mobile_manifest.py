import json
from pathlib import Path
import pandas as pd
import numpy as np

# Загружаем обработанный датасет для извлечения признаков и их диапазонов
dataset_path = Path("data/processed/dataset.parquet")
if not dataset_path.exists():
    dataset_path = Path("data/processed/dataset.csv")

if dataset_path.suffix == ".parquet":
    df = pd.read_parquet(dataset_path)
else:
    df = pd.read_csv(dataset_path)

# Исключаем целевую переменную
target_col = "target" if "target" in df.columns else df.columns[-1]
features_cols = [c for c in df.columns if c != target_col]

features_meta = []
for col in features_cols:
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    features_meta.append({
        "key": col,
        "title": col.replace("_", " ").capitalize(),
        "type": "float32",
        "min": float(np.round(series.min(), 2)) if not series.empty else 0.0,
        "max": float(np.round(series.max(), 2)) if not series.empty else 100.0,
        "default": float(np.round(series.median(), 2)) if not series.empty else 0.0
    })

manifest = {
    "version": "1.0.0",
    "target_classes": {
        "0": "Отёчный панкреатит (низкий риск)",
        "1": "Панкреонекроз (высокий риск)"
    },
    "features": features_meta,
    "models": [
        {
            "id": "catboost",
            "name": "CatBoost Classifier",
            "file": "catboost.onnx",
            "description": "Максимальная обобщающая способность на несбалансированных данных",
            "input_name": "features",
            "output_name": "probabilities"
        },
        {
            "id": "xgboost",
            "name": "XGBoost Classifier",
            "file": "xgboost.onnx",
            "description": "Классический градиентный бустинг с регуляризацией",
            "input_name": "input",
            "output_name": "probabilities"
        },
        {
            "id": "lightgbm",
            "name": "LightGBM Classifier",
            "file": "lightgbm.onnx",
            "description": "Быстрый инференс на мобильных CPU",
            "input_name": "input",
            "output_name": "probabilities"
        },
        {
            "id": "random_forest",
            "name": "Random Forest",
            "file": "random_forest.onnx",
            "description": "Ансамбль решающих деревьев с высокой устойчивостью к выбросам",
            "input_name": "input",
            "output_name": "probabilities"
        },
        {
            "id": "logreg",
            "name": "Logistic Regression",
            "file": "logreg.onnx",
            "description": "Интерпретируемая линейная модель вероятностной оценки",
            "input_name": "input",
            "output_name": "probabilities"
        }
    ]
}

Path("results").mkdir(exist_ok=True)
with open("results/model_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("Манифест успешно создан: results/model_manifest.json")
