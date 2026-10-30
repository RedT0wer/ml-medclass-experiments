"""Интерфейс командной строки для экспериментов по классификации медицинских данных."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabulate import tabulate

from src.dataset.dataset_builder import DatasetBuilder
from src.dataset.loader import load_dataset
from src.export.exporter import export_and_validate, verify_onnx_file
from src.metrics.registry import calculate_metrics, list_metrics
from src.models.registry import (
    MODELS_REGISTRY,
    get_model,
    list_models,
)
from src.tuning.tuner import HyperparameterTuner, prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def _get_default_dataset_path() -> str:
    candidates = [
        "data/processed/dataset.parquet",
        "data/processed/dataset.csv",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return "data/processed/dataset.parquet"


def _format_table(data: List[Dict[str, Any]], tablefmt: str = "github") -> str:
    if not data:
        return ""
    headers = list(data[0].keys())
    rows = [[row.get(h, "") for h in headers] for row in data]
    return tabulate(rows, headers=headers, tablefmt=tablefmt)


CHEATSHEET = """
Примеры использования:
  python cli.py list
  python cli.py prep-data --output data/processed/dataset.parquet
  python cli.py train --model catboost --export-onnx results/catboost.onnx
  python cli.py benchmark --models catboost xgboost logreg --cv 5
  python cli.py tune --model catboost --cv 5
  python cli.py verify-onnx --onnx results/catboost.onnx
"""


def setup_cli_parser() -> argparse.ArgumentParser:
    default_dataset = _get_default_dataset_path()

    parser = argparse.ArgumentParser(
        description="Классификация клинических медицинских данных.",
        epilog=CHEATSHEET,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Команда")

    # 1. list
    subparsers.add_parser(
        "list",
        help="Список доступных моделей и метрик",
    )

    # 2. prep-data
    prep_parser = subparsers.add_parser(
        "prep-data",
        help="Сборка очищенного датасета из сырых данных",
    )
    prep_parser.add_argument(
        "--input",
        type=str,
        default="data/raw/algorithm.txt",
        help="Путь к файлу наблюдений",
    )
    prep_parser.add_argument(
        "--features",
        type=str,
        default="data/raw/features.csv",
        help="Путь к описанию признаков",
    )
    prep_parser.add_argument(
        "--value-features",
        type=str,
        default="data/raw/value_features.csv",
        help="Путь к описанию значений признаков",
    )
    prep_parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dataset.parquet",
        help="Путь для сохранения датасета (.parquet или .csv)",
    )
    prep_parser.add_argument(
        "--target",
        type=str,
        default="diagnosis",
        help="Целевая переменная",
    )

    # 3. train
    train_parser = subparsers.add_parser(
        "train",
        help="Обучение модели с экспортом в ONNX/Joblib и валидацией через ONNX Runtime",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="catboost",
        choices=list_models(),
        help="Имя модели",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Путь к датасету",
    )
    train_parser.add_argument(
        "--export-onnx",
        type=str,
        default=None,
        help="Путь для экспорта в ONNX (по умолчанию: results/<model>.onnx)",
    )
    train_parser.add_argument(
        "--export-joblib",
        type=str,
        default=None,
        help="Путь для экспорта в Joblib (по умолчанию: results/<model>.joblib)",
    )
    train_parser.add_argument(
        "--params-file",
        type=str,
        default=None,
        help="Путь к JSON-файлу гиперпараметров",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки для валидации",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    # 4. benchmark
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Кросс-валидация моделей и построение сводного отчета",
    )
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Путь к датасету",
    )
    bench_parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list_models(),
        help="Список моделей для сравнения",
    )
    bench_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["roc_auc", "f1_macro", "accuracy", "recall", "precision"],
        help="Список рассчитываемых метрик",
    )
    bench_parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Число фолдов кросс-валидации",
    )
    bench_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default="results/leaderboard.csv",
        help="Путь для сохранения лидерборда в CSV",
    )

    # 5. tune
    tune_parser = subparsers.add_parser(
        "tune",
        help="Подбор гиперпараметров по сетке с сохранением в results/",
    )
    tune_parser.add_argument(
        "--model",
        type=str,
        default="catboost",
        choices=list_models(),
        help="Имя модели",
    )
    tune_parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Путь к датасету",
    )
    tune_parser.add_argument(
        "--config",
        type=str,
        default="configs/tuning.yaml",
        help="Путь к YAML-файлу с сетками параметров",
    )
    tune_parser.add_argument(
        "--scoring",
        type=str,
        default="f1_macro",
        help="Метрика оптимизации",
    )
    tune_parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Число фолдов кросс-валидации",
    )
    tune_parser.add_argument(
        "--search-type",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Тип поиска",
    )
    tune_parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Число итераций для random search",
    )
    tune_parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Каталог сохранения результатов",
    )

    # 6. verify-onnx
    verify_parser = subparsers.add_parser(
        "verify-onnx",
        help="Автономная проверка инференса ONNX-файла и сверка с Joblib",
    )
    verify_parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Путь к файлу .onnx",
    )
    verify_parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Путь к датасету для тестового батча",
    )
    verify_parser.add_argument(
        "--joblib",
        type=str,
        default=None,
        help="Путь к связанной Joblib-модели для проверки эквивалентности",
    )
    verify_parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Порог абсолютной разности вероятностей (atol)",
    )

    return parser


def handle_list() -> None:
    print("\nЗарегистрированные модели:")
    models_data = []
    for name in list_models():
        reg = MODELS_REGISTRY[name]
        params_str = ", ".join(f"{k}={v}" for k, v in list(reg.default_params.items())[:3])
        if len(reg.default_params) > 3:
            params_str += ", ..."
        models_data.append({
            "Модель": name,
            "Класс": reg.model_class.__name__,
            "Параметры по умолчанию": params_str or "-",
        })
    print(_format_table(models_data))

    print("\nЗарегистрированные метрики:")
    prob_metrics = {"roc_auc", "pr_auc"}
    metrics_data = []
    for name in list_metrics():
        metrics_data.append({
            "Метрика": name,
            "Входные данные": "Вероятности (y_prob)" if name in prob_metrics else "Классы (y_pred)",
            "Многоклассовый режим": "OVR Macro" if name in prob_metrics or "macro" in name else "Да",
        })
    print(_format_table(metrics_data))
    print()


def handle_prep_data(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("Сборка датасета из: %s", args.input)

    builder = DatasetBuilder(
        features_file=args.features,
        value_features_file=args.value_features,
        data_file=args.input,
        target_name=args.target,
    )

    excluded = ["Ds", "Corse", "effect of treat"]
    builder.save_dataset(
        filepath=args.output,
        excluded_features=excluded,
        use_value_labels=True,
        sep=";",
    )

    elapsed = time.perf_counter() - t0
    schema_path = Path(args.output).with_suffix(".schema.json")
    logger.info("[OK] Датасет сохранен: %s (%.2fs)", args.output, elapsed)
    logger.info("[OK] Схема сохранена: %s", schema_path)


def handle_train(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("Обучение модели '%s' на датасете: %s", args.model, args.dataset)

    X, y, _ = load_dataset(args.dataset)
    X_proc, _ = prepare_features(X)
    y_arr = np.asarray(y)

    extra_params: Dict[str, Any] = {}
    if args.params_file is not None:
        p_path = Path(args.params_file)
        if p_path.exists():
            data = json.loads(p_path.read_text(encoding="utf-8"))
            extra_params = data.get("best_params", data)
            logger.info("Загружены параметры из %s: %s", p_path, extra_params)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc,
        y_arr,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_arr,
    )

    clf = get_model(args.model, **extra_params)
    t_train_start = time.perf_counter()
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t_train_start

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    eval_metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    print(f"\nРезультаты на тестовой выборке ({len(X_test)} объектов, обучение: {t_train:.2f}s):")
    metrics_table = [{"Метрика": k, "Значение": f"{v:.4f}"} for k, v in eval_metrics.items()]
    print(_format_table(metrics_table))

    onnx_target = args.export_onnx or f"results/{args.model}.onnx"
    joblib_target = args.export_joblib or f"results/{args.model}.joblib"

    logger.info("Экспорт модели в Joblib и ONNX...")
    artifacts = export_and_validate(
        model=clf,
        test_batch=X_test,
        onnx_path=onnx_target,
        joblib_path=joblib_target,
        atol=1e-5,
    )

    total_time = time.perf_counter() - t0
    print(f"\n[OK] Экспорт завершен ({total_time:.2f}s):")
    print(f"  Joblib: {artifacts['joblib']}")
    print(f"  ONNX:   {artifacts['onnx']} [ONNX Runtime: OK]\n")


def handle_benchmark(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("Запуск бенчмарка моделей на датасете: %s (фолдов=%d)", args.dataset, args.cv)

    X, y, _ = load_dataset(args.dataset)
    X_proc, _ = prepare_features(X)
    y_arr = np.asarray(y)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    requested_metrics = [m.lower().strip() for m in args.metrics]

    leaderboard_rows: List[Dict[str, Any]] = []

    for idx, model_name in enumerate(args.models, 1):
        t_model_start = time.perf_counter()
        logger.info("[%d/%d] Тестирование модели: %s", idx, len(args.models), model_name)

        fold_metrics: Dict[str, List[float]] = {m: [] for m in requested_metrics}

        for train_idx, val_idx in cv.split(X_proc, y_arr):
            X_tr, X_val = X_proc[train_idx], X_proc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            clf = get_model(model_name)
            clf.fit(X_tr, y_tr)

            y_pred = clf.predict(X_val)
            y_prob = clf.predict_proba(X_val) if hasattr(clf, "predict_proba") else None

            metrics_res = calculate_metrics(
                y_true=y_val,
                y_pred=y_pred,
                y_prob=y_prob,
                metric_names=requested_metrics,
            )

            for m in requested_metrics:
                fold_metrics[m].append(metrics_res.get(m, float("nan")))

        t_model = time.perf_counter() - t_model_start

        row: Dict[str, Any] = {"Модель": model_name}
        for m in requested_metrics:
            vals = fold_metrics[m]
            mean_val = float(np.nanmean(vals))
            std_val = float(np.nanstd(vals))
            row[f"{m} (mean±std)"] = f"{mean_val:.4f} ± {std_val:.3f}"
            row[f"_{m}_mean"] = mean_val

        row["Время (с)"] = f"{t_model:.2f}s"
        leaderboard_rows.append(row)

    sort_key = f"_{requested_metrics[0]}_mean"
    leaderboard_rows.sort(key=lambda r: r.get(sort_key, 0.0), reverse=True)

    display_rows = []
    for r in leaderboard_rows:
        display_rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    print(f"\nСравнение моделей (датасет: {Path(args.dataset).name}, cv={args.cv}):")
    print(_format_table(display_rows))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(display_rows).to_csv(out_path, index=False)

    total_time = time.perf_counter() - t0
    logger.info("[OK] Лидерборд сохранен: %s (%.2fs)\n", out_path, total_time)


def handle_tune(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("Подбор гиперпараметров для модели '%s' (датасет: %s)", args.model, args.dataset)

    X, y, _ = load_dataset(args.dataset)

    tuner = HyperparameterTuner(
        model=args.model,
        config_path=args.config,
        search_type=args.search_type,
        n_iter=args.n_iter,
        scoring=args.scoring,
        cv=args.cv,
    )
    result = tuner.tune(X, y)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"best_params_{args.model}.json"
    result.save(out_file)

    total_time = time.perf_counter() - t0
    print(f"\nРезультаты подбора гиперпараметров: {args.model} ({total_time:.2f}s)")
    print(f"  Метрика:          {args.scoring}")
    print(f"  Лучший результат: {result.best_score:.4f}")
    print("  Параметры:")
    for k, v in result.best_params.items():
        print(f"    {k}: {v}")
    print(f"  Артефакт:         {out_file}\n")


def handle_verify_onnx(args: argparse.Namespace) -> None:
    logger.info("Проверка ONNX модели: %s", args.onnx)

    rep = verify_onnx_file(
        onnx_path=args.onnx,
        dataset_path=args.dataset,
        joblib_path=args.joblib,
        atol=args.atol,
    )

    print("\nРезультаты проверки ONNX Runtime:")
    print(f"  Файл модели:        {rep['path']}")
    print(f"  Размер файла:       {rep['file_size_kb']:.1f} KB")
    print(f"  Время инференса:    {rep['latency_ms']:.2f} ms (батч: {rep['batch_size']} объектов)")

    print("\n  Входы:")
    for inp in rep["inputs"]:
        print(f"    - {inp['name']}: форма {inp['shape']}, тип {inp['type']}")

    print("  Выходы:")
    for out in rep["outputs"]:
        print(f"    - {out['name']}: форма {out['shape']}, тип {out['type']}")

    if rep["matched_joblib"]:
        print(f"\n  Сверка с Joblib:    {rep['matched_joblib']}")
        print(f"  Макс. расхождение:  {rep['max_diff']:.2e} (порог atol={args.atol:.1e})")
        equiv_badge = "[OK]" if rep["is_equivalent"] else "[ERROR]"
        print(f"  Эквивалентность:    {equiv_badge}")

    status_badge = "[OK]" if rep["status"] in ("OK", "PASS") else "[ERROR]"
    print(f"\n  Статус проверки:    {status_badge}\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = setup_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        handle_list()
    elif args.command == "prep-data":
        handle_prep_data(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "tune":
        handle_tune(args)
    elif args.command == "verify-onnx":
        handle_verify_onnx(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
