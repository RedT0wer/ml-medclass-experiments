from pathlib import Path

from src.dataset import DatasetBuilder


def main() -> None:
    features_file = "data/raw/features.csv"
    value_features_file = "data/raw/value_features.csv"
    data_file = "data/raw/algorithm.txt"

    target_name = "diagnosis"
    output_csv = "data/processed/dataset.csv"
    output_schema = "data/processed/dataset.schema.json"

    excluded_features = [
        "Ds",
        "Corse",
        "effect of treat",
    ]

    use_value_labels = True

    builder = DatasetBuilder(
        features_file=features_file,
        value_features_file=value_features_file,
        data_file=data_file,
        target_name=target_name,
    )

    builder.save_dataset(
        filepath=output_csv,
        schema_path=output_schema,
        excluded_features=excluded_features,
        use_value_labels=use_value_labels,
        sep=";",
        encoding="utf-8",
    )

    print(f"[OK] Датасет сохранен: {output_csv}")
    print(f"[OK] Схема сохранена:  {output_schema}")


if __name__ == "__main__":
    main()