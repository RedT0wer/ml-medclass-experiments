from dataclasses import dataclass
from typing import List, Optional

DEFAULT_MISSING_VALUE = "MISSING"


@dataclass(frozen=True)
class MissingValueConfig:
    """Единая правда о представлении пропусков в CSV и в JSON-схеме.

    Attributes:
        value:
            Строка-маркер пропуска. Пишется в ячейки CSV и (опционально)
            в ``categories`` схемы. Не используйте строки, которые pandas
            по умолчанию трактует как NaN ("NA", "N/A", "null", "None"),
            иначе read_csv съест их раньше, чем CategoricalDtype.
        include_in_categories:
            * ``True`` (по умолчанию) — пропуск пишется в CSV как ``value``
              и добавляется в ``categories`` схемы; после ``DatasetStore.load``
              ячейка будет именно этой категорией.
            * ``False`` — пропуск пишется как пустая ячейка (NaN) и в
              ``categories`` схемы отсутствует; после загрузки — NaN.
    """
    value: str = DEFAULT_MISSING_VALUE
    include_in_categories: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("MissingValueConfig.value должен быть непустой строкой")


    def cell_value(self) -> Optional[str]:
        """Что записывать в ячейку CSV при пропуске (None → NaN/пусто)."""
        return self.value if self.include_in_categories else None

    def categories_suffix(self) -> List[str]:
        """Хвост, добавляемый к ``categories`` в схеме."""
        return [self.value] if self.include_in_categories else []