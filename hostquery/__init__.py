"""Запросы к хостам по сохранённым артефактам прогона.

Похожие хосты, группы, объяснение пары, изменения во времени и exploratory-аномалии —
без обучения моделей и без пересчёта метрик бенчмарка. Формы ответов совпадают с теми,
что уже потребляет лаборатория #156 (`noomo-ui/lab/core.js`), поэтому подключение
сводится к замене источника данных.

    python -m hostquery similar --bundle runs/lanl --method temporal_encoder --host Comp1 -k 10
"""
from hostquery.errors import HostQueryError
from hostquery.schema import SCHEMA
from hostquery.store import Dataset, RunStore, open_dataset

__all__ = ["Dataset", "HostQueryError", "RunStore", "SCHEMA", "open_dataset"]
