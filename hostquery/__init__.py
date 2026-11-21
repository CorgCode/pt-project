"""Запросы к хостам по сохранённым артефактам прогона.

Похожие хосты, группы, объяснение пары, изменения во времени и exploratory-аномалии —
без обучения моделей и без пересчёта метрик бенчмарка. Формы ответов совпадают с теми,
что уже потребляет лаборатория #156 (`noomo-ui/lab/core.js`), поэтому подключение
сводится к замене источника данных.

    python -m hostquery similar --bundle runs/lanl --method temporal_encoder --host Comp1 -k 10
"""
from hostquery.errors import HostQueryError
from hostquery.schema import SCHEMA

__all__ = ["Dataset", "HostQueryError", "RunStore", "SCHEMA", "open_dataset"]


def __getattr__(name):
    if name in {"Dataset", "RunStore", "open_dataset"}:
        from hostquery import store

        return getattr(store, name)
    raise AttributeError(name)
