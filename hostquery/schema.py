"""Версия публичного контракта и общий конверт ответа.

Конверт один для CLI и HTTP: query + provenance + (items|dataset|groups|pair|changes)
+ warnings. Версия меняется, только если ломается форма; добавление поля — нет.
"""
from __future__ import annotations

import json

SCHEMA = "hostquery/1"

# Ограничения браузерного примера из noomo-ui/lab/core.js. Сервер может больше,
# но CLI с CSV держит те же границы, чтобы числа совпадали с лабораторией.
MAX_CSV_HOSTS = 500
MAX_CSV_DIMENSIONS = 1024
MAX_CSV_ROWS = 10_000
MAX_CSV_BYTES = 3 * 1024 * 1024


def envelope(query_kind: str, *, provenance: dict, warnings: list[str] | None = None, **payload) -> dict:
    """Единый ответ: сначала запрос и происхождение, потом данные, в конце оговорки."""
    out = {"schema": SCHEMA, "query": {"kind": query_kind}}
    out["query"].update(payload.pop("query", {}))
    out["provenance"] = provenance
    out.update(payload)
    out["warnings"] = list(warnings or [])
    return out


def dumps(obj: dict, pretty: bool = False) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)
