"""Признаки трафика и их имена — один источник правды для сервера и лаборатории.

Первые восемь — простые показатели активности. Следующие двенадцать — хешированные
блоки сервисов (proto:port mod 6): это НЕ протоколы и не роли, поэтому в ответах они
помечены `hashed_service_bucket`. Имена совпадают с `noomo-ui/lab/lab.js`.
"""
from __future__ import annotations

BASE_FEATURES = [
    "Исходящие записи",
    "Входящие записи",
    "Байты отправлено",
    "Байты получено",
    "Пакеты отправлено",
    "Пакеты получено",
    "Разных контактов",
    "Всего записей",
]
SERVICE_FEATURES = [f"Сервисы · исходящие {i}" for i in range(1, 7)] + \
                   [f"Сервисы · входящие {i}" for i in range(1, 7)]
FEATURE_NAMES = BASE_FEATURES + SERVICE_FEATURES
FEATURE_KINDS = ["base"] * len(BASE_FEATURES) + ["hashed_service_bucket"] * len(SERVICE_FEATURES)
N_BASE, N_SERVICE = len(BASE_FEATURES), len(SERVICE_FEATURES)
N_BUCKETS = N_SERVICE // 2
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 20 and N_SERVICE == 2 * N_BUCKETS

EMBEDDING_COORDINATE = "embedding_coordinate"


def service_bucket(proto: object, port: object) -> int:
    """Тот же хеш, что в core.js: 31-битное переполнение и mod 6."""
    text = f"{proto or ''}:{port or ''}"
    value = 0
    for ch in text:
        value = (value * 31 + ord(ch)) & 0xFFFFFFFF
    return value % 6


def flow_names() -> list[str | None]:
    """Имена 20 признаков трафика."""
    return list(FEATURE_NAMES)


def flow_kinds() -> list[str]:
    return list(FEATURE_KINDS)


def embedding_names(count: int) -> list[str | None]:
    """У координат эмбеддинга имён нет: размерность не говорит о смысле оси."""
    return [None] * count


def embedding_kinds(count: int) -> list[str]:
    return [EMBEDDING_COORDINATE] * count


def describe_flow() -> list[dict]:
    return [{"index": i, "name": name, "kind": kind}
            for i, (name, kind) in enumerate(zip(FEATURE_NAMES, FEATURE_KINDS))]


def describe_embedding(count: int) -> list[dict]:
    return [{"index": i, "name": None, "kind": EMBEDDING_COORDINATE} for i in range(count)]
