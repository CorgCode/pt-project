"""
build_edges.py — превращает сетевые события LANL в рёбра графа общения.

Вход: CSV без заголовка, колонки по порядку:
    Time, Duration, SrcDevice, DstDevice, Protocol, SrcPort, DstPort,
    SrcPackets, DstPackets, SrcBytes, DstBytes

Выход: edges.csv с колонками
    src, dst, dst_port, day, count

Правила:
- рёбра ненаправленные: пара сортируется, src <= dst;
- SrcPort не используется (это случайный клиентский порт);
- day = Time // 86400 (Time — секунды от начала датасета);
- count = число событий пары за день на данном DstPort;
- dst_port всегда пишется в файл, а учитывать его или нет решает метрика
  через конфиг (при выключенном учёте строки суммируются по портам).

Использование:
    python build_edges.py out/edges.csv flows/netflow_day-*.csv
    python build_edges.py out/edges.csv flows/*.csv --start-day 1 --end-day 30
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

SECONDS_PER_DAY = 86400
EXPECTED_COLUMNS = 11

# Индексы колонок во входном CSV
TIME_IDX = 0
SRC_IDX = 2
DST_IDX = 3
DST_PORT_IDX = 6

EDGE_HEADER = ["src", "dst", "dst_port", "day", "count"]


def iter_rows(paths: Iterable[Path]):
    """Потоково читает строки из всех входных файлов."""
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            yield from csv.reader(f)


def build_edge_counts(
    paths: Iterable[Path],
    start_day: Optional[int] = None,
    end_day: Optional[int] = None,
) -> tuple[Counter, dict]:
    """
    Считает события по ключу (day, src, dst, dst_port).

    Возвращает счётчики и статистику: сколько строк прочитано,
    сколько пропущено и почему.
    """
    counts: Counter = Counter()
    stats = {"read": 0, "bad_format": 0, "out_of_period": 0, "self_loop_or_empty": 0}

    for row in iter_rows(paths):
        stats["read"] += 1

        if len(row) < EXPECTED_COLUMNS:
            stats["bad_format"] += 1
            continue

        try:
            day = int(row[TIME_IDX]) // SECONDS_PER_DAY
        except ValueError:
            # Сюда попадает и случайная строка заголовка, если она есть
            stats["bad_format"] += 1
            continue

        if (start_day is not None and day < start_day) or (
            end_day is not None and day > end_day
        ):
            stats["out_of_period"] += 1
            continue

        src = row[SRC_IDX].strip()
        dst = row[DST_IDX].strip()
        if not src or not dst or src == dst:
            stats["self_loop_or_empty"] += 1
            continue

        # Ненаправленное ребро: порядок внутри пары фиксируем
        a, b = sorted((src, dst))
        dst_port = row[DST_PORT_IDX].strip()
        counts[(day, a, b, dst_port)] += 1

    return counts, stats


def write_edges(counts: Counter, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EDGE_HEADER)
        for (day, a, b, port), count in sorted(counts.items()):
            writer.writerow([a, b, port, day, count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение edges.csv из сетевых событий LANL")
    parser.add_argument("out", type=Path, help="путь к выходному edges.csv")
    parser.add_argument("inputs", nargs="+", type=Path, help="входные CSV без заголовка")
    parser.add_argument("--start-day", type=int, default=None, help="первый день включительно")
    parser.add_argument("--end-day", type=int, default=None, help="последний день включительно")
    args = parser.parse_args()

    counts, stats = build_edge_counts(args.inputs, args.start_day, args.end_day)
    write_edges(counts, args.out)

    print(f"Строк прочитано: {stats['read']}", file=sys.stderr)
    print(f"Пропущено (формат): {stats['bad_format']}", file=sys.stderr)
    print(f"Вне периода: {stats['out_of_period']}", file=sys.stderr)
    print(f"Пропущено (пусто или self-loop): {stats['self_loop_or_empty']}", file=sys.stderr)
    print(f"Уникальных ключей (день, пара, порт): {len(counts)}", file=sys.stderr)
    print(f"Записано: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
