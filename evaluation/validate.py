"""Проверка формата эмбеддингов перед оценкой.

    python -m evaluation.validate submissions/my_method/embeddings.csv
    python -m evaluation.validate --manifest submission_manifest.csv

Файл эмбеддинга: host_id, emb_0, emb_1, ... — host_id уникален и не пуст (это строка:
"001" и "1" — разные хосты), координаты — конечные числа.
Манифест нескольких окон: method, window_id, path. Пара (method, window_id) уникальна,
а в одном окне у всех методов одинаковый набор хостов.
Проверяется только формат, а не то, на каких данных метод обучался.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(path: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"файл не найден: {file_path}")
    df = pd.read_csv(file_path, dtype={"host_id": "string"})
    if "host_id" not in df.columns:
        raise ValueError("нет колонки host_id")
    cols = [c for c in df.columns if c.startswith("emb_")]
    if not cols:
        raise ValueError("нет колонок эмбеддинга (emb_0, emb_1, ...)")
    if df["host_id"].isna().any() or df["host_id"].str.strip().eq("").any():
        raise ValueError("в host_id есть пустые значения")
    if df["host_id"].duplicated().any():
        raise ValueError(f"повторяющиеся host_id: {df.loc[df['host_id'].duplicated(), 'host_id'].tolist()[:5]}")
    if df[cols].isna().any().any():
        raise ValueError("в эмбеддинге есть пропуски (NaN)")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"нечисловые колонки: {non_numeric}")
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise ValueError("координаты должны быть конечными числами (без Inf и NaN)")
    print(f"OK: {len(df)} хостов, размерность {len(cols)}")


def validate_manifest(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, dtype="string")
    required = {"method", "window_id", "path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"в манифесте нет колонок: {sorted(required - set(manifest.columns))}")
    if manifest.empty or manifest[list(required)].isna().any().any():
        raise ValueError("манифест пуст или в нём пустые поля")
    for column in required:
        if manifest[column].str.strip().eq("").any():
            raise ValueError(f"пустое поле: {column}")
    if manifest.duplicated(["method", "window_id"]).any():
        raise ValueError("повтор пары (method, window_id)")
    cohorts: dict[str, set[str]] = {}
    for row in manifest.itertuples(index=False):
        file = (path.parent / row.path).resolve()
        validate_submission(str(file))
        current = set(pd.read_csv(file, dtype={"host_id": "string"})["host_id"].astype(str))
        if row.window_id in cohorts and current != cohorts[row.window_id]:
            raise ValueError(f"окно {row.window_id}: у методов разные наборы хостов; "
                             f"нет: {sorted(cohorts[row.window_id] - current)[:5]}, "
                             f"лишние: {sorted(current - cohorts[row.window_id])[:5]}")
        cohorts[row.window_id] = current
    print(f"OK: {len(manifest)} файлов, окон: {len(cohorts)}")
    return len(manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", nargs="?", type=Path)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    if (args.submission is None) == (args.manifest is None):
        ap.error("укажите либо файл эмбеддинга, либо --manifest")
    if args.manifest:
        validate_manifest(args.manifest)
    else:
        validate_submission(str(args.submission))


if __name__ == "__main__":
    main()
