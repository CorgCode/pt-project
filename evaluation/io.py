"""Строгая загрузка файлов, общая для всех метрик (форматы — evaluation/README.md)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")


class ReferenceError(ValueError):
    """Файл эталона нарушает формат; в сообщении — файл и правило."""


def load_embeddings(path: Path, id_col: str = "host_id", prefix: str = "emb_") -> pd.DataFrame:
    """Матрица с host_id в индексе. Ошибка на пустых и повторных ID, NaN, нечисловых значениях, Inf."""
    df = pd.read_csv(path, dtype={id_col: "string"})
    if id_col not in df.columns:
        raise ValueError(f"{path}: нет обязательной колонки {id_col!r}")
    if df[id_col].isna().any() or df[id_col].str.strip().eq("").any():
        raise ValueError(f"{path}: пустые значения {id_col}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: повторяющиеся {id_col}")
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"{path}: нет колонок с префиксом {prefix!r}")
    matrix = df.set_index(id_col)[cols]
    if matrix.isna().any().any():
        raise ValueError(f"{path}: пропуски (NaN) недопустимы")
    try:
        coords = matrix.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: колонки должны быть числовыми") from exc
    if not np.isfinite(coords).all():
        raise ValueError(f"{path}: координаты должны быть конечными числами")
    matrix.index = matrix.index.astype(str)
    return matrix


def read_reference_table(path: Path, required: set[str], *, split_values: set[str] | None = None,
                         string_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Прочитать CSV эталона как строки и проверить колонки, пустоту, пустые ячейки."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = required - set(df.columns)
    if missing:
        raise ReferenceError(f"{path}: нет колонок {sorted(missing)}")
    if df.empty:
        raise ReferenceError(f"{path}: empty")
    for col in required:
        if df[col].str.strip().eq("").any():
            raise ReferenceError(f"{path}: пустые значения в {col!r}")
    if split_values is not None and "split" in df.columns:
        bad = sorted(set(df["split"]) - split_values)
        if bad:
            raise ReferenceError(f"{path}: неизвестные split {bad}; ожидались {sorted(split_values)}")
    return df


def select_split(df: pd.DataFrame, split: str, name: str) -> pd.DataFrame:
    out = df[df["split"] == split]
    if out.empty:
        raise ReferenceError(f"{name}: нет строк со split={split!r}")
    return out


def cosine_pairs(emb: pd.DataFrame, a: list[str], b: list[str]) -> np.ndarray:
    """Косинус для пар строк по host_id. Неизвестный хост или нулевой вектор — ошибка."""
    unknown = sorted((set(a) | set(b)) - set(emb.index))
    if unknown:
        raise ReferenceError(f"в эмбеддинге нет хостов из эталона: {unknown[:5]}")
    x = emb.to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1)
    if (norms == 0).any():
        raise ReferenceError("нулевой вектор хоста: косинус не определён")
    pos = {h: i for i, h in enumerate(emb.index)}
    ia = np.fromiter((pos[h] for h in a), dtype=int, count=len(a))
    ib = np.fromiter((pos[h] for h in b), dtype=int, count=len(b))
    return np.einsum("ij,ij->i", x[ia], x[ib]) / (norms[ia] * norms[ib])
