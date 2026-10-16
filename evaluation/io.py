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
        raise ValueError(f"{path}: missing required column {id_col!r}")
    if df[id_col].isna().any() or df[id_col].str.strip().eq("").any():
        raise ValueError(f"{path}: missing or blank {id_col} values")
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicate {id_col} values")
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"{path}: no columns with prefix {prefix!r}")
    matrix = df.set_index(id_col)[cols]
    if matrix.isna().any().any():
        raise ValueError(f"{path}: NaN values are not allowed")
    try:
        coords = matrix.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: columns must be numeric") from exc
    if not np.isfinite(coords).all():
        raise ValueError(f"{path}: coordinates must be finite")
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
        raise ReferenceError(f"{path}: missing columns {sorted(missing)}")
    if df.empty:
        raise ReferenceError(f"{path}: empty")
    for col in required:
        if df[col].str.strip().eq("").any():
            raise ReferenceError(f"{path}: blank values in {col!r}")
    if split_values is not None and "split" in df.columns:
        bad = sorted(set(df["split"]) - split_values)
        if bad:
            raise ReferenceError(f"{path}: unknown split values {bad}; expected {sorted(split_values)}")
    return df


def select_split(df: pd.DataFrame, split: str, name: str) -> pd.DataFrame:
    out = df[df["split"] == split]
    if out.empty:
        raise ReferenceError(f"{name}: no rows with split={split!r}")
    return out


def cosine_pairs(emb: pd.DataFrame, a: list[str], b: list[str]) -> np.ndarray:
    """Косинус для пар строк по host_id. Неизвестный хост или нулевой вектор — ошибка."""
    unknown = sorted((set(a) | set(b)) - set(emb.index))
    if unknown:
        raise ReferenceError(f"embedding lacks hosts used by the reference: {unknown[:5]}")
    x = emb.to_numpy(dtype=float)
    norms = np.linalg.norm(x, axis=1)
    if (norms == 0).any():
        raise ReferenceError("zero-norm host vectors make cosine undefined")
    pos = {h: i for i, h in enumerate(emb.index)}
    ia = np.fromiter((pos[h] for h in a), dtype=int, count=len(a))
    ib = np.fromiter((pos[h] for h in b), dtype=int, count=len(b))
    return np.einsum("ij,ij->i", x[ia], x[ib]) / (norms[ia] * norms[ib])
