"""Векторы MD-TGR для отправки в scoreboard.

Пишет те же файлы, что ждёт ``evaluation.submit``: ``embeddings.csv`` — векторы
train-окна для всей когорты, ``<окно>.csv`` — векторы оцениваемых хостов окна.
Веса блоков можно перебрать без переобучения, поэтому здесь же лежит подбор по
validation (``--tune``).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from methods.mdtgr.dataset import ShardSet, window_tensors
from methods.mdtgr.model import MDTGR

DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)


def load_checkpoint(path: Path) -> MDTGR:
    """Прочитать модель из чекпойнта (torch.load без pickle-объектов проекта)."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    return MDTGR.from_state(state["model"])


def window_vectors(shards: ShardSet, model: MDTGR, window: str,
                   weights: tuple[float, float, float] = DEFAULT_WEIGHTS, *,
                   rows: np.ndarray | None = None) -> np.ndarray:
    """Итоговые векторы окна: anchor ⊕ code ⊕ window с весами блоков."""
    model.eval()
    tensors = window_tensors(shards.shard(window))
    with torch.no_grad():
        blocks, code = model.encode(tensors)
        vectors = model.compose(tensors.anchor, code, blocks, weights)
        if rows is not None:
            vectors = vectors[torch.from_numpy(rows.astype(np.int64))]
    return vectors.numpy()


def write_vectors(path: Path, hosts: list[str], vectors: np.ndarray) -> Path:
    """CSV в формате отправки: host_id + emb_*; проверяет форму и конечность."""
    if vectors.shape[0] != len(hosts):
        raise ValueError(f"{path}: {vectors.shape[0]} векторов на {len(hosts)} хостов")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{path}: в векторах есть NaN/inf")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(vectors.astype(np.float64), index=pd.Index(hosts, name="host_id"))
    frame.add_prefix("emb_").to_csv(path, float_format="%.6g")
    return path


def export(shards: ShardSet, model: MDTGR, out_dir: Path,
           weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
           windows: list[str] | None = None) -> dict[str, Path]:
    """Записать embeddings.csv (train, вся когорта) и по файлу на каждое окно."""
    out_dir = Path(out_dir)
    written: dict[str, Path] = {}
    cohort = shards.hosts
    written["train"] = write_vectors(out_dir / "embeddings.csv", cohort,
                                     window_vectors(shards, model, "train", weights))
    for window in windows or [w for w in shards.windows if w != "train"]:
        shard = shards.shard(window)
        hosts = shards.eval_hosts(window)
        rows = shard.pool()
        written[window] = write_vectors(out_dir / f"{window}.csv", hosts,
                                        window_vectors(shards, model, window, weights, rows=rows))
    return written


def tune_weights(shards: ShardSet, model: MDTGR, validation: str = "validation", *,
                 grid: list[float] | None = None, anchor_grid: list[float] | None = None,
                 window_grid: list[float] | None = None, work_dir: Path | None = None) -> dict:
    """Перебор весов блоков на validation: возвращает лучшие веса и таблицу.

    Оценка — официальная ``evaluation.submit.validation_score``: те же метрики и та
    же формула балла, что на test, но на validation-окне. Test-дни не участвуют.
    ``embeddings.csv`` пересобирается с теми же весами, что и validation.csv: Identity
    сравнивает именно эти два файла, и веса должны быть общими.
    """
    from evaluation import submit

    grid = grid or [1.0, 2.0, 3.0]
    anchor_grid = anchor_grid or [0.5, 1.0, 1.5]
    window_grid = window_grid or [0.5, 1.0, 1.5]

    model.eval()
    with torch.no_grad():
        val_tensors = window_tensors(shards.shard(validation))
        blocks, code = model.encode(val_tensors)
        anchor = val_tensors.anchor
        train_tensors = window_tensors(shards.shard("train"))
        train_blocks, train_code = model.encode(train_tensors)
        train_anchor = train_tensors.anchor

    rows = torch.from_numpy(shards.shard(validation).pool().astype(np.int64))
    tmp = Path(work_dir) if work_dir is not None else Path(shards.root) / "_tune"
    tmp.mkdir(parents=True, exist_ok=True)
    hosts = shards.eval_hosts(validation)
    table = []
    best = {"score": -1.0, "weights": DEFAULT_WEIGHTS, "tasks": None}
    for wa, wc, ww in itertools.product(anchor_grid, grid, window_grid):
        weights = (wa, wc, ww)
        write_vectors(tmp / "embeddings.csv", shards.hosts,
                      model.compose(train_anchor, train_code, train_blocks, weights).numpy())
        vectors = model.compose(anchor, code, blocks, weights)[rows].numpy()
        write_vectors(tmp / f"{validation}.csv", hosts, vectors)
        score, tasks = submit.validation_score(tmp / "embeddings.csv", tmp / f"{validation}.csv")
        table.append({"weights": list(weights), "score": score, "tasks": tasks})
        if score > best["score"]:
            best = {"score": score, "weights": weights, "tasks": tasks}
    (tmp / "weights.json").write_text(json.dumps({"best": best, "table": table}, indent=2,
                                                 ensure_ascii=False) + "\n", encoding="utf-8")
    return {"best": best, "table": table}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Векторы MD-TGR")
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default="1,1,1", help="веса anchor,code,window")
    ap.add_argument("--windows", help="окна через запятую (по умолчанию все, кроме train)")
    args = ap.parse_args(argv)
    shards = ShardSet(args.shards)
    model = load_checkpoint(args.checkpoint)
    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        raise SystemExit("--weights: нужно три числа через запятую")
    windows = [w.strip() for w in args.windows.split(",")] if args.windows else None
    written = export(shards, model, args.out, weights, windows)
    print(json.dumps({k: str(v) for k, v in written.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
