"""Локальные day02-окна; текущие общие эталоны и когорта остаются неизменными."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import COMPACT_DTYPES, ProcessedData
from pipeline.windows import window_mask

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "scoreboard/data"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def prepare(work, source=None, train=None):
    work = Path(work)
    out = work / "data"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        meta = json.loads((out / "manifest.json").read_text())
        if meta["board_manifest_sha256"] != sha(BOARD / "processed/manifest.json"):
            raise ValueError("Общие данные изменились после подготовки")
        return meta
    source = Path(source or ROOT / "data/processed/flows/train.csv").resolve()
    train = Path(train or ROOT / "runs/dc_sage/cache/train_flows.csv").resolve()
    board = json.loads((BOARD / "processed/manifest.json").read_text())
    source_meta = json.loads((source.parent.parent / "manifest.json").read_text())
    windows = {w: board["config"]["windows"][w] for w in ("train", "validation", "test")}
    source_bounds = source_meta["config"]["windows"][source.stem]
    if not all(source_bounds[0] <= a < b <= source_bounds[1] for a,b in windows.values()):
        raise ValueError("Источник не покрывает текущие окна дня 02")
    if source_meta["inputs"][0] != board["inputs"][0]:
        raise ValueError("Исходные архивы не совпадают")
    reused_train = train.exists()
    todo = ["validation", "test"] if reused_train else ["train", "validation", "test"]
    counts = {w: 0 for w in todo}
    events = {w: pd.Series(dtype=float) for w in todo}
    for i, chunk in enumerate(pd.read_csv(source, dtype={c: str for c in ("src", "dst", "src_port", "dst_port", "proto")}, chunksize=500_000)):
        for w in todo:
            part = chunk.loc[window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), windows[w], True)]
            if not len(part):
                continue
            path = out / f"{w}.csv"
            part.to_csv(path, mode="a" if counts[w] else "w", header=not counts[w], index=False)
            counts[w] += len(part)
            real = part[part.src != part.dst]
            events[w] = events[w].add(pd.concat([real.src, real.dst]).value_counts(), fill_value=0)
        if i % 20 == 0:
            print(f"подготовка: {(i+1)*500_000:,} строк; {counts}", flush=True)
    cohort = pd.read_csv(BOARD / "processed/cohort.csv", dtype={"host_id": str}).set_index("host_id")
    for w in todo:
        if counts[w] != board["windows"][w]["n_flows"]:
            raise ValueError(f"Число потоков {w} не совпало: {counts[w]}")
        np.testing.assert_array_equal(events[w].reindex(cohort.index, fill_value=0).to_numpy(), cohort[f"events_{w}"].to_numpy())
    if reused_train:
        # Старый кэш не считаем доказательством: проверяем и его реальные счётчики.
        n, active = 0, pd.Series(dtype=float)
        for chunk in pd.read_csv(train, usecols=["src", "dst", "ts", "duration"], dtype={"src": str,"dst":str}, chunksize=500_000):
            if not window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), windows["train"], True).all():
                raise ValueError("В переиспользуемом train есть поток за границей")
            n += len(chunk)
            real = chunk[chunk.src != chunk.dst]
            active = active.add(pd.concat([real.src, real.dst]).value_counts(), fill_value=0)
        if n != board["windows"]["train"]["n_flows"]:
            raise ValueError("Число потоков train не совпало")
        np.testing.assert_array_equal(active.reindex(cohort.index, fill_value=0).to_numpy(), cohort.events_train.to_numpy())
    paths = {w: str(train if w == "train" and reused_train else (out / f"{w}.csv").resolve()) for w in windows}
    meta = {"paths": paths, "windows": windows, "source": str(source),
            "board_manifest_sha256": sha(BOARD / "processed/manifest.json"),
            "flow_counts": {w: board["windows"][w]["n_flows"] for w in windows},
            "all_cohort_event_counts_verified": True, "available_test_windows": ["test"],
            "missing_test_windows": ["test_day03", "test_day04"]}
    dump(out / "manifest.json", meta)
    return meta


class LocalData(ProcessedData):
    def __init__(self, work):
        super().__init__(BOARD / "processed")
        self.local = json.loads((Path(work) / "data/manifest.json").read_text())
        if self.local["board_manifest_sha256"] != sha(BOARD / "processed/manifest.json"):
            raise ValueError("Общие окна или когорта изменились после подготовки")
        self._cache = {}

    def flows(self, window):
        if window not in self.local["paths"]:
            raise FileNotFoundError(f"Нет локальных потоков {window}")
        if window not in self._cache:
            path = self.local["paths"][window]
            f = pd.read_csv(path, dtype=COMPACT_DTYPES)
            categories = f.src.cat.categories.union(f.dst.cat.categories)
            f["src"] = f.src.cat.set_categories(categories)
            f["dst"] = f.dst.cat.set_categories(categories)
            self._cache[window] = f
        return self._cache[window]
