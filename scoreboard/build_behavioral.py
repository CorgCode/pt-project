"""Эталонные поведенческие признаки (`behavioral` из methods/baselines/temporal.py) -> runs/blocks_ref/<окно>.npz.

    PT_COMPACT=1 python scoreboard/build_behavioral.py                  # данные: runs/processed (или PT_DATA)

Обучение на train.
Выход: по файлу на окно: hosts (порядок runs/processed/cohort.csv) и behav (хостов x 57).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.baselines.temporal import BehavioralBaseline  # noqa: E402
from pipeline.data import ProcessedData                    # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("PT_DATA", str(ROOT / "runs" / "processed")))
    ap.add_argument("--out", default=str(ROOT / "runs" / "blocks_ref"))
    a = ap.parse_args()
    t0 = time.time()

    order = list(pd.read_csv(Path(a.data) / "cohort.csv", dtype={"host_id": str})["host_id"])
    data = ProcessedData(a.data)
    if set(order) != set(data.cohort):
        raise SystemExit("когорта в cohort.csv и в ProcessedData не совпадает")
    pos = pd.Index(order)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log("обучение behavioral на train ...")
    model = BehavioralBaseline().fit(data)
    windows = ["train"] + (["validation"] if "validation" in data.windows else []) + list(data.test_windows)
    rows = []
    for w in windows:
        hosts = list(data.cohort) if w == "train" else list(data.evaluable(w))
        emb = model.embed(data, w, hosts).to_numpy()
        arr = np.zeros((len(order), emb.shape[1]), dtype=np.float32)
        arr[pos.get_indexer(hosts)] = emb
        np.savez_compressed(out / f"{w}.npz", hosts=np.array(order, dtype=str), behav=arr)
        rows.append((w, len(hosts), emb.shape[1], float(np.isfinite(emb).all())))
        log(f"  [{w}] {len(hosts):,} хостов, {emb.shape[1]} признаков ({time.time() - t0:.0f} с)")

    bar = "=" * 14
    print(f"\n{bar} КРАТКИЙ ИТОГ {bar}")
    for w, n, d, ok in rows:
        print(f"  {w:11s} хостов {n:>6,}  признаков {d}  без NaN/inf: {'да' if ok else 'НЕТ'}")
    print(f"Файлы: {out}/<окно>.npz; всего {time.time() - t0:.0f} с")
    print("=" * 83)


if __name__ == "__main__":
    main()
