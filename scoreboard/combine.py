"""Гибрид: склеить эмбеддинги нескольких методов из таблицы в один.

    python scoreboard/combine.py behavioral_pca node2vec --name hybrid_behavioral_node2vec

Каждая часть нормируется (среднее 0, общая норма 1 на хост), чтобы ни одна не перевешивала из-за масштаба,
затем векторы склеиваются. Берутся train (embeddings.csv) и, если есть у всех частей, test (test.csv).
Результат — обычная отправка в scoreboard/submissions/<name>/, она сразу оценивается.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from evaluation.io import load_embeddings  # noqa: E402


def _block(x: pd.DataFrame) -> pd.DataFrame:
    x = x - x.mean()
    norm = np.sqrt((x.to_numpy() ** 2).sum(axis=1).mean())
    return x / (norm if norm > 0 else 1.0)


def combine(parts: list[str], name: str, subs: Path | None = None) -> Path:
    subs = Path(subs or submit.SUBMISSIONS)
    out = subs / name
    out.mkdir(parents=True, exist_ok=True)
    for fname in ("embeddings.csv", "test.csv"):
        files = [subs / p / fname for p in parts]
        if not all(f.exists() for f in files):
            continue
        blocks = [_block(load_embeddings(f)) for f in files]
        hosts = sorted(set.intersection(*(set(b.index) for b in blocks)))
        x = np.hstack([b.loc[hosts].to_numpy() for b in blocks])
        pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(out / fname, float_format="%.7g")
    (out / "SHARED_COORDINATES").unlink(missing_ok=True)       # части могут быть в разных координатах
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+", help="имена методов из scoreboard/submissions")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out = combine(a.parts, a.name)
    r = submit.score(out)
    submit.render()
    print(f"{a.name}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
