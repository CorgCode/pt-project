"""Гибрид: behavioral ×2 + кто собеседники + какие сервисы (TF-IDF → SVD).

    PT_DATA=/путь/к/lanl/processed python scoreboard/hybrid_tfidf.py           # обучить с выбранными параметрами
    PT_DATA=/путь/к/lanl/processed python scoreboard/hybrid_tfidf.py --tune    # перебрать DIM и веса на validation

Части:
  behavioral — профиль трафика (methods.baselines.temporal), вес 2;
  peers      — хост × собеседник с направлением («o:Comp1» — ходит к нему, «i:Comp1» — к нему ходит);
  services   — хост × сервис («u:6/443» — пользуется, «s:6/443» — обслуживает).
Для peers и services: log1p числа потоков → TF-IDF → нормировка строки → TruncatedSVD 128.
Словарь, IDF и SVD подбираются на train и применяются к любому окну — одна система координат.
Веса частей и размерность выбраны на validation (test не смотрели): --tune перебирает сетку GRID
и учитывает все четыре задачи, включая узнаваемость (по эмбеддингу validation-окна).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from methods.baselines.temporal import BehavioralBaseline  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402

NAME = "hybrid_tfidf"
DIM = 128
WEIGHTS = {"behavioral": 2.0, "peers": 1.0, "services": 1.0}


def tokens(flows: pd.DataFrame, kind: str) -> pd.DataFrame:
    src, dst = flows["src"].astype(str), flows["dst"].astype(str)
    if kind == "peers":
        out, inn = "o:" + dst, "i:" + src
    else:
        svc = flows["proto"].astype(str) + "/" + flows["dst_port"].astype(str)
        out, inn = "u:" + svc, "s:" + svc
    return pd.concat([pd.DataFrame({"h": src, "t": out}), pd.DataFrame({"h": dst, "t": inn})], ignore_index=True)


def counts(flows: pd.DataFrame, kind: str, hosts: list[str], vocab: pd.Index) -> sp.csr_matrix:
    t = tokens(flows, kind)
    g = t[t["h"].isin(set(hosts)) & t["t"].isin(vocab)].groupby(["h", "t"]).size()
    rows = pd.Index(hosts).get_indexer(g.index.get_level_values(0))
    cols = vocab.get_indexer(g.index.get_level_values(1))
    return sp.csr_matrix((np.log1p(g.to_numpy(float)), (rows, cols)), shape=(len(hosts), len(vocab)))


class TfidfSVD:
    def __init__(self, kind: str, dim: int = DIM, min_df: int = 2, seed: int = 0):
        self.kind, self.dim, self.min_df, self.seed = kind, dim, min_df, seed

    def fit(self, flows: pd.DataFrame, hosts: list[str]) -> "TfidfSVD":
        t = tokens(flows, self.kind)
        df = t[t["h"].isin(set(hosts))].drop_duplicates().groupby("t").size()
        self.vocab = pd.Index(df[df >= self.min_df].index)     # токен хотя бы у двух хостов
        x = counts(flows, self.kind, hosts, self.vocab)
        self.tfidf = TfidfTransformer().fit(x)
        dim = min(self.dim, len(self.vocab) - 1)                 # на маленьких данных словарь бывает короче DIM
        self.svd = TruncatedSVD(dim, random_state=self.seed).fit(normalize(self.tfidf.transform(x)))
        return self

    def embed(self, flows: pd.DataFrame, hosts: list[str]) -> np.ndarray:
        z = self.svd.transform(normalize(self.tfidf.transform(counts(flows, self.kind, hosts, self.vocab))))
        zero = np.linalg.norm(z, axis=1) == 0                  # хост без токенов: малый шум вместо нуля
        z[zero] = np.random.default_rng(self.seed).standard_normal((zero.sum(), z.shape[1])) * 1e-4
        return z


def _block(x: np.ndarray) -> np.ndarray:
    """Среднее 0, средняя норма строки 1 — как в scoreboard/combine.py."""
    x = x - x.mean(0)
    return x / np.sqrt((x ** 2).sum(1).mean())


GRID = {"dim": (64, 128, 256), "behavioral": (1.0, 2.0, 3.0, 4.0)}


class HybridTfidf:
    """fit(data) учится только на train; embed(data, window, hosts) — как у бейзлайнов (для experiments.robustness)."""
    name = NAME
    shared_coordinate_system = True

    def __init__(self, seed: int = 0, dim: int = DIM, weights: dict | None = None):
        self.seed, self.dim, self.weights = seed, dim, dict(weights or WEIGHTS)

    def fit(self, data: ProcessedData) -> "HybridTfidf":
        self.train = data.training_flows()
        self.beh = BehavioralBaseline().fit(data)
        self.graph = {k: TfidfSVD(k, dim=self.dim, seed=self.seed).fit(self.train, data.cohort)
                      for k in ("peers", "services")}
        return self

    def embed(self, data: ProcessedData, window: str, hosts: list[str] | None = None) -> pd.DataFrame:
        hosts = list(hosts) if hosts is not None else data.cohort
        flows = self.train if window == "train" else data.flows(window)
        parts = {"behavioral": self.beh.embed(data, window, hosts).to_numpy()}
        parts.update({k: m.embed(flows, hosts) for k, m in self.graph.items()})
        x = np.hstack([_block(parts[k]) * w for k, w in self.weights.items()])
        return pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_")


def _write(model: HybridTfidf, data: ProcessedData, out: Path, validation: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    submit.keep_vectors_local(out)
    jobs = [("train", data.cohort, "embeddings.csv")] + [(w, data.evaluable(w), submit.window_file(w))
                                                          for w in data.test_windows]
    if validation and "validation" in data.windows:          # только для подбора; в git не кладётся
        jobs.append(("validation", data.evaluable("validation"), "validation.csv"))
    for window, hosts, fname in jobs:
        model.embed(data, window, hosts).to_csv(out / fname, float_format="%.5g")


def tune(data: ProcessedData) -> dict:
    """Перебор GRID на validation (test не используется); печатает таблицу, возвращает лучшие параметры."""
    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for dim in GRID["dim"]:
            fitted = HybridTfidf(dim=dim).fit(data)                  # части не зависят от весов — учим один раз
            for wb in GRID["behavioral"]:
                fitted.weights = {**WEIGHTS, "behavioral": wb}
                d = Path(tmp) / f"{dim}_{wb}"
                _write(fitted, data, d)
                score, tasks = submit.validation_score(d / "embeddings.csv", d / "validation.csv")
                rows.append({"dim": dim, "behavioral": wb, "score": score, **tasks})
                print(f"  DIM {dim:4d}, behavioral ×{wb:g}: {score:5.1f}  "
                      + ", ".join(f"{k} {v:.0f}" for k, v in tasks.items()), flush=True)
    best = max(rows, key=lambda r: r["score"])
    print(f"лучшее на validation: DIM {best['dim']}, behavioral ×{best['behavioral']:g} ({best['score']:.1f})")
    return {"dim": best["dim"], "weights": {**WEIGHTS, "behavioral": best["behavioral"]}}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true", help="подобрать DIM и вес поведения на validation, потом обучить")
    a = ap.parse_args()
    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    params = tune(data) if a.tune else {"dim": DIM, "weights": WEIGHTS}
    out = ROOT / "scoreboard" / "submissions" / NAME
    _write(HybridTfidf(**params).fit(data), data, out)
    (out / "SHARED_COORDINATES").touch()
    (out / "AUTHOR").write_text("physcorgi\n", encoding="utf-8")
    (out / "params.json").write_text(json.dumps({**params, "tuned_on": "validation" if a.tune else "defaults"},
                                                ensure_ascii=False) + "\n", encoding="utf-8")
    r = submit.score(out)
    submit.render()
    print(f"\n{NAME}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
