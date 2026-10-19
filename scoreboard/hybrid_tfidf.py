"""Гибрид: behavioral ×2 + кто собеседники + какие сервисы (TF-IDF → SVD).

    PT_DATA=/путь/к/lanl/processed python scoreboard/hybrid_tfidf.py

Части:
  behavioral — профиль трафика (methods.baselines.temporal), вес 2;
  peers      — хост × собеседник с направлением («o:Comp1» — ходит к нему, «i:Comp1» — к нему ходит);
  services   — хост × сервис («u:6/443» — пользуется, «s:6/443» — обслуживает).
Для peers и services: log1p числа потоков → TF-IDF → нормировка строки → TruncatedSVD 128.
Словарь, IDF и SVD подбираются на train и применяются к любому окну — одна система координат.
Веса частей и размерность выбраны на validation (test не смотрели).
"""
from __future__ import annotations

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
        self.svd = TruncatedSVD(self.dim, random_state=self.seed).fit(normalize(self.tfidf.transform(x)))
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


def main() -> None:
    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    train = data.training_flows()
    beh = BehavioralBaseline().fit(data)
    graph = {k: TfidfSVD(k).fit(train, data.cohort) for k in ("peers", "services")}
    out = ROOT / "scoreboard" / "submissions" / NAME
    out.mkdir(parents=True, exist_ok=True)
    for window, hosts, fname in (("train", data.cohort, "embeddings.csv"), ("test", data.evaluable("test"), "test.csv")):
        flows = train if window == "train" else data.flows(window)
        parts = {"behavioral": beh.embed(data, window, hosts).to_numpy()}
        parts.update({k: m.embed(flows, hosts) for k, m in graph.items()})
        x = np.hstack([_block(parts[k]) * w for k, w in WEIGHTS.items()])
        pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(out / fname, float_format="%.5g")
    (out / "SHARED_COORDINATES").touch()
    r = submit.score(out)
    submit.render()
    print(f"\n{NAME}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
