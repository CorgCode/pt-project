"""temporal_encoder + «веер по сервису» (идеи 1 и 4 из заметок о методах) + ансамбль нейросетей по зёрнам.

Веер — для каждого сервиса (протокол + порт) число разных собеседников: к скольким хостам хост ходит
как клиент (fu:) и сколько клиентов обслуживает как сервер (fs:). Счётчики → TF-IDF → SVD.
Ансамбль: нейросеть обучается с несколькими зёрнами, их векторы ставятся рядом (у каждой свои оси,
усреднять нельзя), вес каждой — 1/√k, чтобы общий вес части не зависел от числа зёрен.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scoreboard"))
import temporal_encoder as T  # noqa: E402


def fan_counts(flows: pd.DataFrame, hosts: list[str]) -> pd.Series:
    idx, out = pd.Index(hosts), []
    for tag, end, peer in (("fu:", "src", "dst"), ("fs:", "dst", "src")):
        f = flows.loc[flows[end].isin(idx), [end, peer, "proto", "dst_port"]]
        g = f.drop_duplicates().groupby([end, "proto", "dst_port"], observed=True, sort=False).size().reset_index(name="n")
        out.append(pd.DataFrame({"h": g[end].astype(str), "n": g["n"],
                                 "t": tag + g["proto"].astype(str) + "/" + g["dst_port"].astype(str)}))
    return pd.concat(out).groupby(["h", "t"], sort=False)["n"].sum()


class FanEncoder(T.TemporalEncoder):
    """Части как у temporal_encoder плюс fan; seeds > 1 — ансамбль нейросетей."""

    def __init__(self, *args, fan_dim: int = 64, seeds: tuple[int, ...] = (T.SEED,), **kw):
        super().__init__(*args, **kw)
        self.fan_dim, self.seeds = fan_dim, tuple(seeds)

    def fit(self, data):
        recipe, self.recipe = self.recipe, {k: v for k, v in self.recipe.items() if k != "fan"}   # веера ещё нет
        nets = []
        for s in self.seeds:                                  # зерно читается внутри T при обучении
            T.SEED = s
            super().fit(data)
            nets.append((self.net, self.scaler) if self.encoder == "mlp" else (self.proj, self.corr))
        T.SEED = 17
        self._nets = nets
        c = fan_counts(data.training_flows(), self.hosts)
        self.fan_vocab = T.vocabulary(c)
        x = T.matrix(c, self.hosts, self.fan_vocab)
        self.fan_tfidf = TfidfTransformer().fit(x)
        dim = min(self.fan_dim, x.shape[1] - 1)
        self.fan_svd = TruncatedSVD(dim, random_state=0).fit(normalize(self.fan_tfidf.transform(x)))
        self._fan, self._parts_cache = {}, {}
        self.recipe = recipe
        self.fit_pca(data)
        return self

    def _neural(self, x: np.ndarray) -> np.ndarray:
        if len(getattr(self, "_nets", ())) <= 1:
            return super()._neural(x)
        outs = []
        for a, b in self._nets:
            if self.encoder == "mlp":
                self.net = a
            else:
                self.proj, self.corr = a, b
            outs.append(super()._neural(x))
        return np.hstack(outs) / np.sqrt(len(outs))

    def fan(self, data, window: str) -> np.ndarray:
        if window not in self._fan:
            flows = data.training_flows() if window == "train" else data.flows(window)
            x = T.matrix(fan_counts(flows, self.hosts), self.hosts, self.fan_vocab)
            self._fan[window] = self.fan_svd.transform(normalize(self.fan_tfidf.transform(x)))
        return self._fan[window]

    def _parts(self, data, window, hosts):
        parts = super()._parts(data, window, hosts)
        if "fan" not in parts and hasattr(self, "fan_svd"):
            parts["fan"] = T._block(self.fan(data, window)[pd.Index(self.hosts).get_indexer(hosts)])
        return parts
