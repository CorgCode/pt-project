"""Точный расчёт host_profile.py по агрегатам, без разворачивания миллионов потоков."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from host_profile import HostProfileEmbedder, _struct, _tokenize


def aggregate(flows: pd.DataFrame, window_sec: float = 1800) -> pd.DataFrame:
    long = _tokenize(flows.rename(columns={"ts": "time", "src_port": "sport", "dst_port": "dport"}), True)
    long["w"] = (long["t"] // window_sec).astype(np.int64)
    return long.groupby(["host", "key", "w"]).size().rename("n").reset_index()


class AggregatedHostProfile(HostProfileEmbedder):
    """Та же модель; log применяется после суммирования счётчиков всех чанков."""

    def _counts(self, counts):
        counts = counts.copy()
        if not self.direction:
            counts["key"] %= 1_000_000
        return counts.groupby(["host", "key", "w"], as_index=False)["n"].sum()

    def _profile_matrix(self, counts):
        h = self.hosts_.get_indexer(counts["host"])
        k = self.vocab_.get_indexer(counts["key"])
        keep = (h >= 0) & (k >= 0)
        g = counts.loc[keep, ["w", "n"]].copy()
        g["h"], g["k"] = h[keep], k[keep]
        nwin = g[["h", "w"]].drop_duplicates().groupby("h").size()
        g["tf"] = g["n"].astype(float) if self.weighting == "raw" else np.log1p(g["n"])
        a = g.groupby(["h", "k"], as_index=False)["tf"].sum()
        a["tf"] /= nwin.reindex(a["h"]).to_numpy()
        return sp.csr_matrix((a["tf"], (a["h"], a["k"])), shape=(len(self.hosts_), len(self.vocab_)))

    def fit_profile(self, counts, hosts, edges=None):
        if self.skipgram_weight:
            raise ValueError("Skip-gram требует последовательностей; агрегаты их не сохраняют")
        self.hosts_ = pd.Index(hosts)
        counts = self._counts(counts)
        counts = counts[counts["host"].isin(self.hosts_)]
        self.vocab_ = pd.Index(np.sort(counts["key"].unique()))
        X = self._profile_matrix(counts)
        n, p = X.shape
        if min(n, p) < 3:
            raise ValueError("Для SVD нужны минимум три хоста и три токена")
        self.idf_ = np.log((n + 1) / (np.asarray(X.getnnz(axis=0)) + 1)) + 1
        X = self._weight(X)
        d = min(self.dim, p - 1, n - 1)
        self.svd_ = TruncatedSVD(d, random_state=self.seed).fit(X)
        Z = self.svd_.transform(X)
        act = np.asarray(X.getnnz(axis=1)) > 0
        self.mean_ = Z[act].mean(0) if self.center else np.zeros(d)
        if self.struct_weight:
            S = _struct(edges, self.hosts_)
            self.s_mu_, self.s_sd_ = S[act].mean(0), S[act].std(0) + 1e-9
        return self

    def transform_profile(self, counts, edges=None):
        X = self._weight(self._profile_matrix(self._counts(counts)))
        act = np.asarray(X.getnnz(axis=1)) > 0
        Z = self.svd_.transform(X)
        Z[act] -= self.mean_
        blocks = [normalize(Z)]
        if self.struct_weight:
            S = (_struct(edges, self.hosts_) - self.s_mu_) / self.s_sd_
            S[~act] = 0
            blocks.append(self.struct_weight * normalize(S))
        E = normalize(np.hstack(blocks))
        E[~act] = 0
        return pd.DataFrame(E, index=pd.Index(self.hosts_, name="host_id"),
                            columns=[f"emb_{i}" for i in range(E.shape[1])])
