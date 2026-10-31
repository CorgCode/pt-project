"""Разреженные отпечатки трафика и память нескольких наблюдений каждого хоста."""
from __future__ import annotations

import hashlib
import marshal
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection

from experiments.overnight.model import T


def fingerprint_counts(flows, hosts):
    # Self-flow мог бы содержать собственный ID в признаке собеседника.
    flows = flows.loc[flows.src != flows.dst, ["src", "dst", "proto", "dst_port"]]
    index, records = pd.Index(hosts), {"peers": [], "context": []}
    events = np.zeros(len(hosts))
    for side, end, peer in [("o", "src", "dst"), ("i", "dst", "src")]:
        selected = flows.loc[flows[end].isin(index)]
        grouped = selected.groupby([end, peer, "proto", "dst_port"], observed=True, sort=False).size().reset_index(name="n")
        h = grouped[end].astype(str)
        p = side + ":" + grouped[peer].astype(str)
        context = p + "|" + grouped.proto.astype(str) + "/" + grouped.dst_port.astype(str)
        for kind, token in [("peers", p), ("context", context)]:
            records[kind].append(pd.DataFrame({"h": h, "t": token, "n": grouped.n}))
        events += np.bincount(index.get_indexer(h), weights=grouped.n, minlength=len(hosts))
    return {"counts": {k: pd.concat(v).groupby(["h", "t"], sort=False).n.sum() for k, v in records.items()},
            "events": events}


def read_fingerprints(data, window, hosts, cache):
    path = data.root / "flows" / f"{window}.csv"
    stat = path.stat()
    payload = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, tuple(hosts),
               hashlib.sha256(marshal.dumps(fingerprint_counts.__code__)).hexdigest())
    key = hashlib.sha256(repr(payload).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key}.pkl"
    if target.exists():
        with target.open("rb") as stream:
            return pickle.load(stream)
    flows = data.training_flows() if window == "train" else data.flows(window)
    counts = fingerprint_counts(flows, hosts)
    temp = target.with_suffix(f".tmp{os.getpid()}")
    with temp.open("wb") as stream:
        pickle.dump(counts, stream)
    temp.replace(target)
    return counts


class FingerprintBank:
    def __init__(self, latent_dim=256):
        self.latent_dim = latent_dim

    def fit(self, observations, hosts):
        if not observations or not hosts:
            raise ValueError("Нужны обучающие наблюдения и фиксированная когорта")
        self.hosts, self.days = list(hosts), len(observations)
        self.vocab, self.tfidf = {}, {}
        channels = []
        for kind in ["peers", "context"]:
            # Сохраняются и редкие токены: нет ограничения min_df=2 и SVD-ранга.
            vocabulary = sorted(set().union(*[set(v["counts"][kind].index.get_level_values("t")) for v in observations]))
            self.vocab[kind] = pd.Index(vocabulary)
            x = sp.vstack([T.matrix(v["counts"][kind], self.hosts, self.vocab[kind]).astype(np.float32)
                           for v in observations], format="csr")
            tfidf = TfidfTransformer().fit(x)
            self.tfidf[kind] = tfidf
            channels.append(tfidf.transform(x).astype(np.float32))
        memory = normalize(sp.hstack(channels, format="csr")).astype(np.float32)
        active = np.concatenate([v["events"] >= 20 for v in observations])
        self.memory = memory.multiply(active[:, None]).tocsr()
        self.memory.eliminate_zeros()
        n = len(self.hosts)
        weights = 2.0 ** (-np.arange(self.days - 1, -1, -1) / 3.0)
        prototype = sum(self.memory[i*n:(i+1)*n] * weight for i, weight in enumerate(weights))
        self.prototype = normalize(prototype).astype(np.float32)
        self.projection = SparseRandomProjection(n_components=min(self.latent_dim, memory.shape[1]),
                                                  dense_output=True, random_state=31).fit(self.prototype)
        self.class_projection = np.random.default_rng(73).standard_normal((n, self.latent_dim)).astype(np.float32)
        self.class_projection /= np.sqrt(self.latent_dim)
        return self

    def transform(self, counts):
        channels = []
        for kind in ["peers", "context"]:
            x = T.matrix(counts["counts"][kind], self.hosts, self.vocab[kind]).astype(np.float32)
            channels.append(self.tfidf[kind].transform(x).astype(np.float32))
        return normalize(sp.hstack(channels, format="csr")).astype(np.float32)

    def set_stability(self, minimum_days=2):
        if not hasattr(self, "_raw_memory"):
            self._raw_memory = self.memory
        n = len(self.hosts)
        occurrences = sum(self._raw_memory[i*n:(i+1)*n].sign() for i in range(self.days))
        reliability = occurrences.copy()
        reliability.data = ((reliability.data >= minimum_days)
                            * np.sqrt(reliability.data / self.days)).astype(np.float32)
        reliability.eliminate_zeros()
        blocks = [normalize(self._raw_memory[i*n:(i+1)*n].multiply(reliability)).astype(np.float32)
                  for i in range(self.days)]
        self.memory = sp.vstack(blocks, format="csr")
        weights = 2.0 ** (-np.arange(self.days - 1, -1, -1) / 3.0)
        self.prototype = normalize(sum(x * w for x, w in zip(blocks, weights))).astype(np.float32)
        return self

    def direct(self, x):
        return normalize(np.asarray(self.projection.transform(x), dtype=np.float32))

    def posterior(self, x, temperature=0.05, batch=128, topk=16):
        """Вход — только признаки трафика, без ID запрашиваемых хостов."""
        if temperature <= 0:
            raise ValueError("Температура должна быть положительной")
        outputs, predicted = [], []
        n, k = len(self.hosts), min(topk, len(self.hosts))
        memory = self.memory.T.tocsr()
        for offset in range(0, x.shape[0], batch):
            similarities = (x[offset:offset+batch] @ memory).toarray()
            scores = similarities.reshape(-1, self.days, n).max(axis=1)
            nearest = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            local = np.take_along_axis(scores, nearest, axis=1)
            labels = nearest[np.arange(len(nearest)), local.argmax(axis=1)]
            labels[scores.max(axis=1) <= 0] = -1
            predicted.extend(labels)
            logits = (local - local.max(axis=1, keepdims=True)) / temperature
            probability = np.exp(logits)
            probability /= probability.sum(axis=1, keepdims=True)
            projected = np.einsum("bk,bkd->bd", probability, self.class_projection[nearest])
            projected[scores.max(axis=1) <= 0] = 0
            outputs.append(normalize(projected).astype(np.float32))
        return np.vstack(outputs), np.asarray(predicted)


class FingerprintHybrid:
    def __init__(self, base, bank, cache):
        self.base, self.bank, self.cache = base, bank, str(cache)
        self.mode, self.weight, self.temperature = "memory", 0.5, 0.05
        self.hosts = list(bank.hosts)
        self._encoded = {}
        # Эти массивы нужны были обучению прежней головы, а не её выводу.
        for name in ["_xs", "_ok", "_active", "_training_counts"]:
            self.base.__dict__.pop(name, None)
        self.base._parts_cache = {}
        self.base._evaluation_cache, self.base._behavior_cache = {}, {}

    def embed(self, data, window, hosts):
        anchor = self.base.embed(data, window, hosts)
        if not self.weight:
            return anchor
        key = (str(data.root), window, tuple(hosts), self.mode, self.temperature)
        if key not in self._encoded:
            rows = pd.Index(self.hosts).get_indexer(hosts)
            if (rows < 0).any():
                raise ValueError("Хост вне фиксированной обучающей когорты")
            if window == "train":
                x = self.bank.prototype[rows]
            else:
                counts = read_fingerprints(data, window, self.hosts, self.cache)
                x = self.bank.transform(counts)[rows]
            part = self.bank.direct(x) if self.mode == "direct" else self.bank.posterior(x, self.temperature)[0]
            self._encoded[key] = T._block(part)
        vectors = np.hstack([T._block(anchor.to_numpy()), self._encoded[key] * self.weight])
        return pd.DataFrame(vectors, index=anchor.index).add_prefix("emb_")
