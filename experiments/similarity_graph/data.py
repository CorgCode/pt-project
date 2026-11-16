"""Потоковое извлечение признаков окна: host_id используется только для группировки."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


@dataclass
class Window:
    name: str
    day: int
    start: float
    end: float
    flows: str
    baseline: str | None = None


@dataclass
class Observation:
    window: Window
    counts: list[dict]
    numeric: np.ndarray
    events: np.ndarray


def signature(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_window(window, hosts, cache, chunksize=250_000):
    if window.end <= window.start:
        raise ValueError('пустое окно')
    path = Path(window.flows)
    stat = path.stat()
    stamp = json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                        window.start, window.end, hosts, 1]).encode()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / (hashlib.sha256(stamp).hexdigest() + '.joblib')
    if saved.exists():
        counts, numeric, events = joblib.load(saved)
        return Observation(window, counts, numeric, events)
    index = pd.Index(hosts)
    counts = [Counter() for _ in hosts]
    stats = np.zeros((len(hosts), 20), dtype=float)
    usecols = ['src', 'dst', 'proto', 'dst_port', 'ts', 'duration', 'bytes', 'packets', 'resp_bytes', 'resp_packets']
    for chunk in pd.read_csv(path, usecols=usecols, dtype={k: str for k in ('src', 'dst', 'proto', 'dst_port')}, chunksize=chunksize):
        f = chunk.loc[(chunk.ts >= window.start) & (chunk.ts < window.end) &
                      (chunk.ts + chunk.duration < window.end) & (chunk.src != chunk.dst)]
        for side, endpoint, peer in ((0, 'src', 'dst'), (1, 'dst', 'src')):
            part = f.loc[f[endpoint].isin(index)]
            if part.empty:
                continue
            direction = 'o' if side == 0 else 'i'
            g = part.groupby([endpoint, peer], sort=False).size()
            for (host, partner), count in g.items():
                counts[index.get_loc(host)][f'p:{direction}:{partner}'] += int(count)
            g = part.groupby([endpoint, 'proto', 'dst_port'], sort=False).size()
            for (host, proto, port), count in g.items():
                counts[index.get_loc(host)][f's:{direction}:{proto}/{port}'] += int(count)
            rows = index.get_indexer(part[endpoint])
            offset = side * 10
            stats[:, offset] += np.bincount(rows, minlength=len(hosts))
            byte_column, packet_column = ('bytes', 'packets') if side == 0 else ('resp_bytes', 'resp_packets')
            for k, column in enumerate((byte_column, packet_column, 'duration'), 1):
                stats[:, offset + k] += np.bincount(rows, weights=part[column], minlength=len(hosts))
            bins = np.clip(((part.ts.to_numpy() - window.start) * 6 / (window.end - window.start)).astype(int), 0, 5)
            for b in range(6):
                stats[:, offset + 4 + b] += np.bincount(rows[bins == b], minlength=len(hosts))
    events = stats[:, 0] + stats[:, 10]
    numeric = np.log1p(stats * 3600 / (window.end - window.start))
    joblib.dump((counts, numeric, events), saved, compress=3)
    return Observation(window, counts, numeric, events)


def contacts(observation):
    """Ненаправленные частоты точно как build_partners; сервисные токены не используются."""
    out = []
    for row in observation.counts:
        peers = Counter()
        for token, count in row.items():
            if token.startswith('p:'):
                peers[token[4:]] += count
        out.append(dict(peers))
    return out


def check_schedule(train, development, tests):
    if not train or not development:
        raise ValueError('нужны train и development')
    train = sorted(train, key=lambda w: w.start)
    if any(a.end > b.start for a, b in zip(train, train[1:])):
        raise ValueError('обучающие окна пересекаются')
    if max(w.end for w in train) > min(w.start for w in development):
        raise ValueError('train пересекает development')
    if tests and max(w.end for w in development) > min(w.start for w in tests):
        raise ValueError('development пересекает test')
    names = [w.name for w in train + development + tests]
    if len(names) != len(set(names)):
        raise ValueError('имена окон должны быть уникальны')
    if any(w.end<=w.start for w in train+development+tests):
        raise ValueError('пустое окно')
