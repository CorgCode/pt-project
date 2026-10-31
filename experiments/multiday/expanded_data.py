"""32 обучающих часа, потоковый хешированный словарь без будущих наблюдений."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from experiments.multiday.search import dump, load, write_json
from pipeline.data import ProcessedData
from pipeline.lanl_netflow import canonicalize, iter_raw_chunks


PEERS, CONTEXT = 65536, 196608
WIDTH = PEERS + CONTEXT
HOURS = (10, 11, 12, 13)


def token_columns(tokens, kind):
    tokens = np.asarray(tokens, dtype=object)
    hashed = pd.util.hash_array(tokens, hash_key='0123456789123456')
    size, offset = (PEERS, 0) if kind == 'peers' else (CONTEXT, PEERS)
    return (hashed % size).astype(np.int32) + offset


def count_flows(flows, hosts, slots=None, n_slots=1):
    """В строке только traffic features; собственный ID не становится токеном."""
    index = pd.Index(hosts)
    if slots is None:
        slots = np.zeros(len(flows), dtype=np.int32)
    flows = flows.assign(_slot=np.asarray(slots))
    flows = flows.loc[flows.src != flows.dst]
    matrices, events = [], np.zeros(n_slots * len(hosts), dtype=np.float64)
    for side, end, peer in [('o', 'src', 'dst'), ('i', 'dst', 'src')]:
        selected = flows.loc[flows[end].isin(index)]
        grouped = selected.groupby(['_slot', end, peer, 'proto', 'dst_port'], observed=True, sort=False).size().reset_index(name='n')
        rows = grouped._slot.to_numpy(dtype=np.int32) * len(hosts) + index.get_indexer(grouped[end].astype(str))
        values = grouped.n.to_numpy(dtype=np.float32)
        events += np.bincount(rows, weights=values, minlength=len(events))
        peers = side + ':' + grouped[peer].astype(str)
        contexts = peers + '|' + grouped.proto.astype(str) + '/' + grouped.dst_port.astype(str)
        for kind, tokens in [('peers', peers), ('context', contexts)]:
            columns = token_columns(tokens, kind)
            matrices.append(sp.csr_matrix((values, (rows, columns)), shape=(len(events), WIDTH)))
    return sum(matrices).tocsr(), events


def count_series(counts, hosts):
    index, matrices = pd.Index(hosts), []
    for kind in ['peers', 'context']:
        series = counts['counts'][kind]
        rows = index.get_indexer(series.index.get_level_values('h'))
        if (rows < 0).any():
            raise ValueError('Незнакомый хост в агрегатах')
        columns = token_columns(series.index.get_level_values('t'), kind)
        matrices.append(sp.csr_matrix((series.to_numpy(dtype=np.float32), (rows, columns)), shape=(len(hosts), WIDTH)))
    return sum(matrices).tocsr()


def extract_day(raw, hosts, out, day, chunksize=500_000):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    accum = sp.csr_matrix((len(HOURS) * len(hosts), WIDTH), dtype=np.float32)
    events = np.zeros(accum.shape[0], dtype=np.float64)
    seen, kept, invalid = 0, 0, 0
    start = (day - 1) * 86400 + HOURS[0] * 3600
    end = (day - 1) * 86400 + (HOURS[-1] + 1) * 3600
    for chunk in iter_raw_chunks(raw, chunksize):
        seen += len(chunk)
        ts = pd.to_numeric(chunk.Time, errors='coerce')
        duration = pd.to_numeric(chunk.Duration, errors='coerce')
        selected = (ts >= start) & (ts < end) & (duration >= 0)
        # Исключаем пересечение часовых границ, как в исходном протоколе.
        selected &= ts + duration <= (np.floor(ts / 3600) + 1) * 3600
        if selected.any():
            flows, bad = canonicalize(chunk.loc[selected])
            invalid += sum(bad.values())
            slots = ((flows.ts.to_numpy() - start) // 3600).astype(np.int32)
            part, local_events = count_flows(flows, hosts, slots, len(HOURS))
            accum += part
            events += local_events
            kept += len(flows)
        if seen % 10_000_000 < chunksize:
            print(json.dumps({'day': day, 'raw_rows': seen, 'kept': kept, 'nnz': accum.nnz}), flush=True)
    # Полное чтение архива проверяет CRC; complete пишется только после успеха.
    sp.save_npz(out / 'counts.npz', accum)
    np.save(out / 'events.npy', events)
    write_json(out / 'complete.json', {'day': day, 'hours': HOURS, 'hosts': len(hosts),
        'raw': str(raw), 'raw_bytes': raw.stat().st_size, 'raw_rows': seen, 'kept': kept,
        'invalid': invalid, 'interval': [start, end], 'nnz': accum.nnz})


class ExpandedBank:
    def fit(self, day_dirs, hosts, validation_start):
        manifests = [json.loads((p / 'complete.json').read_text()) for p in day_dirs]
        if any(m['interval'][1] > validation_start for m in manifests):
            raise ValueError('Обучение пересекает validation')
        if len({m['day'] for m in manifests}) != len(manifests):
            raise ValueError('Обучающий день повторён')
        self.hosts, self.days = list(hosts), len(day_dirs) * len(HOURS)
        self.training_manifests = manifests
        raw = sp.vstack([sp.load_npz(p / 'counts.npz') for p in day_dirs], format='csr')
        events = np.concatenate([np.load(p / 'events.npy') for p in day_dirs])
        self.tfidf = TfidfTransformer().fit(raw)
        memory = self.transform_sparse(raw)
        del raw
        n = len(hosts)
        # Надёжность токена определяется повторением в разных TRAIN-днях,
        # а не четырьмя повторениями в один день. На query ID-фильтр не применяется.
        occurrences = sp.csr_matrix((n, WIDTH), dtype=np.float32)
        for day in range(len(day_dirs)):
            within = sum(memory[(day*len(HOURS)+hour)*n:(day*len(HOURS)+hour+1)*n].sign() for hour in range(len(HOURS)))
            occurrences += within.sign()
        reliability = occurrences.copy()
        reliability.data = np.where(reliability.data < 2, 0.15, np.sqrt(reliability.data / len(day_dirs))).astype(np.float32)
        blocks = []
        for view in range(self.days):
            part = memory[view*n:(view+1)*n].multiply(reliability)
            part = normalize(part).multiply((events[view*n:(view+1)*n] >= 20)[:, None]).tocsr()
            part.eliminate_zeros()
            blocks.append(part.astype(np.float32))
        self.memory = sp.vstack(blocks, format='csr')
        weights = 2.0 ** (-np.arange(self.days-1, -1, -1) / (3 * len(HOURS)))
        self.prototype = normalize(sum(part * w for part, w in zip(blocks, weights))).astype(np.float32)
        return self

    def transform_sparse(self, raw):
        x = self.tfidf.transform(raw).astype(np.float32)
        return normalize(sp.hstack([normalize(x[:, :PEERS]), normalize(x[:, PEERS:])], format='csr')).astype(np.float32)

    def transform(self, counts):
        return self.transform_sparse(count_series(counts, self.hosts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--base', required=True, type=Path)
    ap.add_argument('--raw', type=Path)
    ap.add_argument('--day', type=int)
    ap.add_argument('--fit', action='store_true')
    args = ap.parse_args()
    data = ProcessedData(args.base)
    if args.fit:
        days = [args.work / f'day{day:02d}' for day in range(2, 10)]
        bank = ExpandedBank().fit(days, data.cohort, 9*86400 + 12.5*3600)
        sp.save_npz(args.work / 'training.npz', bank.memory)
        # Голова уже получает хешированные признаки; remapping — единичный.
        np.save(args.work / 'buckets.npy', np.arange(WIDTH, dtype=np.int32))
        write_json(args.work / 'hosts.json', bank.hosts)
        dump(bank, args.work / 'bank.pkl')
        write_json(args.work / 'inputs_ready.json', {'train_days': list(range(2,10)), 'hours': HOURS,
            'train_hours_total': len(days)*len(HOURS), 'observations': bank.memory.shape[0], 'nnz': bank.memory.nnz,
            'width': WIDTH, 'validation_used_for_training': False})
    else:
        if args.raw is None or args.day is None:
            ap.error('--raw и --day обязательны без --fit')
        extract_day(args.raw, data.cohort, args.work / f'day{args.day:02d}', args.day)


if __name__ == '__main__':
    main()
