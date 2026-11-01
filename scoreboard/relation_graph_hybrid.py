"""Направленные профили окружения отдельно по сервисам; общий train, подбор на validation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scoreboard'))
from temporal_encoder import TemporalEncoder, _block, view
from evaluation import submit, scoreboard
from evaluation.knn_consistency import build_reference_features
from pipeline.data import ProcessedData

NAME = 'relation_graph_hybrid'


def relation_adjacency(counts, hosts, service=None, excluded=()):
    """Граф хостов когорты: направление, тип сервиса, log-вес и подавление хабов."""
    g = counts.reset_index(name='n')
    g = g.loc[g.t.str.startswith('o:')].copy()
    split = g.t.str.split('|', n=1, expand=True)
    g['peer'], g['service'] = split[0].str[2:], split[1]
    keep = g.service.eq(service) if service is not None else ~g.service.isin(excluded)
    g = g.loc[keep].groupby(['h', 'peer'], sort=False)['n'].sum().reset_index()
    index = pd.Index(hosts)
    a, b = index.get_indexer(g.h), index.get_indexer(g.peer)
    ok = (a >= 0) & (b >= 0) & (a != b)
    graph = sp.csr_matrix((np.log1p(g.n.to_numpy()[ok]), (a[ok], b[ok])), shape=(len(hosts), len(hosts)))
    def norm(m):
        degree = np.asarray(m.sum(axis=0)).ravel()
        return normalize(m @ sp.diags(1 / np.sqrt(np.maximum(degree, 1))), norm='l1', axis=1)
    return norm(graph), norm(graph.T.tocsr())


class RelationGraphHybrid(TemporalEncoder):
    name = NAME

    def __init__(self, pca_dim=384, weight=1.0, top_services=4):
        super().__init__(recipe={'behavioral': 2.0, 'peers': 1.0, 'services': 1.0, 'relation': weight},
                         encoder='cca', pca_dim=pca_dim, splits=2)
        self.top_services = top_services

    def fit(self, data):
        self.relations = None
        return super().fit(data)

    def _fit_cca(self, xs, ok):
        # В этом методе временная CCA не используется; обучаются только TF-IDF/SVD и итоговый PCA.
        pass

    def _features(self, v):
        z = super()._features(v)
        counts = v['counts']['context']
        if self.relations is None:
            g = counts.reset_index(name='n')
            g = g.loc[g.t.str.startswith('o:')]
            service = g.t.str.split('|', n=1).str[1]
            totals = g.assign(service=service).groupby('service')['n'].sum()
            self.relations = list(totals.sort_index().sort_values(ascending=False, kind='stable').head(self.top_services).index)
        features = []
        for relation in self.relations + [None]:
            out, inn = relation_adjacency(counts, self.hosts, relation, self.relations)
            features.extend([out @ z['services'], inn @ z['services']])
        # Два шага через общего партнёра; нормировки применяются на каждом шаге.
        all_out, all_in = relation_adjacency(counts, self.hosts)
        features.extend([all_out @ (all_in @ z['services']), all_in @ (all_out @ z['services'])])
        z['relation'] = np.hstack(features)
        return z

    def _parts(self, data, window, hosts):
        key = (str(data.root), window)
        if key not in self._parts_cache:
            if window == 'train':
                z = self._train
            else:
                w = data.manifest['windows'][window]
                z = self._features(view(data.flows(window), self.hosts, w['end'] - w['start']))
            behavior = build_reference_features(data.flows(window), hosts=self.hosts, fitted=self.beh._fitted).to_numpy()
            parts = {'behavioral': behavior, 'peers': z['peers'], 'services': z['services'], 'relation': z['relation']}
            self._parts_cache[key] = {k: _block(v) for k, v in parts.items()}
        rows = pd.Index(self.hosts).get_indexer(hosts)
        if (rows < 0).any():
            raise ValueError('неизвестный хост')
        return {k: v[rows] for k, v in self._parts_cache[key].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=os.environ.get('PT_DATA', ROOT / 'scoreboard/data/processed'))
    ap.add_argument('--output', type=Path, default=ROOT / 'scoreboard/submissions' / NAME)
    ap.add_argument('--tune', action='store_true')
    args = ap.parse_args()
    data = ProcessedData(args.data)
    official = ProcessedData(ROOT / 'scoreboard/data/processed')
    if data.cohort != official.cohort or data.manifest['windows'] != official.manifest['windows']:
        raise ValueError('для общей оценки нужны общая когорта и окна')
    if (args.output / 'embeddings.csv').exists():
        raise FileExistsError('посылка уже существует')
    model = RelationGraphHybrid().fit(data)
    tuning = []
    if args.tune:
        with tempfile.TemporaryDirectory() as tmp:
            for weight in (0.0, 0.5, 1.0, 2.0):
                model.recipe['relation'] = weight
                model.fit_pca(data)
                a, b = Path(tmp) / 'train.csv', Path(tmp) / 'validation.csv'
                model.embed(data, 'train').to_csv(a)
                model.embed(data, 'validation', data.evaluable('validation')).to_csv(b)
                score, tasks = submit.validation_score(a, b)
                if not np.isfinite(score) or len(tasks) != 4:
                    raise ValueError('неполная validation-оценка')
                tuning.append({'weight': weight, 'score': score, 'tasks': tasks})
                print(tuning[-1], flush=True)
        model.recipe['relation'] = max(tuning, key=lambda r: r['score'])['weight']
        model.fit_pca(data)
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / 'README.md').exists():
        shutil.copyfile(ROOT / 'scoreboard/submissions' / NAME / 'README.md', args.output / 'README.md')
    for w in ['train'] + data.test_windows:
        name = 'embeddings.csv' if w == 'train' else submit.window_file(w)
        model.embed(data, w, data.evaluable(w)).to_csv(args.output / name, float_format='%.8g')
    (args.output / 'AUTHOR').write_text('DrMogger\n')
    (args.output / 'SHARED_COORDINATES').touch()
    (args.output / 'params.json').write_text(json.dumps({'recipe': model.recipe, 'relations': model.relations,
        'pca_dim': model.pca_dim, 'tuned_on': 'validation' if args.tune else None, 'tuning': tuning},
        ensure_ascii=False, indent=2) + '\n')
    result = submit.score(args.output)
    cols = {s['metric']: s['title'] for s in result['summary']}
    print('SCORE', scoreboard.overall_score([result], cols).iloc[0], flush=True)


if __name__ == '__main__':
    main()
