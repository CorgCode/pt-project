"""Предсказывает профиль следующего среза train. Подбор — только на validation."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scoreboard'))
from temporal_encoder import TemporalEncoder, _encoder, _block, view
from evaluation import submit, scoreboard
from evaluation.knn_consistency import build_reference_features
from pipeline.data import ProcessedData

NAME = 'future_profile_predictor'


class FutureProfilePredictor(TemporalEncoder):
    name = NAME

    def __init__(self, epochs=20, pca_dim=384, splits=4, weight=1.0):
        super().__init__(recipe={'behavioral': 2.0, 'peers': 1.0, 'services': 1.0, 'neural': weight},
                         epochs=epochs, pca_dim=pca_dim, splits=splits)

    def _fit_mlp(self, data, xs, ok, active):
        import torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(2)
        self.net = _encoder(xs[0].shape[1])
        decoder = nn.Linear(128, xs[0].shape[1])
        opt = torch.optim.AdamW(list(self.net.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-4)
        xs = [torch.from_numpy(x) for x in xs]
        rng = np.random.default_rng(17)
        self.loss_history = []
        pairs = [(i, i + 1, np.flatnonzero(ok[i] & ok[i + 1])) for i in range(len(xs) - 1)]
        if not any(len(hosts) >= 2 for _, _, hosts in pairs):
            raise ValueError('нет активных хостов в соседних срезах train')
        for epoch in range(self.epochs):
            self.net.train()
            losses = []
            for i, j, hosts in pairs:
                hosts = rng.permutation(hosts)
                for start in range(0, len(hosts), 256):
                    batch = hosts[start:start + 256]
                    if len(batch) < 2:
                        continue
                    za, zb = self.net(xs[i][batch]), self.net(xs[j][batch])
                    # Прогноз только вперёд; контрастивный член сохраняет различимость хостов.
                    prediction = F.smooth_l1_loss(decoder(za), xs[j][batch])
                    logits = za @ zb.T / 0.15
                    labels = torch.arange(len(batch))
                    contrast = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                    loss = prediction + 0.2 * contrast
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(self.net.parameters()) + list(decoder.parameters()), 5.0)
                    opt.step()
                    losses.append(float(loss.detach()))
            self.loss_history.append(float(np.mean(losses)))
            print(f'epoch {epoch + 1}: {self.loss_history[-1]:.4f}', flush=True)

    def _parts(self, data, window, hosts):
        # Все хосты проходят fitted-преобразование: порядок портов совпадает во всех окнах.
        key = (str(data.root), window)
        if key not in self._parts_cache:
            if window == 'train':
                z = self._train
            else:
                w = data.manifest['windows'][window]
                z = self._features(view(data.flows(window), self.hosts, w['end'] - w['start']))
            behavior = build_reference_features(data.flows(window), hosts=self.hosts, fitted=self.beh._fitted).to_numpy()
            parts = {'behavioral': behavior, 'neural': self._neural(z['input']),
                     'peers': z['peers'], 'services': z['services']}
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
    model = FutureProfilePredictor().fit(data)
    tuning = []
    if args.tune:
        with tempfile.TemporaryDirectory() as tmp:
            for weight in (0.0, 0.5, 1.0, 2.0):
                model.recipe['neural'] = weight
                model.fit_pca(data)
                a, b = Path(tmp) / 'train.csv', Path(tmp) / 'validation.csv'
                model.embed(data, 'train').to_csv(a)
                model.embed(data, 'validation', data.evaluable('validation')).to_csv(b)
                score, tasks = submit.validation_score(a, b)
                if not np.isfinite(score) or len(tasks) != 4:
                    raise ValueError('неполная validation-оценка')
                tuning.append({'weight': weight, 'score': score, 'tasks': tasks})
                print(tuning[-1], flush=True)
        model.recipe['neural'] = max(tuning, key=lambda r: r['score'])['weight']
        model.fit_pca(data)
    args.output.mkdir(parents=True, exist_ok=True)
    if not (args.output / 'README.md').exists():
        shutil.copyfile(ROOT / 'scoreboard/submissions' / NAME / 'README.md', args.output / 'README.md')
    for w in ['train'] + data.test_windows:
        hosts = data.evaluable(w)
        name = 'embeddings.csv' if w == 'train' else submit.window_file(w)
        model.embed(data, w, hosts).to_csv(args.output / name, float_format='%.8g')
    (args.output / 'AUTHOR').write_text('DrMogger\n')
    (args.output / 'SHARED_COORDINATES').touch()
    (args.output / 'params.json').write_text(json.dumps({'recipe': model.recipe, 'epochs': model.epochs,
        'splits': model.splits, 'pca_dim': model.pca_dim, 'seed': 17, 'tuned_on': 'validation' if args.tune else None,
        'tuning': tuning, 'loss_history': model.loss_history}, ensure_ascii=False, indent=2) + '\n')
    result = submit.score(args.output)
    cols = {s['metric']: s['title'] for s in result['summary']}
    print('SCORE', scoreboard.overall_score([result], cols).iloc[0], flush=True)


if __name__ == '__main__':
    main()
