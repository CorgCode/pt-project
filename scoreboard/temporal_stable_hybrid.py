"""Линейный гибрид с проекцией, устойчивой между частями train.

python scoreboard/temporal_stable_hybrid.py --tune --output runs/temporal_stable_hybrid
PT_DATA должен содержать потоки и манифест общей подготовки. Подбор использует только validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scoreboard'))
from evaluation import submit  # noqa: E402
from evaluation.knn_consistency import build_reference_features  # noqa: E402
from methods.baselines.temporal import BehavioralBaseline  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402
from temporal_encoder import matrix, view, vocabulary  # noqa: E402


def stable_projection(views, activity, dim=128, regularization=0.1, min_flows=5):
    """Минимум временного разброса при единичной общей ковариации (slow feature analysis)."""
    x = np.asarray(views, dtype=float)
    n = np.asarray(activity, dtype=float)
    if x.ndim != 3 or n.shape != x.shape[:2] or len(x) < 2:
        raise ValueError("ожидаются признаки и активности хостов минимум в двух срезах")
    valid = n >= min_flows
    weights = np.sqrt(n)
    weights = np.where(valid, weights, 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("нет активных хостов в срезах train")
    mean = (x * weights[..., None]).sum(axis=(0, 1)) / total
    centered = x - mean
    covariance = np.einsum('shd,she,sh->de', centered, centered, weights) / total
    difference = np.zeros_like(covariance)
    pairs = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            w = np.minimum(weights[i], weights[j])
            delta = x[i] - x[j]
            difference += np.einsum('hd,he,h->de', delta, delta, w)
            pairs += w.sum()
    if pairs <= 0:
        raise ValueError("нет хостов, активных хотя бы в двух срезах train")
    difference /= pairs
    scale = max(float(np.trace(covariance) / covariance.shape[0]), 1e-12)
    covariance += regularization * scale * np.eye(covariance.shape[0])
    _, vectors = eigh(difference, covariance, check_finite=True)
    return mean, vectors[:, :min(dim, x.shape[2])]


class TemporalStableHybrid:
    name = "temporal_stable_hybrid"
    shared_coordinate_system = True

    @classmethod
    def load(cls, path):
        """Загрузить сохранённое состояние модели без повторного обучения."""
        if Path(path).suffix == '.npz':
            return cls._load_portable(path)
        model = cls()
        model.__dict__.update(joblib.load(path))
        return model

    def export(self, path):
        """Сохранить веса и словари как массивы без Python pickle и кэша окон."""
        fitted = self.beh._fitted
        meta = {'format': 1, 'hosts': self.hosts,
                'vocab': {k: list(v) for k, v in self.vocab.items()},
                'ports': list(fitted['ports']), 'protos': list(fitted['protos']),
                'excluded': list(fitted.get('excluded', ())), 'blocks': list(fitted['scalers']),
                'parameters': {k: getattr(self, k) for k in ('dim', 'stable_dim', 'splits',
                    'stable_weight', 'behavioral_weight', 'min_flows', 'regularization')},
                'training_window': getattr(self, 'training_window', None)}
        arrays = {'metadata': np.array(json.dumps(meta, ensure_ascii=False))}
        for name in ('stable_mean', 'projection', 'stable_center', 'stable_scale'):
            arrays[name] = np.asarray(getattr(self, name))
        for kind, (tfidf, svd) in self.models.items():
            arrays[kind + '_idf'] = tfidf.idf_
            arrays[kind + '_svd'] = svd.components_
        for name, scaler in fitted['scalers'].items():
            arrays['scaler_' + name + '_mean'] = scaler.mean_
            arrays['scaler_' + name + '_scale'] = scaler.scale_
            arrays['scaler_' + name + '_var'] = scaler.var_
        for name in self.means:
            arrays['mean_' + name] = self.means[name]
            arrays['scale_' + name] = np.asarray(self.scales[name])
        np.savez_compressed(path, **arrays)

    @classmethod
    def _load_portable(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            if meta['format'] != 1:
                raise ValueError('неподдерживаемая версия checkpoint')
            model = cls(**meta['parameters'])
            model.hosts = meta['hosts']
            model.training_window = meta['training_window']
            model.vocab = {k: pd.Index(v) for k, v in meta['vocab'].items()}
            model.models = {}
            for kind in model.vocab:
                tfidf = TfidfTransformer()
                tfidf.idf_ = saved[kind + '_idf'].copy()
                svd = TruncatedSVD(saved[kind + '_svd'].shape[0])
                svd.components_ = saved[kind + '_svd'].copy()
                svd.n_features_in_ = svd.components_.shape[1]
                model.models[kind] = tfidf, svd
            scalers = {}
            for name in meta['blocks']:
                scaler = StandardScaler()
                for key in ('mean', 'scale', 'var'):
                    setattr(scaler, key + '_', saved['scaler_' + name + '_' + key].copy())
                scaler.n_features_in_ = len(scaler.mean_)
                scalers[name] = scaler
            model.beh = BehavioralBaseline()
            model.beh._fitted = {'ports': meta['ports'], 'protos': meta['protos'],
                                'excluded': tuple(meta['excluded']), 'scalers': scalers}
            for name in ('stable_mean', 'projection', 'stable_center', 'stable_scale'):
                setattr(model, name, saved[name].copy())
            model.means = {k: saved['mean_' + k].copy() for k in ('behavioral', 'peers', 'services')}
            model.scales = {k: float(saved['scale_' + k]) for k in model.means}
            return model

    def __init__(self, dim=128, stable_dim=128, splits=4, stable_weight=0.5, behavioral_weight=2.0,
                 min_flows=5, regularization=0.1):
        if splits < 2 or dim < 1 or stable_dim < 1 or regularization <= 0:
            raise ValueError("некорректные параметры модели")
        self.dim, self.stable_dim, self.splits = dim, stable_dim, splits
        self.stable_weight, self.behavioral_weight = stable_weight, behavioral_weight
        self.min_flows, self.regularization = min_flows, regularization
        self.cache = {}

    def _tokens(self, v):
        return {k: model[1].transform(normalize(model[0].transform(matrix(v['counts'][k], self.hosts, self.vocab[k]))))
                for k, model in self.models.items()}

    def _raw(self, flows, seconds, v=None):
        v = v if v is not None else view(flows, self.hosts, seconds)
        behavior = build_reference_features(flows, hosts=self.hosts, fitted=self.beh._fitted).to_numpy()
        tokens = self._tokens(v)
        return {"behavioral": behavior, **tokens}, v['flows']

    def _scaled(self, parts):
        return {k: (x - self.means[k]) / self.scales[k] for k, x in parts.items()}

    def fit(self, data):
        self.hosts = list(data.cohort)
        print('Чтение train и поведенческие признаки...', flush=True)
        flows = data.training_flows()
        self.beh = BehavioralBaseline().fit(data)
        window = data.manifest['windows']['train']
        start, end = float(window['start']), float(window['end'])
        self.training_window = [start, end]
        print('Обучение TF-IDF и SVD...', flush=True)
        full = view(flows, self.hosts, end - start)
        self.vocab, self.models = {}, {}
        for kind in ('peers', 'services'):
            self.vocab[kind] = vocabulary(full['counts'][kind])
            x = matrix(full['counts'][kind], self.hosts, self.vocab[kind])
            n = min(self.dim, x.shape[0] - 1, x.shape[1] - 1)
            if n < 1:
                raise ValueError(f"недостаточно токенов {kind} для SVD")
            tfidf = TfidfTransformer().fit(x)
            svd = TruncatedSVD(n, random_state=0).fit(normalize(tfidf.transform(x)))
            self.models[kind] = (tfidf, svd)
        # Порядок портов после fit может отличаться от порядка при fitted-преобразовании;
        # train, срезы и отложенные окна должны проходить одинаковый путь.
        parts, _ = self._raw(flows, end - start, full)
        self.means = {k: x.mean(axis=0) for k, x in parts.items()}
        self.scales = {k: max(np.sqrt(np.mean(np.sum((x - self.means[k]) ** 2, axis=1))), 1e-12)
                       for k, x in parts.items()}
        self.cache['train'] = self._scaled(parts)
        edges = np.linspace(start, end, self.splits + 1)
        vectors, activity = [], []
        for index, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1):
            print(f'Временной срез {index}/{self.splits}...', flush=True)
            part = flows.loc[(flows.ts >= a) & (flows.ts < b)]
            p, n = self._raw(part, b - a)
            p = self._scaled(p)
            vectors.append(np.hstack([p[k] for k in ('behavioral', 'peers', 'services')]))
            activity.append(n)
        print('Расчёт устойчивой проекции...', flush=True)
        self.stable_mean, self.projection = stable_projection(vectors, activity, self.stable_dim,
                                                               self.regularization, self.min_flows)
        raw_train = self._base(self.cache['train'])
        projected = (raw_train - self.stable_mean) @ self.projection
        self.stable_center = projected.mean(axis=0)
        self.stable_scale = max(np.sqrt(np.mean(np.sum((projected - self.stable_center) ** 2, axis=1))), 1e-12)
        return self

    @staticmethod
    def _base(parts):
        return np.hstack([parts[k] for k in ('behavioral', 'peers', 'services')])

    def embed(self, data, window, hosts=None):
        hosts = list(data.cohort if hosts is None else hosts)
        rows = pd.Index(self.hosts).get_indexer(hosts)
        if (rows < 0).any():
            raise ValueError("хост отсутствует в train-когорте")
        if window not in self.cache:
            w = data.manifest['windows'][window]
            parts, _ = self._raw(data.flows(window), float(w['end']) - float(w['start']))
            self.cache[window] = self._scaled(parts)
        p = self.cache[window]
        stable = ((self._base(p) - self.stable_mean) @ self.projection - self.stable_center) / self.stable_scale
        x = np.hstack([p['behavioral'] * self.behavioral_weight, p['peers'], p['services'],
                       stable * self.stable_weight])[rows]
        return pd.DataFrame(x, index=pd.Index(hosts, name='host_id')).add_prefix('emb_')


def tune(model, data, weights=(0.0, 0.5, 1.0, 2.0), behavioral_weights=(2.0, 3.0)):
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for wb in behavioral_weights:
            for ws in weights:
                model.behavioral_weight, model.stable_weight = wb, ws
                model.embed(data, 'train').to_csv(root / 'train.csv')
                model.embed(data, 'validation', data.evaluable('validation')).to_csv(root / 'validation.csv')
                score, tasks = submit.validation_score(root / 'train.csv', root / 'validation.csv')
                if not np.isfinite(score) or len(tasks) != 4:
                    raise ValueError('неполная оценка validation: нужны все четыре задачи')
                rows.append({'behavioral_weight': wb, 'stable_weight': ws, 'score': score, 'tasks': tasks})
                print(f'behavioral={wb:g} stable={ws:g}: validation={score:.2f}', flush=True)
    best = max(rows, key=lambda x: x['score'])
    model.behavioral_weight, model.stable_weight = best['behavioral_weight'], best['stable_weight']
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True, help='новая папка для локальных результатов')
    ap.add_argument('--tune', action='store_true', help='выбрать веса только на validation общей таблицы')
    ap.add_argument('--validation-only', action='store_true', help='обучить и сохранить только train и validation')
    ap.add_argument('--checkpoint', type=Path, help='применить готовую модель без обучения и подбора')
    args = ap.parse_args()
    if args.checkpoint and args.tune:
        ap.error('--checkpoint и --tune несовместимы: test не используется для подбора')
    data = ProcessedData(os.environ.get('PT_DATA', ROOT / 'scoreboard/data/processed'))
    if args.tune:
        official = ROOT / 'scoreboard/data/processed'
        def digest(path):
            return hashlib.sha256(path.read_bytes()).digest()
        if (digest(data.root / 'cohort.csv') != digest(official / 'cohort.csv') or
                data.manifest['windows'] != json.loads((official / 'manifest.json').read_text())['windows']):
            raise ValueError('--tune использует общие эталоны: нужны их когорта и манифест окон')
    if args.output.exists():
        raise FileExistsError(f'{args.output} уже существует; укажите новую папку')
    model = TemporalStableHybrid.load(args.checkpoint) if args.checkpoint else TemporalStableHybrid().fit(data)
    if model.hosts != data.cohort:
        raise ValueError('когорта checkpoint отличается от train-когорты данных')
    train_window = data.manifest['windows']['train']
    if getattr(model, 'training_window', None) not in (None, [float(train_window['start']), float(train_window['end'])]):
        raise ValueError('train-окно checkpoint отличается от train-окна данных')
    rows = tune(model, data) if args.tune else []
    args.output.mkdir(parents=True)
    jobs = [('train', data.cohort, 'embeddings.csv')]
    if not args.validation_only:
        jobs += [(w, data.evaluable(w), submit.window_file(w)) for w in data.test_windows]
    if 'validation' in data.windows:
        jobs += [('validation', data.evaluable('validation'), 'validation.csv')]
    for window, hosts, filename in jobs:
        model.embed(data, window, hosts).to_csv(args.output / filename, float_format='%.8g')
    (args.output / 'SHARED_COORDINATES').touch()
    model.cache.clear()
    joblib.dump(model.__dict__, args.output / 'model.joblib', compress=3)
    (args.output / 'params.json').write_text(json.dumps({'splits': model.splits, 'dim': model.dim,
        'stable_dim': model.stable_dim, 'regularization': model.regularization, 'min_flows': model.min_flows,
        'behavioral_weight': model.behavioral_weight, 'stable_weight': model.stable_weight,
        'tuned_on': 'validation' if args.tune else None, 'tuning': rows}, ensure_ascii=False, indent=2) + '\n')
    print(f'Векторы записаны в {args.output}; для оценки нужна карточка метода и общие эталоны.')


if __name__ == '__main__':
    main()
