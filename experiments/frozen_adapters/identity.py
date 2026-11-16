"""Ridge, поиск редких устойчивых контактов и gate на строго прошлых OOF-окнах."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize


def unit(x, fallback=None):
    x = np.asarray(x, float).copy()
    norm = np.linalg.norm(x, axis=1)
    bad = norm < 1e-12
    if bad.any():
        x[bad] = fallback if fallback is not None else np.eye(1, x.shape[1])[0]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class IdentityAdapter:
    def __init__(self, source, min_events=300, svd_dim=128, alpha=10.0):
        self.source = np.asarray(source, float).copy()
        self.gallery = unit(self.source)
        self.min_events, self.svd_dim, self.alpha = min_events, svd_dim, alpha

    @staticmethod
    def rows(observation):
        rate = 3600 / (observation.window.end - observation.window.start)
        return [{k: np.log1p(v * rate) for k, v in row.items()} for row in observation.counts]

    def matrix(self, observation):
        x = self.vocabulary.transform(self.rows(observation)).tocsr()
        return normalize(x.multiply(self.idf).tocsr())

    def fit(self, observations):
        if not observations:
            raise ValueError('нет обучающих окон')
        self.training_end = max(o.window.end for o in observations)
        rows = [row for obs in observations for row in self.rows(obs)]
        self.vocabulary = DictVectorizer(sparse=True, sort=True).fit(rows)
        del rows
        raw = [self.vocabulary.transform(self.rows(o)).tocsr() for o in observations]
        population = sum(raw, sp.csr_matrix(raw[0].shape))
        df = np.asarray((population > 0).sum(0)).ravel()
        self.idf = np.log((len(self.source) + 1) / (df + 1)) + 1
        matrices = [normalize(x.multiply(self.idf).tocsr()) for x in raw]
        active = [o.events >= self.min_events for o in observations]
        if not any(a.any() for a in active):
            raise ValueError('нет активных обучающих хостов')
        train = sp.vstack([x[a] for x, a in zip(matrices, active)], format='csr')
        n = min(self.svd_dim, train.shape[0] - 1, train.shape[1] - 1)
        if n < 1:
            raise ValueError('недостаточно признаков для SVD')
        self.svd = TruncatedSVD(n, random_state=17).fit(train)
        numeric = np.vstack([o.numeric[a] for o, a in zip(observations, active)])
        self.scaler = StandardScaler().fit(numeric)
        features = np.hstack([self.svd.transform(train), self.scaler.transform(numeric)])
        targets = np.vstack([self.gallery[a] for a in active])
        self.ridge = Ridge(alpha=self.alpha).fit(features, targets)
        # Устойчивость — число дней присутствия токена у данного хоста.
        day_presence, day_active = [], []
        for day in sorted({o.window.day for o in observations}):
            indexes = [i for i,o in enumerate(observations) if o.window.day == day]
            seen = sum([(raw[i] > 0).astype(float).multiply(active[i][:,None]) for i in indexes],
                       sp.csr_matrix(raw[0].shape))
            day_presence.append((seen > 0).astype(float))
            day_active.append(np.any([active[i] for i in indexes],axis=0))
        presence = sum(day_presence, sp.csr_matrix(raw[0].shape))
        denominator = np.maximum(np.sum(day_active, axis=0), 1)
        persistence = sp.diags(1 / denominator) @ presence
        total = sum([x.multiply(a[:, None]) for x, a in zip(matrices, active)], sp.csr_matrix(raw[0].shape))
        self.profiles = normalize(total.multiply(persistence).tocsr())
        self.gate = None
        return self

    def ridge_prediction(self, observation):
        x = self.matrix(observation)
        features = np.hstack([self.svd.transform(x), self.scaler.transform(observation.numeric)])
        return unit(self.ridge.predict(features), self.gallery[0]), x

    def components(self, observation, temperature=0.05, neighbors=20):
        ridge, x = self.ridge_prediction(observation)
        rare = np.empty_like(ridge)
        quality = np.zeros((len(ridge), 6))
        k = min(neighbors, len(self.gallery))
        for start in range(0, len(ridge), 128):
            stop = min(start + 128, len(ridge))
            similarity = (x[start:stop] @ self.profiles.T).toarray()
            top = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
            values = np.take_along_axis(similarity, top, axis=1)
            weights = np.exp((values - values[:, :1]) / temperature) * (values > 0)
            weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
            prediction = np.einsum('bk,bkd->bd', weights, self.gallery[top])
            empty = weights.sum(1) == 0
            prediction[empty] = ridge[start:stop][empty]
            rare[start:stop] = unit(prediction, self.gallery[0])
            direct = ridge[start:stop] @ self.gallery.T
            best = np.sort(direct, axis=1)[:, -min(2, len(self.gallery)):]
            quality[start:stop] = np.column_stack([
                np.log1p(observation.events[start:stop]), values[:, 0],
                values[:, 0] - values[:, min(1, k - 1)], best[:, -1] - best[:, 0],
                np.log1p(x[start:stop].getnnz(axis=1)), top[:, 0] == direct.argmax(1)])
        return ridge, rare, quality

    def fit_gate(self, observations):
        ordered = sorted(observations, key=lambda o: o.window.start)
        features, labels, history = [], [], []
        fold_models={}
        for i in range(1, len(ordered)):
            # Только полностью предшествующие окна; нет ни vocabulary, ни IDF будущего fold.
            previous = [o for o in ordered if o.window.day < ordered[i].window.day and o.window.end <= ordered[i].window.start]
            if not previous:
                continue
            day=ordered[i].window.day
            if day not in fold_models:
                fold_models[day]=IdentityAdapter(self.source,self.min_events,self.svd_dim,self.alpha).fit(previous)
            fold=fold_models[day]
            obs = ordered[i]
            ridge, rare, quality = fold.components(obs)
            active = np.flatnonzero(obs.events >= self.min_events)
            targets = np.zeros(len(active))
            for start in range(0, len(active), 128):
                rows = active[start:start + 128]
                a, b = ridge[rows] @ self.gallery.T, rare[rows] @ self.gallery.T
                dot = (ridge[rows] * rare[rows]).sum(1)
                losses = []
                for g in (0.0, 0.25, 0.5, 0.75, 1.0):
                    length = np.sqrt(np.maximum((1-g)**2 + g*g + 2*g*(1-g)*dot, 1e-12))
                    scores = ((1-g)*a + g*b) / length[:, None] / 0.1
                    losses.append(logsumexp(scores, axis=1) - scores[np.arange(len(rows)), rows])
                targets[start:start + len(rows)] = np.argmin(losses, axis=0) / 4
            features.append(quality[active]); labels.append(targets)
            history.append({'query_window': obs.window.name, 'training_end': fold.training_end,
                            'query_start': obs.window.start, 'n_queries': len(active)})
        if not features or not sum(len(y) for y in labels):
            raise ValueError('для gate нужны минимум два непересекающихся train-окна')
        x, y = np.vstack(features), np.concatenate(labels)
        self.gate_scaler = StandardScaler().fit(x)
        x = np.column_stack([np.ones(len(x)), self.gate_scaler.transform(x)])
        def loss(w):
            z = x @ w
            value = np.mean(np.logaddexp(0, z) - y * z) + 0.01 * (w[1:]**2).sum()
            gradient = x.T @ (expit(z)-y) / len(y)
            gradient[1:] += 0.02 * w[1:]
            return value, gradient
        result = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B')
        if not result.success:
            raise RuntimeError(f'gate не сошёлся: {result.message}')
        self.gate = result.x
        self.gate_history = history
        return self

    def predict(self, observation, variant, temperature=0.05):
        ridge, rare, quality = self.components(observation, temperature)
        if variant == 'ridge': return ridge
        if variant == 'rare': return rare
        if variant == 'gate':
            if self.gate is None: raise ValueError('gate не обучен')
            x = np.column_stack([np.ones(len(quality)), self.gate_scaler.transform(quality)])
            g = expit(x @ self.gate)[:, None]
        elif variant.startswith('mix:'):
            g = float(variant.split(':')[1])
        else: raise ValueError('неизвестный адаптер')
        return unit((1-g)*ridge + g*rare, self.gallery[0])
