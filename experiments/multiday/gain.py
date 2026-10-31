"""Обучение между днями; подбор на дне 04, день 05 остаётся отложенным.

Все потоки читаются по одному окну. В памяти остаются агрегированные признаки,
а не объединённые сырые потоки нескольких дней.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cloudpickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize

from evaluation import scoreboard, submit
from evaluation.evaluate import run
from experiments.overnight.model import T
from experiments.overnight.search import safe_to_continue
from methods.baselines.temporal import BehavioralBaseline
from pipeline.data import ProcessedData


def combine_views(views: list[dict]) -> dict:
    """Объединить равные по длительности часы: суммы токенов, средняя интенсивность."""
    return {
        "counts": {k: pd.concat([v["counts"][k] for v in views]).groupby(level=["h", "t"]).sum()
                   for k in T.KINDS},
        "activity": np.log1p(np.mean([np.expm1(v["activity"]) for v in views], axis=0)),
        "flows": np.sum([v["flows"] for v in views], axis=0),
    }


def check_chronology(training: list[ProcessedData], target_start: float) -> None:
    intervals = sorted(T._span(d, "train") for d in training)
    if any(end > target_start for _, end in intervals):
        raise ValueError("Обучающие потоки должны предшествовать validation/test")
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Обучающие окна пересекаются")
    if len({b - a for a, b in intervals}) != 1:
        raise ValueError("Ожидаются обучающие окна одинаковой длительности")


def streaming_cca(xs, ok):
    """Статистики CCA без конкатенации всех пар дней в огромную матрицу."""
    dim = xs[0].shape[1]
    xx, xy, total = np.zeros((dim, dim)), np.zeros((dim, dim)), np.zeros(dim)
    n = 0
    for i, j in itertools.combinations(range(len(xs)), 2):
        active = ok[i] & ok[j]
        a, b = xs[i][active].astype(float), xs[j][active].astype(float)
        if not len(a):
            continue
        xx += a.T @ a + b.T @ b
        xy += a.T @ b + b.T @ a
        total += a.sum(0) + b.sum(0)
        n += 2 * len(a)
    if not n:
        raise ValueError("Между днями нет активных общих хостов")
    mean = total / n
    caa = xx / n - np.outer(mean, mean) + T.CCA_REG * np.eye(dim)
    cab = xy / n - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(caa)
    whitening = vectors @ np.diag(values ** -0.5) @ vectors.T
    cross = whitening @ cab @ whitening
    values, vectors = np.linalg.eigh((cross + cross.T) / 2)
    order = np.argsort(values)[::-1][:T.CCA_DIM]
    return whitening @ vectors[:, order], np.clip(values[order], 0, None), mean


class MeanBehavior:
    def __init__(self, base: BehavioralBaseline, features: pd.DataFrame):
        self.base, self.features = base, features

    def embed(self, data, window, hosts):
        if window == "train":
            return self.features.loc[hosts]
        return self.base.embed(data, window, hosts)


class MultiDayEncoder(T.TemporalEncoder):
    """Общие словари по дням и обучение узнавать хост в разных днях/получасах."""

    def view(self, flows, hosts, seconds):
        return T.view(flows, hosts, seconds)

    def fit_days(self, training: list[ProcessedData]):
        self._evaluation_cache, self._behavior_cache, self._parts_cache = {}, {}, {}
        cache_parent = Path(os.environ.get("PT_VIEW_CACHE_DIR", Path.cwd() / "runs"))
        cache_parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="multiday-views-", dir=cache_parent) as tmp:
            return self._fit_days_cached(training, Path(tmp))

    def _fit_days_cached(self, training, cache):
        first = training[0]
        self.training_sources = [str(d.root.resolve()) for d in training]
        self.hosts = list(first.cohort)
        behavior = BehavioralBaseline().fit(first)
        aggregate, paths, profiles, masks = None, [], [], []
        for day_index, day in enumerate(training):
            if not safe_to_continue():
                raise RuntimeError("Недостаточно ресурсов для следующего обучающего дня")
            day.cohort = list(self.hosts)
            flows = day.training_flows()
            start, end = T._span(day, "train")
            full = self.view(flows, self.hosts, end - start)
            views = [full]
            for a, b in zip(np.linspace(start, end, 3)[:-1], np.linspace(start, end, 3)[1:]):
                mask = (flows.ts >= a) & (flows.ts < b) & ((flows.ts + flows.duration) < b)
                views.append(self.view(flows.loc[mask], self.hosts, b - a))
            for i, view in enumerate(views):
                path = cache / f"view_{len(paths)}.pkl"
                with path.open("wb") as file:
                    pickle.dump(view, file)
                paths.append(path)
                masks.append(view["flows"] >= (20 if i == 0 else 10))
            if aggregate is None:
                aggregate = full
            else:
                aggregate = {
                    "counts": {k: aggregate["counts"][k].add(full["counts"][k], fill_value=0)
                               for k in T.KINDS},
                    "activity": np.log1p((np.expm1(aggregate["activity"]) * day_index
                                            + np.expm1(full["activity"])) / (day_index + 1)),
                    "flows": aggregate["flows"] + full["flows"],
                }
            if day.root == first.root:
                features = behavior.embed(first, "train", self.hosts)
            else:
                # _features использует только flows(window); train этого дня преобразуется
                # как новое наблюдение через преобразование, обученное на первом дне.
                class Observation:
                    cohort = self.hosts

                    def flows(self, window):
                        return flows
                features = behavior.embed(Observation(), "observation", self.hosts)
            profiles.append(features.to_numpy())
            del flows, views, full, view
            gc.collect()
            print(f"Признаки дня {day.root}: готовы", flush=True)
        full = aggregate
        self.beh = MeanBehavior(behavior, pd.DataFrame(np.mean(profiles, axis=0),
                                                     index=pd.Index(self.hosts, name="host_id"),
                                                     columns=features.columns))
        self.vocab = {k: T.vocabulary(full["counts"][k], T.CONTEXT_VOCAB if k == "context" else None)
                      for k in T.KINDS}
        self.models = {}
        for kind in T.KINDS:
            x = T.matrix(full["counts"][kind], self.hosts, self.vocab[kind])
            tfidf = TfidfTransformer().fit(x)
            dim = min(T.SVD_DIM, x.shape[0] - 1, x.shape[1] - 1)
            seed = T.SEED if kind == "context" else 0
            self.models[kind] = tfidf, TruncatedSVD(dim, random_state=seed).fit(normalize(tfidf.transform(x)))
        if hasattr(self, "scaler"):
            del self.scaler
        scaler, raw_paths, self._training_counts = StandardScaler(), [], []
        for path in paths:
            with path.open("rb") as file:
                view = pickle.load(file)
            raw = self._features(view)
            scaler.partial_fit(raw["raw"])
            raw_path = path.with_suffix(".npy")
            np.save(raw_path, raw["raw"].astype(np.float32))
            raw_paths.append(raw_path)
            self._training_counts.append(raw.get("_counts"))
            del view, raw
        self.scaler = scaler
        self._train = self._features(full)
        self._xs = [np.clip(self.scaler.transform(np.load(p)), -8, 8).astype(np.float32) for p in raw_paths]
        # Полчаса требуют вдвое меньше событий, чем целый час.
        self._ok = np.stack(masks)
        self._active = np.flatnonzero(self._ok.sum(0) >= 2)
        self.splits = len(paths)
        return self

    def _parts(self, data, window, hosts):
        key = (window if window == "train" else (str(data.root), window), tuple(hosts))
        if key not in self._parts_cache:
            if window == "train":
                features = self._train
                behavior = self.beh.embed(data, window, hosts).to_numpy()
            else:
                features, behavior = self.evaluation_features(data, window, hosts)
            rows = pd.Index(self.hosts).get_indexer(hosts)
            parts = {"behavioral": behavior,
                     "neural": self._neural(features["input"])[rows]}
            parts.update({k: features[k][rows] for k in ("peers", "services", "context", "diffusion")})
            self._parts_cache[key] = {k: T._block(v) for k, v in parts.items()}
        return self._parts_cache[key]

    def evaluation_features(self, data, window, hosts):
        # Здесь только входы для вывода. Обучение читает _xs/_train, не эти кэши.
        if not hasattr(self, "_evaluation_cache"):
            self._evaluation_cache, self._behavior_cache = {}, {}
        key = (str(data.root), window)
        if key not in self._evaluation_cache:
            a, b = T._span(data, window)
            self._evaluation_cache[key] = self._features(self.view(data.flows(window), self.hosts, b - a))
        behavior_key = (key, tuple(hosts))
        if behavior_key not in self._behavior_cache:
            self._behavior_cache[behavior_key] = self.beh.embed(data, window, hosts).to_numpy()
        return self._evaluation_cache[key], self._behavior_cache[behavior_key]

    def train_head(self, data, kind: str, epochs: int):
        self.posterior_mix = 0.0
        self.encoder = "cca" if kind == "cca" else "mlp"
        self.epochs = epochs
        if kind == "cca":
            self.proj, self.corr, self.cca_mean = streaming_cca(self._xs[::3], self._ok[::3])
        elif kind == "classifier":
            self._fit_classifier()
        else:
            self._fit_mlp(data, self._xs, self._ok, self._active)
        self._parts_cache = {}
        return self

    def _fit_classifier(self):
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(max(1, int(os.environ.get("PT_TORCH_THREADS", "2"))))
        self.net = T._encoder(self._xs[0].shape[1])
        classifier = torch.nn.Parameter(torch.randn(len(self.hosts), 128) * 0.02)
        optimizer = torch.optim.AdamW([*self.net.parameters(), classifier], lr=1e-3, weight_decay=1e-4)
        xs = [torch.from_numpy(x) for x in self._xs]
        rng = np.random.default_rng(T.SEED)
        teacher = torch.from_numpy(normalize(self.beh.features.to_numpy()).astype(np.float32))
        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(self._active)
            for off in range(0, len(order), T.BATCH):
                batch = order[off:off + T.BATCH]
                views = [rng.choice(np.flatnonzero(self._ok[:, h])) for h in batch]
                x = torch.stack([xs[v][h] for h, v in zip(batch, views)])
                z = self.net(x)
                labels = torch.from_numpy(batch)
                logits = z @ F.normalize(classifier, dim=1).T / 0.1
                target = teacher[labels] @ teacher[labels].T
                loss = F.cross_entropy(logits, labels) + 0.5 * F.mse_loss(z @ z.T, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*self.net.parameters(), classifier], 5.0)
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Классификатор: эпоха {epoch + 1}/{self.epochs}", flush=True)
        self.classifier = classifier.detach()
        self.class_projection = np.random.default_rng(97).standard_normal((len(self.hosts), 128)).astype(np.float32)

    def _neural(self, x):
        if self.encoder == "cca" and hasattr(self, "cca_mean"):
            z = normalize(((x - self.cca_mean) @ self.proj) * self.corr)
        else:
            z = super()._neural(x)
        mix = getattr(self, "posterior_mix", 0.0)
        if not mix:
            return z
        import torch
        from torch.nn import functional as F
        parts = []
        weights = F.normalize(self.classifier, dim=1)
        projection = torch.from_numpy(self.class_projection)
        with torch.no_grad():
            for off in range(0, len(z), 512):
                logits = torch.from_numpy(z[off:off + 512]) @ weights.T / 0.1
                parts.append((torch.softmax(logits, dim=1) @ projection).numpy())
        # Предсказание класса строится только из трафика; ID запрашиваемого хоста
        # не подаётся в сеть и не используется как индекс таблицы векторов.
        return np.hstack([z, normalize(np.vstack(parts)) * mix])


RECIPES = [
    dict(T.RECIPE),
    dict(T.RECIPE, behavioral=4.5, peers=0.75, neural=3.0, diffusion=0.0),
    dict(T.RECIPE, behavioral=1.0, neural=4.0, context=0.25),
    dict(T.RECIPE, behavioral=0.5, neural=8.0, context=0.25),
]


def restore_training_features(path, data, training, reuse_evaluation_features=False):
    """Повторно использовать только преобразования и наблюдения train, без PCA/выбранной головы."""
    with Path(path).open("rb") as source:
        model = pickle.load(source)
    if list(model.hosts) != list(data.cohort) or len(model._xs) != 3 * len(training):
        raise ValueError("Когорта или число обучающих дней общего checkpoint не совпадает")
    expected = [str(d.root.resolve()) for d in training]
    if hasattr(model, "training_sources") and model.training_sources != expected:
        raise ValueError("Источники обучающих признаков не совпадают")
    model.recipe = dict(T.RECIPE)
    model.pca, model.posterior_mix, model._parts_cache = None, 0.0, {}
    if not reuse_evaluation_features:
        model._evaluation_cache, model._behavior_cache = {}, {}
    for name in ["net", "proj", "corr", "cca_mean", "classifier", "class_projection"]:
        model.__dict__.pop(name, None)
    return model


def counts_from_view(view, hosts, vocab):
    x = T.matrix(view["counts"]["peers"], hosts, vocab, log=False)
    targets = pd.Index(hosts).get_indexer([t[2:] if t.startswith("o:") else "" for t in vocab])
    keep = targets >= 0
    x = x[:, keep].tocoo()
    counts = sp.csr_matrix((x.data, (x.row, targets[keep][x.col])), shape=(len(hosts), len(hosts)))
    counts.setdiag(0)
    counts.eliminate_zeros()
    return counts


def evaluate(model, data, window: str, ref: Path, out: Path, name: str, n_boot=0) -> tuple[float, dict]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for w, hosts, filename in [("train", data.cohort, "embeddings.csv"),
                               (window, data.evaluable(window), "later.csv")]:
        path = out / filename
        model.embed(data, w, hosts).to_csv(path, float_format="%.5g", index_label="host_id")
        files.append(path)
    result = run(files[0], ref, name, n_boot=n_boot,
                 fit_reference_dir=Path("scoreboard/data/reference/validation"))
    submit.add_identity(result, files[0], files[1], ref, shared=True, n_boot=n_boot)
    cols = {s["metric"]: s["title"] for s in result["summary"]}
    reference_manifest = json.loads((ref / "manifest.json").read_text())
    ideal = reference_manifest.get("score_neighbors_ideal")
    previous = scoreboard.NEIGHBORS_IDEAL.get(window)
    try:
        if ideal is not None:
            scoreboard.NEIGHBORS_IDEAL[window] = float(ideal)
            result["score_neighbors_ideal"] = float(ideal)
        score = scoreboard._one_score(result, cols, window=window)
    finally:
        if previous is None:
            scoreboard.NEIGHBORS_IDEAL.pop(window, None)
        else:
            scoreboard.NEIGHBORS_IDEAL[window] = previous
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return score, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--extra", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/multiday_gain"))
    ap.add_argument("--validation-window", default="test_day04")
    ap.add_argument("--reference-dir", type=Path, default=Path("scoreboard/data/reference/test_day04"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = ProcessedData(a.base)
    training = [data, *[ProcessedData(p) for p in a.extra]]
    check_chronology(training, T._span(data, a.validation_window)[0])
    protocol = {"training": [str(d.root) for d in training],
                "selection": {"window": a.validation_window, "interval": T._span(data, a.validation_window)},
                "final_holdout": "not read by this command",
                "baseline": "temporal_encoder, one hour day02", "target_gain": 10, "target_score": 80,
                "warning": "day04 is development data; its gains are not final test gains"}
    (a.out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    ref = a.reference_dir
    saved_baseline = a.out / "baseline/result.json"
    if saved_baseline.exists():
        baseline_result = json.loads(saved_baseline.read_text())
        cols = {s["metric"]: s["title"] for s in baseline_result["summary"]}
        previous = scoreboard.NEIGHBORS_IDEAL.get(a.validation_window)
        ideal = baseline_result.get("score_neighbors_ideal")
        if ideal is not None:
            scoreboard.NEIGHBORS_IDEAL[a.validation_window] = ideal
        base_score = scoreboard._one_score(baseline_result, cols, window=a.validation_window)
        if previous is None:
            scoreboard.NEIGHBORS_IDEAL.pop(a.validation_window, None)
        else:
            scoreboard.NEIGHBORS_IDEAL[a.validation_window] = previous
    else:
        baseline = T.TemporalEncoder().fit(data)
        base_score, _ = evaluate(baseline, data, a.validation_window, ref, a.out / "baseline", "baseline")
        with (a.out / "baseline.pkl").open("wb") as f:
            pickle.dump(baseline, f)
        del baseline
    print(f"Исходная модель на validation: {base_score:.3f}", flush=True)
    gc.collect()
    if not safe_to_continue():
        raise RuntimeError("Недостаточно ресурсов перед обучением между днями")
    model = MultiDayEncoder().fit_days(training)
    rows, best = [], -np.inf
    for kind, epochs in [("cca", 0), ("contrastive", 20), ("contrastive", 80), ("classifier", 60)]:
        if not safe_to_continue():
            raise RuntimeError("Подбор остановлен по состоянию машины")
        model.train_head(data, kind, epochs)
        for i, recipe in enumerate(RECIPES):
            if not safe_to_continue():
                raise RuntimeError("Подбор остановлен по состоянию машины")
            model.recipe, model.pca_dim = recipe, 384
            model.fit_pca(data)
            name = f"{kind}_{epochs}_{i}"
            score, result = evaluate(model, data, a.validation_window, ref, a.out / name, name)
            row = {"name": name, "kind": kind, "epochs": epochs, "recipe": recipe, "score": score,
                   "gain": score - base_score, "identity": result["metrics"]["identity_top1"]}
            rows.append(row)
            with (a.out / "trials.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if score > best:
                best = score
                with (a.out / "best.pkl").open("wb") as f:
                    pickle.dump(model, f)
                (a.out / "best.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
            time.sleep(20)
    rows.sort(key=lambda r: -r["score"])
    report = ["# Подбор между днями", "", "Это validation. Финальный test не использован.", "",
              f"Исходная модель: {base_score:.3f}.", "", "| модель | балл | прирост | Identity |",
              "|---|---|---|---|"]
    report += [f"| {r['name']} | {r['score']:.3f} | {r['gain']:+.3f} | {r['identity']:.3f} |" for r in rows]
    (a.out / "VALIDATION.md").write_text("\n".join(report) + "\n")
    print("Подбор завершён; финальный день 05 ещё не оценивался", flush=True)


if __name__ == "__main__":
    # Сохранённая модель должна ссылаться на импортируемый модуль, а не __main__.
    from experiments.multiday.gain import main as run_main
    run_main()
