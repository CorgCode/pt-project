"""DC-SAGE между днями поверх общего многодневного представления признаков.

Адаптация: граф ограничен фиксированной когортой, вход дополнен компонентами
temporal_encoder, выход 128. Это не буквальное повторение 95→64 из спецификации.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize

from experiments import multiday_dc_core as dc
from experiments.multiday_gain import MultiDayEncoder, RECIPES, check_chronology, evaluate
from experiments.multiday_roles import RoleMixin
from experiments.overnight.model import T
from experiments.overnight.search import safe_to_continue
from pipeline.data import ProcessedData


def counts_from_view(view, hosts, vocab):
    x = T.matrix(view["counts"]["peers"], hosts, vocab, log=False)
    targets = pd.Index(hosts).get_indexer([t[2:] if t.startswith("o:") else "" for t in vocab])
    keep = targets >= 0
    x = x[:, keep].tocoo()
    counts = sp.csr_matrix((x.data, (x.row, targets[keep][x.col])), shape=(len(hosts), len(hosts)))
    counts.setdiag(0)
    counts.eliminate_zeros()
    return counts


class MultiDayDC(RoleMixin, MultiDayEncoder):
    def _features(self, view):
        out = super()._features(view)
        out["_counts"] = counts_from_view(view, self.hosts, self.vocab["peers"])
        return out

    def train_graph(self, epochs, seed=17, link_weight=0.3):
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        config = dc.Config(dim=128, epochs=epochs, link_weight=link_weight)
        self.idf = np.log1p(len(self.hosts) / (1 + self._train["_counts"].getnnz(axis=0)))
        self.net = dc.DCSAGE(self._xs[0].shape[1], config)
        decoder = dc.Decoder(config.dim)
        parameters = [*self.net.parameters(), *decoder.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        rng = np.random.default_rng(seed)
        profile = torch.from_numpy(normalize(self._train["services"]).astype(np.float32))
        # Индексы 0,3,... — полные часы разных дней; полчаса используются в словарях/нормировке.
        pairs = [(i, i + 3) for i in range(0, len(self._xs) - 3, 3)]
        if not pairs:
            raise ValueError("Для обучения DC-SAGE между днями нужны хотя бы два дня")
        samplers = {p: dc.LinkSampler(self._training_counts[p[0]], self._training_counts[p[1]],
                                    np.arange(len(self.hosts)), config) for p in pairs}
        weights = {i: dc.prepare_graph(self._training_counts[i], self.idf, config)
                   for i in {v for pair in pairs for v in pair}}
        xs = {i: torch.from_numpy(self._xs[i]) for i in weights}
        for epoch in range(epochs):
            i, j = pairs[epoch % len(pairs)]
            active = np.flatnonzero(self._ok[i] & self._ok[j])
            stats = dc.train_epoch(self.net, decoder, optimizer, xs[i], xs[j], weights[i], weights[j],
                                   active, profile, samplers[(i, j)], rng)
            schedule.step()
            if not all(np.isfinite(v) for v in stats.values()):
                raise FloatingPointError("Нечисловая диагностика DC-SAGE")
            if (epoch + 1) % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, **stats}), flush=True)
                if not safe_to_continue():
                    raise RuntimeError("DC-SAGE остановлен по состоянию машины")
        self._parts_cache = {}
        return self

    def _parts(self, data, window, hosts):
        key = (window, tuple(hosts))
        if key not in self._parts_cache:
            if window == "train":
                features = self._train
            else:
                a, b = T._span(data, window)
                features = self._features(self.view(data.flows(window), self.hosts, b - a))
            weights = dc.prepare_graph(features["_counts"], self.idf, self.net.config)
            neural = dc.infer(self.net, features["input"], weights)[0]
            rows = pd.Index(self.hosts).get_indexer(hosts)
            parts = {"behavioral": self.beh.embed(data, window, hosts).to_numpy(), "neural": neural[rows]}
            parts.update({k: features[k][rows] for k in ("peers", "services", "context", "diffusion")})
            self._parts_cache[key] = {k: T._block(v) for k, v in parts.items()}
        return self._parts_cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--extra", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/multiday_dc"))
    ap.add_argument("--validation-window", default="test_day04")
    ap.add_argument("--reference-dir", type=Path, default=Path("scoreboard/data/reference/test_day04"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = ProcessedData(args.base)
    training = [data, *[ProcessedData(p) for p in args.extra]]
    check_chronology(training, T._span(data, args.validation_window)[0])
    model = MultiDayDC().fit_days(training)
    ref = args.reference_dir
    best = -np.inf
    for epochs, link_weight in [(100, 0.3), (300, 0.3), (100, 0.0)]:
        if not safe_to_continue():
            raise RuntimeError("Недостаточно ресурсов перед DC-SAGE")
        model.train_graph(epochs, link_weight=link_weight)
        for i, recipe in enumerate(RECIPES):
            model.recipe, model.pca_dim = recipe, 384
            model.fit_pca(data)
            name = f"dc_{epochs}_{link_weight}_{i}"
            score, result = evaluate(model, data, args.validation_window, ref, args.out / name, name)
            row = {"name": name, "score": score, "identity": result["metrics"]["identity_top1"],
                   "epochs": epochs, "link_weight": link_weight, "recipe": recipe}
            with (args.out / "trials.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if score > best:
                best = score
                with (args.out / "best.pkl").open("wb") as out:
                    cloudpickle.dump(model, out)
                (args.out / "best.json").write_text(json.dumps(row, indent=2) + "\n")
            gc.collect()


if __name__ == "__main__":
    from experiments.multiday_dc import main as run_main
    run_main()
