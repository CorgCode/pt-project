"""Обучаемый bag-of-traffic: разреженные токены → устойчивый междневный вектор."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cloudpickle
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F

from experiments.multiday.memory import FingerprintHybrid, read_fingerprints
from experiments.overnight.model import T


def hash_sparse(x, buckets, width):
    x = x.tocsr()
    result = sp.csr_matrix((x.data.copy(), buckets[x.indices], x.indptr.copy()), shape=(x.shape[0], width))
    result.sum_duplicates()
    result.eliminate_zeros()
    return normalize(result).astype(np.float32)


def tensor_rows(x, rows):
    part = x[rows].tocsr()
    return (torch.from_numpy(part.indices.astype(np.int64)),
            torch.from_numpy(part.indptr.astype(np.int64)), torch.from_numpy(part.data.astype(np.float32)))


@dataclass
class BagConfig:
    width: int = 131072
    dim: int = 128
    temperature: float = 0.1
    margin: float = 0.1
    dropout: float = 0.15
    seed: int = 17


class BagNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.EmbeddingBag(config.width, config.dim, mode="sum", include_last_offset=True)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.project = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, config.dim * 2),
                                     nn.GELU(), nn.Dropout(0.1), nn.Linear(config.dim * 2, config.dim))

    def forward(self, indices, offsets, weights):
        return F.normalize(self.project(self.embedding(indices, offsets, per_sample_weights=weights)), dim=1)


class BagHead:
    def __init__(self, config, classes):
        self.config = config
        torch.manual_seed(config.seed)
        self.net = BagNetwork(config)
        self.classifier = nn.Parameter(torch.randn(classes, config.dim) * 0.02)
        self.class_projection = torch.from_numpy(np.random.default_rng(73).standard_normal((classes, 128)).astype(np.float32))

    def encode(self, x, mode="neural", batch=256):
        self.net.eval()
        output = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch):
                rows = np.arange(start, min(start + batch, x.shape[0]))
                z = self.net(*tensor_rows(x, rows))
                if mode == "posterior":
                    logits = z @ F.normalize(self.classifier, dim=1).T / self.config.temperature
                    scores, nearest = torch.topk(logits, min(16, logits.shape[1]), dim=1)
                    probability = torch.softmax(scores, dim=1)
                    z = F.normalize((self.class_projection[nearest] * probability[:, :, None]).sum(1), dim=1)
                values = z.numpy()
                values[np.diff(x.indptr)[rows] == 0] = 0
                output.append(values)
        return np.vstack(output)


def train_head(x, hosts, teacher, config, out, checkpoints=(40, 120)):
    torch.set_num_threads(max(1, int(os.environ.get("PT_TORCH_THREADS", "4"))))
    classes, days = len(hosts), x.shape[0] // len(hosts)
    if days * classes != x.shape[0]:
        raise ValueError("Число наблюдений не соответствует обучающим дням и когорте")
    ok = (np.diff(x.indptr) > 0).reshape(days, classes)
    active = np.flatnonzero(ok.sum(axis=0) >= 2)
    if not len(active):
        raise ValueError("Нет хостов с наблюдениями хотя бы в двух днях")
    head = BagHead(config, classes)
    optimizer = torch.optim.AdamW([*head.net.parameters(), head.classifier], lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(checkpoints))
    teacher = torch.from_numpy(normalize(teacher).astype(np.float32))
    rng = np.random.default_rng(config.seed)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, max(checkpoints) + 1):
        head.net.train()
        order, losses = rng.permutation(active), []
        for offset in range(0, len(order), 256):
            labels = order[offset:offset+256]
            pairs = np.array([rng.choice(np.flatnonzero(ok[:, h]), 2, replace=False) for h in labels])
            rows = np.concatenate([pairs[:, 0] * classes + labels, pairs[:, 1] * classes + labels])
            indices, pointers, weights = tensor_rows(x, rows)
            weights = weights * (torch.rand_like(weights) >= config.dropout) / (1 - config.dropout)
            z = head.net(indices, pointers, weights)
            targets = torch.from_numpy(np.tile(labels, 2))
            logits = z @ F.normalize(head.classifier, dim=1).T / config.temperature
            logits[torch.arange(len(targets)), targets] -= config.margin / config.temperature
            classification = F.cross_entropy(logits, targets)
            a, b = z[:len(labels)], z[len(labels):]
            pair_logits = a @ b.T / 0.1
            positive = torch.arange(len(labels))
            contrast = 0.5 * (F.cross_entropy(pair_logits, positive) + F.cross_entropy(pair_logits.T, positive))
            t = teacher[torch.from_numpy(labels)]
            relation = F.mse_loss(a @ a.T, t @ t.T)
            loss = classification + 0.25 * contrast + 0.1 * relation
            if not torch.isfinite(loss):
                raise FloatingPointError("Нечисловая потеря bag-of-traffic")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*head.net.parameters(), head.classifier], 3)
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        if epoch % 10 == 0:
            print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "config": asdict(config)}), flush=True)
        if epoch in checkpoints:
            with (out / f"epoch{epoch}.pkl").open("wb") as stream:
                cloudpickle.dump(head, stream)
    (out / "complete.json").write_text(json.dumps({"epochs": list(checkpoints), "config": asdict(config)}) + "\n")


class BagHybrid(FingerprintHybrid):
    def __init__(self, base, bank, cache, buckets, head):
        super().__init__(base, bank, cache)
        self.buckets, self.head, self.mode = buckets, head, "neural"

    def embed(self, data, window, hosts):
        import pandas as pd
        anchor = self.base.embed(data, window, hosts)
        if not self.weight:
            return anchor
        key = (str(data.root), window, tuple(hosts), self.mode)
        if key not in self._encoded:
            rows = pd.Index(self.hosts).get_indexer(hosts)
            if (rows < 0).any():
                raise ValueError("Хост вне обучающей когорты")
            if window == "train":
                x = self.bank.prototype[rows]
            else:
                counts = read_fingerprints(data, window, self.hosts, self.cache)
                x = self.bank.transform(counts)[rows]
            hashed = hash_sparse(x, self.buckets, self.head.config.width)
            self._encoded[key] = T._block(self.head.encode(hashed, self.mode))
        vectors = np.hstack([T._block(anchor.to_numpy()), self._encoded[key] * self.weight])
        return pd.DataFrame(vectors, index=anchor.index).add_prefix("emb_")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--hosts", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    hosts = json.loads(args.hosts.read_text())
    config = BagConfig(dim=args.dim, margin=0.1 if args.dim == 128 else 0.15,
                       temperature=0.1 if args.dim == 128 else 0.07)
    train_head(sp.load_npz(args.features), hosts, np.load(args.teacher), config, args.out)


if __name__ == "__main__":
    main()
