"""Направленный контрастный GraphSAGE; PyTorch — опциональная зависимость эксперимента."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    dim: int = 64
    hidden: int = 128
    layers: int = 2
    graph: bool = True
    directed: bool = True
    hub_idf: bool = True
    contrastive: bool = True
    false_negative_mask: bool = True
    threshold: float = .95
    temperature: float = .1
    link_weight: float = .3
    topk: int = 64
    edge_dropout: float = .2
    feature_dropout: float = .1
    batch_size: int = 512
    link_batch: int = 4096
    negatives: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50

    def dictionary(self):
        return asdict(self)


def topk_rows(matrix, k):
    matrix = matrix.tocsr().copy()
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row+2]
        if end - start > k:
            order = np.lexsort((matrix.indices[start:end], -matrix.data[start:end]))
            matrix.data[start:end][order[k:]] = 0
    matrix.eliminate_zeros()
    return matrix


def prepare_graph(counts, idf, config):
    if not config.directed:
        counts = counts + counts.T
    idf = idf if config.hub_idf else np.ones(len(idf), dtype=np.float32)
    result = []
    for matrix in (counts, counts.T):
        w = matrix.tocsr().astype(np.float32)
        w.data = np.log1p(w.data)
        result.append(topk_rows(w @ sp.diags(np.asarray(idf, dtype=np.float32), dtype=np.float32), config.topk))
    return result


def norm_adj(matrix, drop=0., rng=None):
    matrix = matrix.copy()
    if drop:
        matrix.data *= rng.random(matrix.nnz) >= drop
        matrix.eliminate_zeros()
    sums = np.asarray(matrix.sum(1)).ravel()
    matrix = (sp.diags(1 / np.maximum(sums, 1e-12)) @ matrix).tocoo()
    return torch.sparse_coo_tensor(np.vstack([matrix.row, matrix.col]), matrix.data.astype(np.float32), matrix.shape,
                                  check_invariants=True).coalesce()


class Layer(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.self_map = nn.Linear(hidden, hidden)
        self.out_map = nn.Linear(hidden, hidden, bias=False)
        self.in_map = nn.Linear(hidden, hidden, bias=False)
        self.norm, self.dropout = nn.LayerNorm(hidden), nn.Dropout(dropout)

    def forward(self, h, out, incoming):
        update = self.self_map(h) + self.out_map(out) + self.in_map(incoming)
        return h + self.dropout(self.norm(F.relu(update)))


class DCSAGE(nn.Module):
    def __init__(self, inputs, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        self.input = nn.Sequential(nn.Linear(inputs, c.hidden), nn.LayerNorm(c.hidden))
        self.layers = nn.ModuleList([Layer(c.hidden, c.feature_dropout) for _ in range(c.layers)])
        self.skip = nn.Sequential(nn.Linear(inputs, 256), nn.ReLU(), nn.Linear(256, c.dim))
        self.graph_map = nn.Linear(c.hidden, c.dim)
        nn.init.zeros_(self.graph_map.weight)
        nn.init.zeros_(self.graph_map.bias)
        if not c.graph:
            for module in (self.input, self.layers, self.graph_map):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(self, x, outgoing=None, incoming=None):
        own = self.skip(x)
        if self.config.graph:
            h = self.input(x)
            for layer in self.layers:
                h = layer(h, torch.sparse.mm(outgoing, h), torch.sparse.mm(incoming, h))
            graph = self.graph_map(h)
        else:
            graph = torch.zeros_like(own)
        return F.normalize(own + graph, dim=-1), own, graph


class Decoder(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.source = nn.Linear(dim, rank, bias=False)
        self.destination = nn.Linear(dim, rank, bias=False)
        self.beta = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, source, destination, log_degree):
        return (self.source(source) * self.destination(destination)).sum(-1) + self.beta * log_degree + self.bias


def info_nce(a, b, profile, config):
    logits = a @ b.T / config.temperature
    if config.false_negative_mask:
        p = F.normalize(profile, dim=-1)
        false = (p @ p.T) > config.threshold
        false.fill_diagonal_(False)
        logits = logits.masked_fill(false, -1e9)
    target = torch.arange(len(a))
    return .5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


class LinkSampler:
    """Новые позитивы — половина; негативы исключают self и все реальные рёбра S2."""
    def __init__(self, first, second, cohort, config):
        self.config = config
        n = first.shape[0]
        self.n = n
        allowed = np.zeros(n, dtype=bool)
        allowed[cohort] = True
        first, second = first.tocoo(), second.tocoo()
        first_codes = first.row.astype(np.int64) * n + first.col
        codes = second.row.astype(np.int64) * n + second.col
        self.positive_set = set(codes)
        keep = allowed[second.row] & allowed[second.col]
        self.edges = np.column_stack([second.row[keep], second.col[keep]])
        new = ~np.isin(codes[keep], first_codes)
        self.new, self.old = self.edges[new], self.edges[~new]
        if not len(self.edges):
            raise ValueError("Нет рёбер между хостами когорты в S2")
        degree = np.asarray(first.tocsr().getnnz(axis=0) + first.tocsr().getnnz(axis=1)).ravel().astype(float)
        self.distribution = np.maximum(degree, 1) ** .75 * allowed
        self.distribution /= self.distribution.sum()
        self.log_degree = torch.tensor(np.log1p(first.tocsr().getnnz(axis=0)), dtype=torch.float32)

    def sample(self, rng):
        size = self.config.link_batch
        a = self.new if len(self.new) else self.edges
        b = self.old if len(self.old) else self.edges
        positives = np.vstack([a[rng.integers(len(a), size=size//2)], b[rng.integers(len(b), size=size-size//2)]])
        negatives = rng.choice(self.n, (size, self.config.negatives), p=self.distribution)
        def invalid():
            source = np.repeat(positives[:, 0], self.config.negatives)
            target = negatives.ravel()
            bad = np.fromiter((int(u)*self.n + int(v) in self.positive_set for u, v in zip(source, target)), bool, len(source))
            return (bad | (source == target)).reshape(negatives.shape)
        for _ in range(100):
            bad = invalid()
            if not bad.any():
                break
            negatives[bad] = rng.choice(self.n, int(bad.sum()), p=self.distribution)
        else:
            raise ValueError("Не удалось набрать настоящие отрицательные связи")
        return torch.from_numpy(positives.astype(np.int64)), torch.from_numpy(negatives.astype(np.int64))


def link_loss(z, decoder, positive, negative, log_degree):
    source, target = positive.T
    pos = decoder(z[source], z[target], log_degree[target])
    neg = decoder(z[source].repeat_interleave(negative.shape[1], 0), z[negative.ravel()], log_degree[negative.ravel()])
    return F.binary_cross_entropy_with_logits(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))


@torch.no_grad()
def infer(model, x, weights):
    model.eval()
    return tuple(value.numpy() for value in model(torch.tensor(x, dtype=torch.float32), *(norm_adj(w) for w in weights)))


def train_epoch(model, decoder, optimizer, x1, x2, w1, w2, positives, profile, sampler, rng):
    model.train()
    c = model.config
    optimizer.zero_grad(set_to_none=True)
    x1 = F.dropout(x1, p=c.feature_dropout, training=True)
    x2 = F.dropout(x2, p=c.feature_dropout, training=True)
    z1, own, graph = model(x1, *(norm_adj(w, c.edge_dropout, rng) for w in w1))
    z2, _, _ = model(x2, *(norm_adj(w, c.edge_dropout, rng) for w in w2))
    # Один полнографный шаг за эпоху; все активные хосты проходят батчи InfoNCE.
    nce = torch.zeros(())
    if c.contrastive:
        order = rng.permutation(positives)
        batches = [order[i:i+c.batch_size] for i in range(0, len(order), c.batch_size)]
        for batch in batches:
            nce = nce + info_nce(z1[batch], z2[batch], profile[batch], c) * len(batch) / len(positives)
    link = torch.zeros(())
    if c.link_weight:
        pos, neg = sampler.sample(rng)
        link = link_loss(z1, decoder, pos, neg, sampler.log_degree)
    loss = nce + c.link_weight * link
    if not torch.isfinite(loss):
        raise FloatingPointError("DC-SAGE: нечисловая потеря")
    loss.backward()
    nn.utils.clip_grad_norm_([*model.parameters(), *decoder.parameters()], 1.)
    optimizer.step()
    with torch.no_grad():
        sample = z1[positives[:512]]
        cosine = (sample.sum(0).square().sum() - len(sample)) / max(len(sample)*(len(sample)-1), 1)
        ratio = (graph[positives].norm(dim=1) / own[positives].norm(dim=1).clamp_min(1e-9)).mean()
    return {"loss": float(loss.detach()), "nce": float(nce.detach()), "link": float(link.detach()),
            "mean_cosine": float(cosine), "std_coordinates": float(sample.std(0).mean()), "graph_own_ratio": float(ratio)}
