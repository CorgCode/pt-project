"""Лоссы MD-TGR. Партнёрский лосс повторяет формулу метрики, а не «похожую» задачу.

Метрика «новые собеседники» ранжирует кандидатов c для хоста A по

    s(A, c) = Σ_B w(A, B) · M[B, c],   w(A, B) = max(cos(z_A, z_B), 0) на top-k соседей,

где M — контакты source-окна (train), кандидаты — все партнёры из train, а старые
собеседники A исключены. Поэтому и лосс считает ровно s(A, c) и требует, чтобы
хотя бы один настоящий новый партнёр оказался выше остальных:

    L = logsumexp_c s(A, c) - logsumexp_{c ∈ targets} s(A, c)

Это listwise-суррогат минимального ранга (MRR). Он напрямую двигает то число,
которое потом измеряется, и не требует отрицательной выборки.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


def rows_to_mask(mat: sparse.spmatrix, rows: np.ndarray, n_cols: int, *,
                 device: torch.device | str = "cpu") -> torch.Tensor:
    """Плотная булева маска (len(rows), n_cols) по ненулевым элементам строк матрицы."""
    sub = mat[rows].tocoo()
    mask = torch.zeros((len(rows), n_cols), dtype=torch.bool, device=device)
    if sub.nnz:
        mask[torch.from_numpy(sub.row.astype(np.int64)),
             torch.from_numpy(sub.col.astype(np.int64))] = True
    return mask


def partner_scores(z: torch.Tensor, anchors: torch.Tensor, m_t: torch.Tensor, *,
                   k: int = 10) -> torch.Tensor:
    """s = (W @ M): z — векторы пула (P, d), m_t — sparse (C, P), результат (B, C).

    ``anchors`` — позиции якорей внутри пула. Соседи берутся только из пула
    (на evaluation оцениваемые хосты), как в ``evaluation.partners``.
    """
    sim = z[anchors] @ z.t()                                   # (B, P)
    rows = torch.arange(sim.shape[0], device=sim.device)
    sim[rows, anchors] = -torch.inf                            # сам хост не сосед
    k = min(k, sim.shape[1] - 1)
    top = torch.topk(sim, k, dim=1).indices
    weight = torch.zeros_like(sim)
    weight.scatter_(1, top, torch.gather(sim, 1, top).clamp(min=0))
    return torch.sparse.mm(m_t, weight.t()).t()                 # (B, C)


def partner_listwise_loss(scores: torch.Tensor, targets: torch.Tensor, blocked: torch.Tensor, *,
                          tau: float = 1.0, transform: str = "log1p") -> torch.Tensor:
    """Listwise-лосс по метрике; строки без целей не участвуют.

    ``transform="log1p"`` сжимает тяжёлый хвост счётчиков контактов: ранжирование
    кандидатов от этого не меняется (преобразование монотонно), но softmax перестаёт
    быть одношаговым и даёт устойчивый градиент. Центрирование по строке убирает
    зависимость температуры от «масштаба» конкретного хоста.
    """
    if scores.shape[0] == 0:
        return scores.sum() * 0
    allowed = ~blocked
    s = scores.clamp_min(0)
    if transform == "log1p":
        s = torch.log1p(s)
    count = allowed.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (s * allowed).sum(dim=1, keepdim=True) / count
    floor = -torch.finfo(scores.dtype).max / 4
    s = ((s - mean) / tau).masked_fill(blocked, floor)
    has_target = targets.any(dim=1)
    if not bool(has_target.any()):
        return scores.sum() * 0
    log_all = torch.logsumexp(s, dim=1)
    log_true = torch.logsumexp(s.masked_fill(~targets, floor), dim=1)
    return (log_all - log_true)[has_target].mean()


def link_loss(z: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, *,
              tau: float = 0.1) -> torch.Tensor:
    """BCE на косинусе: повторяет то, что делает обучаемый классификатор связей."""
    if len(positives) == 0 or len(negatives) == 0:
        return z.sum() * 0
    pos = (z[positives[:, 0]] * z[positives[:, 1]]).sum(1) / tau
    neg = (z[negatives[:, 0]] * z[negatives[:, 1]]).sum(1) / tau
    return F.softplus(-pos).mean() + F.softplus(neg).mean()


def temporal_infonce(z_a: torch.Tensor, z_b: torch.Tensor, mask: torch.Tensor, *,
                     tau: float = 0.1) -> torch.Tensor:
    """InfoNCE между двумя днями: тот же хост — положительная пара, остальные — отрицательные."""
    if int(mask.sum()) < 2:
        return z_a.sum() * 0
    a, b = z_a[mask], z_b[mask]
    logits = a @ b.t() / tau
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def sample_negatives(n: int, count: int, generator: torch.Generator, device) -> torch.Tensor:
    """Случайные пары вершин (индексы внутри пула) для link-лосса."""
    if count <= 0 or n < 2:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    left = torch.randint(0, n, (count,), generator=generator, device=device)
    right = torch.randint(0, n - 1, (count,), generator=generator, device=device)
    right = right + (right >= left).long()                     # без петель
    return torch.stack([left, right], dim=1)
