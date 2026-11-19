"""Локальная проверка MD-TGR: метрика «новые собеседники» без эталонных файлов.

Функция считает ровно то, что делает ``evaluation.partners.partner_ranking`` для
view=«new»: те же кандидаты (партнёры source-окна), то же исключение старых
собеседников и самого хоста, те же top-k соседей и веса-косинусы, тот же средний
ранг при равенстве. Нужна, чтобы видеть тренд на dev-дне, не собирая эталоны.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse

from methods.mdtgr.losses import partner_scores


def new_targets(t: sparse.spmatrix, m: sparse.spmatrix) -> sparse.csr_matrix:
    """Контакты дня, которых не было в source-окне (новые собеседники)."""
    m_bool = m.copy()
    m_bool.data = np.ones_like(m_bool.data)
    new = (t - t.multiply(m_bool)).tocsr()
    new.eliminate_zeros()
    return new


def _mid_ranks(row: np.ndarray, items: np.ndarray) -> np.ndarray:
    """Ранг (1 = лучший) целей в строке; при равенстве — средний ранг, как в метрике."""
    values = row[items]
    greater = (row[None, :] > values[:, None]).sum(1)
    ties = (row[None, :] == values[:, None]).sum(1) - 1
    return 1.0 + greater + ties / 2.0


def new_partner_metrics(emb: np.ndarray, m: sparse.spmatrix, t: sparse.spmatrix,
                        host_col: np.ndarray, *, k: int = 10, top_n: int = 10,
                        batch: int = 512) -> dict:
    """MRR и Recall@top_n по новым собеседникам; emb — (P, d) для пула окна."""
    m, t = m.tocsr(), t.tocsr()
    targets = new_targets(t, m)
    # float64 как в эталонной реализации: иначе редкие равенства косинусов
    # разыгрывались бы иначе, чем в метрике, и dev-число «дрожало» бы на ровном месте
    x = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float64))
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    mt = m.T.tocoo()
    m_t = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([mt.row, mt.col]).astype(np.int64)),
        torch.from_numpy(mt.data.astype(np.float64)),
        (m.shape[1], m.shape[0])).coalesce()
    rr, rec, n_hosts = [], [], 0
    for start in range(0, x.shape[0], batch):
        idx = torch.arange(start, min(start + batch, x.shape[0]))
        scores = partner_scores(x, idx, m_t, k=k).numpy()
        rows = np.arange(len(idx))
        self_pos = host_col[start:start + len(idx)]
        for row, h in zip(rows, self_pos):
            if h >= 0:
                scores[row, h] = -np.inf
            scores[row, m.indices[m.indptr[start + row]:m.indptr[start + row + 1]]] = -np.inf
        for row, h in enumerate(range(start, start + len(idx))):
            items = targets.indices[targets.indptr[h]:targets.indptr[h + 1]]
            if len(items) == 0:
                continue
            ranks = _mid_ranks(scores[row], items)
            rr.append(1.0 / ranks.min())
            rec.append(float((ranks <= top_n).mean()))
            n_hosts += 1
    return {"mrr": float(np.mean(rr)) if rr else None,
            "recall@%d" % top_n: float(np.mean(rec)) if rec else None,
            "n_hosts": n_hosts}
