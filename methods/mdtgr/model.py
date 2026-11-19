"""Модель MD-TGR: три блока представления и одна общая «голова» на день.

Представление хоста в окне — конкатенация трёх блоков:

``anchor``  поведенческий профиль ЭТОГО окна (нормировка по окну). Он повторяет
            публичный поведенческий эталон, поэтому держит «Соседей» и «Пары»
            на уровне потолка, не давая обучаемой части их размыть.
``code``    персональный латентный код хоста, один на всю историю 02–12. Он
            стабилен по построению — это и есть «узнаваемость через время»;
            обучается он партнёрским лоссом, то есть отвечает за то, кого хост
            находит в следующие дни.
``window``  нейросетевой блок окна: поведение в общей системе координат,
            контакты и сервисы через EmbeddingBag и два слоя message passing
            по рёбрам host–host. Он ловит то, что меняется от дня ко дню.

Веса блоков — не обучаемые параметры, а настройки вывода: их можно перебрать
на validation, не переобучая модель (``compose``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.mdtgr.dataset import WindowTensors


class Bag(nn.Module):
    """Взвешенная сумма векторов словаря (аналог TF-IDF-блока, но обучаемый)."""

    def __init__(self, n_tokens: int, dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_tokens, dim, mode="sum")
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, indices: torch.Tensor, offsets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        out = self.embedding(indices, offsets, per_sample_weights=weights)
        return self.dropout(self.norm(out))


class GraphLayer(nn.Module):
    """Слой message passing по ненаправленным рёбрам host–host."""

    def __init__(self, dim: int, *, dropout: float = 0.1):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        msg = self.neigh_lin(h)[src] * weight.unsqueeze(1)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        out = h + F.gelu(self.self_lin(h) + agg)
        return self.dropout(self.norm(out))


class MDTGR(nn.Module):
    """Энкодер окна: поведение + контакты + сервисы + два слоя графа."""

    def __init__(self, n_hosts: int, n_partners: int, n_services: int, behavior_width: int, *,
                 dim: int = 128, code_dim: int = 64, bag_dim: int = 64, service_dim: int = 32,
                 hidden: int = 192, dropout: float = 0.1):
        super().__init__()
        self.config = {"n_hosts": n_hosts, "n_partners": n_partners, "n_services": n_services,
                       "behavior_width": behavior_width, "dim": dim, "code_dim": code_dim,
                       "bag_dim": bag_dim, "service_dim": service_dim, "hidden": hidden,
                       "dropout": dropout}
        self.bag_out = Bag(n_partners, bag_dim, dropout=dropout)
        self.bag_in = Bag(n_partners, bag_dim, dropout=dropout)
        self.bag_service = Bag(n_services, service_dim, dropout=dropout)
        self.input_lin = nn.Sequential(
            nn.Linear(behavior_width + 2 * bag_dim + service_dim, hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.graph1 = GraphLayer(hidden, dropout=dropout)
        self.graph2 = GraphLayer(hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.code = nn.Embedding(n_hosts, code_dim)
        nn.init.normal_(self.code.weight, std=0.02)

    # ------------------------------------------------------------------ прямой проход

    def encode(self, tensors: WindowTensors) -> tuple[torch.Tensor, torch.Tensor]:
        """(window-блок, code-блок) по окну; оба L2-нормированы построчно."""
        x = torch.cat([
            tensors.beh_shared,
            self.bag_out(tensors.out_indices, tensors.out_offsets, tensors.out_weights),
            self.bag_in(tensors.in_indices, tensors.in_offsets, tensors.in_weights),
            self.bag_service(tensors.svc_indices, tensors.svc_offsets, tensors.svc_weights),
        ], dim=1)
        h = self.graph2(self.graph1(self.input_lin(x), tensors.edge_src, tensors.edge_dst,
                                    tensors.edge_weight), tensors.edge_src, tensors.edge_dst,
                        tensors.edge_weight)
        window = F.normalize(self.head(h), dim=1)
        code = F.normalize(self.code.weight, dim=1)
        return window, code

    @staticmethod
    def compose(anchor: torch.Tensor, code: torch.Tensor, window: torch.Tensor,
                weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> torch.Tensor:
        """Итоговый вектор: anchor ⊕ code ⊕ window с весами блоков.

        anchor нормируется здесь, потому что у неактивных хостов он нулевой.
        """
        a = F.normalize(anchor, dim=1, eps=1e-6) * float(weights[0])
        return torch.cat([a, code * float(weights[1]), window * float(weights[2])], dim=1)

    def blocks(self, tensors: WindowTensors, weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
               *, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Готовый эмбеддинг окна (для инференса и метрик)."""
        window, code = self.encode(tensors)
        anchor = tensors.anchor
        if rows is not None:
            anchor, code, window = anchor[rows], code[rows], window[rows]
        return self.compose(anchor, code, window, weights)

    # ------------------------------------------------------------------ сохранение

    def state(self) -> dict:
        return {"config": dict(self.config), "weights": self.state_dict()}

    @classmethod
    def from_state(cls, state: dict) -> "MDTGR":
        model = cls(**state["config"])
        model.load_state_dict(state["weights"])
        model.eval()
        return model
