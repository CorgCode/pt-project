"""Срезы окон MD-TGR как тензоры и эпизоды «train → целевой день».

Срез (``methods.mdtgr.shards``) — это матрицы по когорте фиксированного состава:
поведенческий профиль, контакты с партнёрами в обе стороны, сервисы и рёбра
host–host. Здесь они читаются один раз и дальше не пересобираются.

Эпизод обучения — это пара (``train``, день d). Матрица ``M`` берётся из
train-окна и НЕ меняется между эпизодами: ровно так устроена метрика «новые
собеседники» — новые партнёры дня 14 определяются относительно train-окна.
Цель эпизода — контакты дня d, поэтому обучение повторяет механику метрики.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


def _csr(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray,
         shape: tuple[int, int]) -> sparse.csr_matrix:
    """CSR из массивов среза; индексы и данные копируются в нативные типы scipy."""
    mat = sparse.csr_matrix((data.astype(np.float32, copy=False),
                             indices.astype(np.int32, copy=False),
                             indptr.astype(np.int64, copy=False)), shape=shape)
    mat.sort_indices()
    return mat


@dataclass
class WindowShard:
    """Одно окно: всё, что нужно модели и метрике, в порядке когорты."""

    name: str
    behavior: np.ndarray          # (N, F) профиль, стандартизованный по окну (якорь Behavior)
    beh_shared: np.ndarray        # (N, F) тот же профиль в train-системе координат (вход модели)
    evaluable: np.ndarray         # (N,) bool — хост активен в окне
    peer_out_counts: sparse.csr_matrix
    peer_out_bytes: sparse.csr_matrix
    peer_in_counts: sparse.csr_matrix
    peer_in_bytes: sparse.csr_matrix
    service: sparse.csr_matrix
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_flows: np.ndarray
    edge_bytes: np.ndarray
    edge_packets: np.ndarray

    @property
    def n_hosts(self) -> int:
        return int(self.behavior.shape[0])

    @property
    def contacts(self) -> sparse.csr_matrix:
        """Ненаправленные контакты с партнёрами: ровно матрица M из метрики."""
        return (self.peer_out_counts + self.peer_in_counts).tocsr()

    def pool(self) -> np.ndarray:
        """Индексы оцениваемых хостов: активные в окне (как eval_hosts эталона)."""
        return np.flatnonzero(self.evaluable)


def load_shard(path: Path, n_partners: int, n_services: int) -> WindowShard:
    """Прочитать один ``<окно>.npz``."""
    with np.load(path) as raw:
        shape_p = (int(raw["behavior"].shape[0]), n_partners)
        shape_s = (int(raw["behavior"].shape[0]), n_services)
        return WindowShard(
            name=Path(path).stem,
            behavior=raw["behavior"].astype(np.float32, copy=False),
            beh_shared=raw["beh_shared"].astype(np.float32, copy=False),
            evaluable=raw["evaluable"].astype(bool, copy=False),
            peer_out_counts=_csr(raw["peer_out_indptr"], raw["peer_out_indices"],
                                 raw["peer_out_counts"], shape_p),
            peer_out_bytes=_csr(raw["peer_out_indptr"], raw["peer_out_indices"],
                                raw["peer_out_bytes"], shape_p),
            peer_in_counts=_csr(raw["peer_in_indptr"], raw["peer_in_indices"],
                                raw["peer_in_counts"], shape_p),
            peer_in_bytes=_csr(raw["peer_in_indptr"], raw["peer_in_indices"],
                               raw["peer_in_bytes"], shape_p),
            service=_csr(raw["service_indptr"], raw["service_indices"],
                         raw["service_counts"], shape_s),
            edge_src=raw["edge_src"].astype(np.int64, copy=False),
            edge_dst=raw["edge_dst"].astype(np.int64, copy=False),
            edge_flows=raw["edge_flows"].astype(np.float32, copy=False),
            edge_bytes=raw["edge_bytes"].astype(np.float32, copy=False),
            edge_packets=raw["edge_packets"].astype(np.float32, copy=False),
        )


class ShardSet:
    """Набор срезов одного протокола: словари, когорта и ленивое чтение окон."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.vocab = json.loads((self.root / "vocab.json").read_text(encoding="utf-8"))
        self.partners: list[str] = list(self.vocab["partners"])
        self.services: list[str] = list(self.vocab["services"])
        self.windows: list[str] = list(self.manifest["windows"])
        self._cache: dict[str, WindowShard] = {}
        self._hosts: list[str] | None = None

    @property
    def hosts(self) -> list[str]:
        """Имена когорты в порядке строк всех срезов."""
        if self._hosts is None:
            path = self.root / "hosts.txt"
            if not path.exists():
                raise FileNotFoundError(f"{path}: нет списка хостов когорты")
            self._hosts = path.read_text(encoding="utf-8").splitlines()
        return self._hosts

    @property
    def n_hosts(self) -> int:
        return int(self.manifest["n_hosts"])

    @property
    def n_partners(self) -> int:
        return len(self.partners)

    @property
    def n_services(self) -> int:
        return len(self.services)

    def __contains__(self, name: str) -> bool:
        return name in self.windows

    def shard(self, name: str) -> WindowShard:
        if name not in self._cache:
            self._cache[name] = load_shard(self.root / f"{name}.npz", self.n_partners, self.n_services)
        return self._cache[name]

    def episode_days(self, days: list[str] | None = None) -> list[str]:
        """Дни, на которых учимся: по умолчанию все окна, кроме train и validation."""
        if days is not None:
            unknown = [d for d in days if d not in self.windows]
            if unknown:
                raise KeyError(f"нет срезов для окон {unknown}")
            return list(days)
        return [w for w in self.windows if w not in ("train", "validation")]

    def eval_hosts(self, name: str) -> list[str]:
        """Имена оцениваемых хостов окна в порядке когорты (для записи эмбеддингов)."""
        hosts = self.hosts
        return [hosts[i] for i in self.shard(name).pool()]


@dataclass
class WindowTensors:
    """Тензоры одного окна: то, что реально получает модель на каждом шаге."""

    beh_shared: torch.Tensor          # (N, F) float32
    anchor: torch.Tensor              # (N, F) float32, без градиента
    out_indices: torch.Tensor
    out_offsets: torch.Tensor
    out_weights: torch.Tensor
    in_indices: torch.Tensor
    in_offsets: torch.Tensor
    in_weights: torch.Tensor
    svc_indices: torch.Tensor
    svc_offsets: torch.Tensor
    svc_weights: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor
    pool: torch.Tensor                # (P,) индексы активных хостов


def bag_inputs(mat: sparse.csr_matrix, *, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Вход EmbeddingBag из CSR: плоские индексы, смещения и веса log1p."""
    mat = mat.tocsr()
    indices = torch.from_numpy(mat.indices.astype(np.int64, copy=True))
    offsets = torch.from_numpy(mat.indptr[:-1].astype(np.int64, copy=True))
    weights = torch.from_numpy(np.log1p(mat.data).astype(np.float32, copy=False) * np.float32(scale))
    return indices, offsets, weights


def window_tensors(shard: WindowShard, *, device: torch.device | str = "cpu") -> WindowTensors:
    """Собрать тензоры окна; рёбра дублируются в обе стороны (граф ненаправленный)."""
    n = shard.n_hosts
    src, dst = shard.edge_src, shard.edge_dst
    weights = np.log1p(shard.edge_flows.astype(np.float64))
    # симметричная нормировка по степеням, как в GCN: сообщение не растёт со степенью
    deg = np.bincount(np.concatenate([src, dst]), minlength=n).astype(np.float64)
    norm = 1.0 / np.sqrt(np.maximum(deg[src] * deg[dst], 1.0))
    edge_weight = (weights * norm).astype(np.float32)
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    both_w = np.concatenate([edge_weight, edge_weight])
    out_i, out_o, out_w = bag_inputs(shard.peer_out_counts)
    in_i, in_o, in_w = bag_inputs(shard.peer_in_counts)
    svc_i, svc_o, svc_w = bag_inputs(shard.service)
    return WindowTensors(
        beh_shared=torch.from_numpy(shard.beh_shared).to(device),
        anchor=torch.from_numpy(shard.behavior).to(device),
        out_indices=out_i.to(device), out_offsets=out_o.to(device), out_weights=out_w.to(device),
        in_indices=in_i.to(device), in_offsets=in_o.to(device), in_weights=in_w.to(device),
        svc_indices=svc_i.to(device), svc_offsets=svc_o.to(device), svc_weights=svc_w.to(device),
        edge_src=torch.from_numpy(both_src).to(device),
        edge_dst=torch.from_numpy(both_dst).to(device),
        edge_weight=torch.from_numpy(both_w).to(device),
        pool=torch.from_numpy(shard.pool().astype(np.int64)).to(device),
    )


def pool_matrices(shard: WindowShard, source: WindowShard) -> tuple[sparse.csr_matrix, ...]:
    """Матрицы метрики для окна: M (контакты pool × партнёры в source) и T (контакты дня).

    Строки — только оцениваемые хосты окна (``shard.pool()``); столбцы — только те
    партнёры, с которыми pool контактировал в source-окне. Именно это множество
    кандидатов строит ``evaluation.partners`` (universe = source-партнёры оцениваемых
    хостов). Третий элемент — позиции этих столбцов в полном словаре партнёров.
    """
    pool = shard.pool()
    m = source.contacts.tocsr()[pool].tocsc()
    keep = np.flatnonzero(np.diff(m.indptr) > 0).astype(np.int64)
    m = m[:, keep].tocsr().astype(np.float32)
    t = shard.contacts.tocsr()[pool].tocsc()[:, keep].tocsr().astype(np.float32)
    return m, t, keep


def host_columns(partners: list[str], hosts: list[str], keep: np.ndarray,
                 pool: np.ndarray) -> np.ndarray:
    """Позиция хоста среди кандидатов (столбцов keep) или -1, если он не партнёр.

    Нужна для того же запрета, что в метрике: сам хост не может быть своим партнёром.
    """
    index = {p: i for i, p in enumerate(partners)}
    full = np.array([index.get(hosts[i], -1) for i in pool], dtype=np.int64)
    if len(keep) == 0:
        return np.full(len(pool), -1, dtype=np.int64)
    pos = np.searchsorted(keep, np.maximum(full, 0))
    pos = np.minimum(pos, len(keep) - 1)
    valid = (full >= 0) & (keep[pos] == full)
    return np.where(valid, pos, -1).astype(np.int64)


def to_torch_sparse_t(mat: sparse.spmatrix, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Транспонированная матрица как torch sparse COO: M^T @ W^T даёт счётчики партнёров."""
    coo = mat.T.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()
