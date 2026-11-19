"""Обучение MD-TGR на последовательности дней.

Один эпизод — «признаки train → цели дня d». Признаки (поведение, контакты, сервисы,
рёбра) всегда берутся из train-окна: на scoreboard именно этот вектор лежит в
``embeddings.csv``, и метрики Behavior / Future links / Shared partners / Pairs
считаются по нему против эталона позднего дня. Цель и матрица старых контактов M —
из дня d и train соответственно (так устроена метрика).

Иначе (признаки дня d) модель училась бы читать новые связи прямо из графа дня,
которого на настоящем тесте нет, и качество не переносилось бы.

Дни 14–16 запрещены: это финальная оценка, и в обучение они попасть не должны
ни по имени, ни по времени.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from methods.mdtgr.dataset import (ShardSet, WindowTensors, host_columns, pool_matrices,
                                   to_torch_sparse_t, window_tensors)
from methods.mdtgr.evaluate import new_partner_metrics
from methods.mdtgr.losses import (link_loss, partner_listwise_loss, partner_scores, rows_to_mask,
                                  sample_negatives)
from methods.mdtgr.model import MDTGR

# день 14 12:30 — начало официальных test-окон; всё, что позже, в обучение не идёт
OFFICIAL_TEST_START = 1168200.0


@dataclass
class TrainConfig:
    """Настройки обучения; всё, что влияет на число, лежит здесь и пишется в чекпойнт."""

    episodes: list[str] = field(default_factory=list)
    dev: str | None = None
    epochs: int = 40
    anchors: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    tau: float = 1.0
    link_weight: float = 0.05
    link_pairs: int = 4096
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    random_weights: bool = False
    eval_every: int = 5
    seed: int = 0
    dim: int = 96
    code_dim: int = 64
    bag_dim: int = 64
    service_dim: int = 32
    hidden: int = 192
    dropout: float = 0.1
    log: bool = True


@dataclass
class Episode:
    """Всё, что нужно шагу обучения по одному дню."""

    day: str
    tensors: WindowTensors
    pool: np.ndarray
    pool_tensor: torch.Tensor
    m: object
    t: object
    t_new: object
    m_t: torch.Tensor
    host_col: np.ndarray
    link_pos: torch.Tensor


def _cohort_to_pool(n_hosts: int, pool: np.ndarray) -> np.ndarray:
    mapping = np.full(n_hosts, -1, dtype=np.int64)
    mapping[pool] = np.arange(len(pool), dtype=np.int64)
    return mapping


def _link_positives(shard, source, pool: np.ndarray, n_hosts: int,
                    limit: int) -> torch.Tensor:
    """Рёбра дня внутри пула, которых не было в source-окне (новые связи)."""
    mapping = _cohort_to_pool(n_hosts, pool)
    src, dst = shard.edge_src, shard.edge_dst
    ok = (mapping[src] >= 0) & (mapping[dst] >= 0)
    a, b = mapping[src[ok]], mapping[dst[ok]]
    if len(a) == 0:
        return torch.zeros((0, 2), dtype=torch.long)

    old_lo = np.minimum(source.edge_src, source.edge_dst).astype(np.int64)
    old_hi = np.maximum(source.edge_src, source.edge_dst).astype(np.int64)
    old_keys = np.unique(old_lo * n_hosts + old_hi)
    lo, hi = np.minimum(src[ok], dst[ok]), np.maximum(src[ok], dst[ok])
    keys = lo.astype(np.int64) * n_hosts + hi.astype(np.int64)
    fresh = ~np.isin(keys, old_keys)
    a, b = a[fresh], b[fresh]
    if len(a) == 0:
        return torch.zeros((0, 2), dtype=torch.long)
    take = np.random.default_rng(0).choice(len(a), size=min(limit, len(a)), replace=False)
    return torch.from_numpy(np.stack([a[take], b[take]], axis=1).astype(np.int64))


def build_episode(shards: ShardSet, day: str, source_name: str = "train", *,
                  features: str = "train", link_pairs: int = 4096) -> Episode:
    """Подготовить эпизод: срез дня-цели, матрицы метрики и положительные рёбра.

    ``features`` — окно, из которого берутся входные признаки модели. По умолчанию
    это ``train``: ровно он же попадает в отправку (``embeddings.csv``), поэтому
    обучение и оценка смотрят на один и тот же вход.
    """
    shard = shards.shard(day)
    source = shards.shard(source_name)
    pool = shard.pool()
    m, t, keep = pool_matrices(shard, source)
    t_new = (t - t.multiply(_binary(m))).tocsr()
    t_new.eliminate_zeros()
    host_col = host_columns(shards.partners, shards.hosts, keep, pool)
    return Episode(day=day, tensors=window_tensors(shards.shard(features)), pool=pool,
                   pool_tensor=torch.from_numpy(pool.astype(np.int64)),
                   m=m, t=t, t_new=t_new, m_t=to_torch_sparse_t(m), host_col=host_col,
                   link_pos=_link_positives(shard, source, pool, shards.n_hosts, link_pairs))


def _binary(mat):
    out = mat.copy()
    out.data = np.ones_like(out.data)
    return out


def _mask_pair(m, t_new, anchors: torch.Tensor, n_cols: int, host_col: np.ndarray,
               device) -> tuple[torch.Tensor, torch.Tensor]:
    """Маски цели и запрещённых кандидатов для батча якорей."""
    rows = anchors.detach().cpu().numpy()
    targets = rows_to_mask(t_new, rows, n_cols, device=device)
    blocked = rows_to_mask(m, rows, n_cols, device=device)
    cols = host_col[rows]
    keep = cols >= 0
    if keep.any():
        idx = torch.arange(len(rows), device=device)[torch.from_numpy(keep)]
        blocked[idx, torch.from_numpy(cols[keep]).to(device)] = True
    return targets, blocked


def _sample_weights(cfg: TrainConfig, generator: torch.Generator) -> tuple[float, float, float]:
    """Веса блоков: фиксированные или случайные (чтобы каждый блок был полезен сам по себе)."""
    if not cfg.random_weights:
        return cfg.weights
    low, high = math.log(0.5), math.log(2.0)
    return tuple(float(math.exp(low + (high - low) * torch.rand(1, generator=generator).item()))
                 for _ in range(3))


def _dev_score(model: MDTGR, episode: Episode, weights) -> dict:
    model.eval()
    with torch.no_grad():
        window, code = model.encode(episode.tensors)
        z = model.compose(episode.tensors.anchor, code, window, weights)
        z = F.normalize(z, dim=1)[episode.pool_tensor].numpy()
    return new_partner_metrics(z, episode.m, episode.t, episode.host_col)


def train(shards: ShardSet, cfg: TrainConfig, out_dir: Path, *,
          allow_official_test: bool = False) -> dict:
    """Обучить модель; вернуть историю и метрики dev."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = shards.manifest.get("window_times", {})
    for day in cfg.episodes:
        start = times.get(day, [0.0, 0.0])[0]
        if start >= OFFICIAL_TEST_START and not allow_official_test:
            raise ValueError(f"окно {day} начинается в {start:.0f} — это официальный test, "
                             "обучаться на нём нельзя")
    if cfg.dev and cfg.dev in cfg.episodes:
        raise ValueError(f"dev-день {cfg.dev} не может быть эпизодом обучения")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    shard = shards.shard("train")
    model = MDTGR(shards.n_hosts, shards.n_partners, shards.n_services,
                  int(shard.behavior.shape[1]), dim=cfg.dim, code_dim=cfg.code_dim,
                  bag_dim=cfg.bag_dim, service_dim=cfg.service_dim, hidden=cfg.hidden,
                  dropout=cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    episodes = [build_episode(shards, d, link_pairs=cfg.link_pairs) for d in cfg.episodes]
    dev = build_episode(shards, cfg.dev, link_pairs=0) if cfg.dev else None
    history: list[dict] = []
    best = {"mrr": -1.0, "epoch": -1}
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(episodes), generator=generator).tolist()
        totals = {"loss": 0.0, "partner": 0.0, "link": 0.0, "steps": 0}
        for j in order:
            ep = episodes[j]
            weights = _sample_weights(cfg, generator)
            window, code = model.encode(ep.tensors)
            z_all = F.normalize(model.compose(ep.tensors.anchor, code, window, weights), dim=1)
            z = z_all[ep.pool_tensor]
            n_pool = z.shape[0]
            n_anchor = min(cfg.anchors, n_pool)
            anchors = torch.randperm(n_pool, generator=generator)[:n_anchor]
            scores = partner_scores(z, anchors, ep.m_t, k=10)
            targets, blocked = _mask_pair(ep.m, ep.t_new, anchors, scores.shape[1],
                                          ep.host_col, z.device)
            loss_p = partner_listwise_loss(scores, targets, blocked, tau=cfg.tau)
            loss = loss_p
            loss_l = torch.zeros(())
            if cfg.link_weight > 0 and len(ep.link_pos) > 0:
                take = min(cfg.link_pairs, len(ep.link_pos))
                pos = ep.link_pos[torch.randint(0, len(ep.link_pos), (take,), generator=generator)]
                neg = sample_negatives(n_pool, take, generator, z.device)
                loss_l = link_loss(z, pos, neg)
                loss = loss + cfg.link_weight * loss_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals["loss"] += float(loss.detach())
            totals["partner"] += float(loss_p.detach())
            totals["link"] += float(loss_l.detach())
            totals["steps"] += 1

        record = {"epoch": epoch, "seconds": round(time.time() - started, 1),
                  **{k: (v / max(totals["steps"], 1)) for k, v in totals.items() if k != "steps"}}
        if dev is not None and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            record["dev"] = _dev_score(model, dev, cfg.weights)
            if record["dev"]["mrr"] is not None and record["dev"]["mrr"] > best["mrr"]:
                best = {"mrr": record["dev"]["mrr"], "epoch": epoch}
                torch.save({"model": model.state(), "train": asdict(cfg), "dev": record["dev"]},
                           out_dir / "best.pt")
        history.append(record)
        if cfg.log:
            dev_text = ""
            if "dev" in record:
                dev_text = f" dev_mrr={record['dev']['mrr']:.4f} n={record['dev']['n_hosts']}"
            print(f"эпоха {epoch:3d}  loss={record['loss']:.4f}  partner={record['partner']:.4f}"
                  f"  link={record['link']:.4f}{dev_text}", flush=True)

    torch.save({"model": model.state(), "train": asdict(cfg), "dev": best}, out_dir / "last.pt")
    if not (out_dir / "best.pt").exists():
        torch.save({"model": model.state(), "train": asdict(cfg), "dev": None}, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return {"history": history, "best": best}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Обучение MD-TGR")
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", required=True, help="дни обучения через запятую")
    ap.add_argument("--dev", help="день для контроля (не входит в обучение)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--link-weight", type=float, default=0.05)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--code-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-official-test", action="store_true",
                    help="разрешить окна official test (дни 14–16) — только для отладки")
    ap.add_argument("--config", type=Path, help="JSON с настройками (перекрывает флаги)")
    args = ap.parse_args(argv)

    cfg = TrainConfig(episodes=[d.strip() for d in args.episodes.split(",") if d.strip()],
                      dev=args.dev, epochs=args.epochs, anchors=args.anchors, lr=args.lr,
                      tau=args.tau, link_weight=args.link_weight, random_weights=args.random_weights,
                      eval_every=args.eval_every, dim=args.dim, code_dim=args.code_dim, seed=args.seed)
    if args.config:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        cfg = TrainConfig(**{**asdict(cfg), **raw})
    shards = ShardSet(args.shards)
    result = train(shards, cfg, args.out, allow_official_test=args.allow_official_test)
    print(json.dumps({"best": result["best"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
