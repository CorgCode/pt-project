"""Training surrogates: exact hard votes forward, approximate gradients."""
import numpy as np
import torch
from torch.nn import functional as F


def episodic_identity(source, late, *, temperature=.1):
    """Stop-gradient Procrustes followed by contrastive evaluation on held-out batch hosts."""
    n = len(source)
    if n < 8:
        raise ValueError('at least 8 matched hosts are required')
    perm = torch.randperm(n, device=source.device)
    cal, ev = perm[:n // 2], perm[n // 2:]
    mean_a, mean_b = source[cal].mean(0), late[cal].mean(0)
    with torch.no_grad():
        cross = (late[cal] - mean_b).T @ (source[cal] - mean_a)
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        rotation = u @ vh
    a = F.normalize(source[ev] - mean_a, dim=1)
    b = F.normalize((late[ev] - mean_b) @ rotation, dim=1)
    labels = torch.arange(len(ev), device=source.device)
    logits = a @ b.T / temperature
    loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    # Direct contrast stabilizes directions not determined by a small calibration batch.
    direct = source @ late.T / temperature
    all_labels = torch.arange(n, device=source.device)
    return loss + .25 * .5 * (F.cross_entropy(direct, all_labels) + F.cross_entropy(direct.T, all_labels))


def straight_through_votes(z, query, active, source_transpose, *, k=10, temperature=.03):
    """Forward is actual cosine/top-k/count vote; sigmoid surrogate is biased."""
    if len(active) <= k or temperature <= 0:
        raise ValueError('insufficient active donors or invalid temperature')
    z = F.normalize(z, dim=1)
    sim = z[query] @ z[active].T
    allowed = query[:, None] != active[None, :]
    masked = sim.masked_fill(~allowed, -1e6)
    top = masked.topk(k, dim=1).indices
    hard = torch.zeros_like(sim).scatter(1, top, 1.)
    with torch.no_grad():
        lo = masked.masked_fill(~allowed, torch.inf).min(1, keepdim=True).values - 1
        hi = masked.max(1, keepdim=True).values + 1
        for _ in range(24):
            mid = (lo + hi) / 2
            mass = (torch.sigmoid((masked-mid)/temperature) * allowed).sum(1, keepdim=True)
            lo, hi = torch.where(mass > k, mid, lo), torch.where(mass > k, hi, mid)
        threshold = (lo + hi) / 2
        p = torch.sigmoid((masked-threshold)/temperature) * allowed
        derivative = p * (1-p)
    implicit = (sim * derivative).sum(1, keepdim=True) / derivative.sum(1, keepdim=True).clamp_min(1e-12)
    threshold = threshold + implicit - implicit.detach()
    soft = torch.sigmoid((masked-threshold)/temperature) * allowed
    gate = hard + soft - soft.detach()
    weights = gate * sim.clamp_min(0)
    return torch.sparse.mm(source_transpose, weights.T).T, top


def vote_rank_loss(votes, valid, positive, *, temperature=.05):
    if not positive.any(1).all() or (positive & ~valid).any():
        raise ValueError('all queries need a valid positive')
    # Scaling prevents large raw counts from saturating all pair comparisons.
    scale = votes.detach().amax(1, keepdim=True).clamp_min(1.)
    v = votes / scale
    best = v.masked_fill(~positive, -torch.inf).max(1).values
    rank = .5 + (torch.sigmoid((v-best[:, None])/temperature) * valid).sum(1)
    return rank.clamp_min(1).log().mean()


def relation_loss(z, node_features, contacts):
    """Preserve a mix of byte/behavior and observed-peer cosine relations."""
    behavior = F.normalize(node_features.reshape(len(z), -1), dim=1, eps=1e-8)
    peers = F.normalize(contacts, dim=1, eps=1e-8)
    target = .5 * (behavior @ behavior.T) + .5 * (peers @ peers.T)
    return F.mse_loss(z @ z.T, target.detach())
