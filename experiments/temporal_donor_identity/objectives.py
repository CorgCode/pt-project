"""Train-time surrogates and exact discrete donor voting (not model scores)."""
import torch
from torch.nn import functional as F


def hard_votes(vectors, source_counts, queries, active, k=10):
    """Actual cosine-weighted votes; query indices and rows share source order.

    source_counts is dense or sparse COO/CSR host×partner source02 counts.
    Self is excluded and all donors come from the active subset.
    """
    n = vectors.shape[0]
    if vectors.ndim != 2 or source_counts.ndim != 2 or source_counts.shape[0] != n:
        raise ValueError('incompatible vector and count shapes')
    if active.shape != (n,) or active.dtype != torch.bool:
        raise ValueError('active must be a boolean mask of hosts')
    if queries.ndim != 1 or queries.dtype != torch.long or not len(queries):
        raise ValueError('queries must be a non-empty vector of indices')
    if k < 1 or not bool(active[queries].all()) or int(active.sum()) <= k:
        raise ValueError('queries must be active and have at least k other active donors')
    z = F.normalize(vectors, dim=1)
    sim = z[queries] @ z.T
    sim = sim.masked_fill(~active.unsqueeze(0), -torch.inf)
    sim[torch.arange(len(queries), device=queries.device), queries] = -torch.inf
    values, indices = sim.topk(k, dim=1)
    weights = torch.zeros_like(sim).scatter(1, indices, values.clamp_min(0))
    if source_counts.layout == torch.strided:
        votes = weights @ source_counts
    else:
        votes = torch.sparse.mm(source_counts.to_sparse_coo().transpose(0, 1), weights.T).T
    return votes, indices, weights


def soft_votes(vectors, source_counts, queries, active, *, k=10, temperature=0.05):
    """Differentiable approximation; fractional donors make this non-official."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    # Reuse validation and compute the discrete votes for diagnostics separately.
    _, _, _ = hard_votes(vectors, source_counts, queries, active, k)
    z = F.normalize(vectors, dim=1)
    sim = z[queries] @ z.T
    allowed = active.unsqueeze(0).expand(len(queries), -1).clone()
    allowed[torch.arange(len(queries), device=queries.device), queries] = False
    masked = sim.masked_fill(~allowed, -1e6)
    with torch.no_grad():
        lo = masked.masked_fill(~allowed, torch.inf).min(dim=1, keepdim=True).values - 2
        hi = masked.max(dim=1, keepdim=True).values + 2
        for _ in range(48):
            mid = (lo + hi) / 2
            mass = (torch.sigmoid((masked - mid) / temperature) * allowed).sum(1, keepdim=True)
            lo = torch.where(mass > k, mid, lo)
            hi = torch.where(mass > k, hi, mid)
        threshold = (lo + hi) / 2
        p = torch.sigmoid((masked - threshold) / temperature) * allowed
        derivative = p * (1 - p)
    implicit = (sim * derivative).sum(1, keepdim=True) / derivative.sum(1, keepdim=True).clamp_min(1e-12)
    threshold = threshold + implicit - implicit.detach()
    weights = torch.sigmoid((masked - threshold) / temperature) * allowed * sim.clamp_min(0)
    if source_counts.layout == torch.strided:
        return weights @ source_counts
    return torch.sparse.mm(source_counts.to_sparse_coo().transpose(0, 1), weights.T).T


def first_positive_rank_loss(votes, valid, positive, temperature=0.1):
    """Smooth first-ANY rank; exact midrank metrics must be computed separately."""
    if temperature <= 0 or votes.shape != valid.shape or valid.shape != positive.shape:
        raise ValueError('bad shapes or temperature')
    if valid.dtype != torch.bool or positive.dtype != torch.bool:
        raise ValueError('masks must be boolean')
    if (positive & ~valid).any() or not positive.any(dim=1).all():
        raise ValueError('every row must contain at least one valid positive')
    # Smooth minimum of positive ranks; duplicates/multiple positives are not
    # treated as a categorical distribution of alternative outcomes.
    # Only the highest-scoring positive can have the best rank. This avoids
    # allocating a quadratic partners×partners matrix for the full universe.
    best = votes.masked_fill(~positive, -torch.inf).max(dim=1).values
    ranks = 0.5 + (torch.sigmoid((votes - best[:, None]) / temperature) * valid).sum(dim=1)
    return ranks.clamp_min(1).log().mean()


def preserve_neighbors_loss(base, combined, queries, active, temperature=0.05):
    """KL from original cosine-neighbor distribution to the new one."""
    if temperature <= 0 or base.shape[0] != combined.shape[0]:
        raise ValueError('invalid temperature or mismatched rows')
    if active.dtype != torch.bool or active.shape != (len(base),):
        raise ValueError('invalid active mask')
    if not bool(active[queries].all()) or int(active.sum()) < 2:
        raise ValueError('each query needs another active host')
    mask = active.unsqueeze(0).expand(len(queries), -1).clone()
    mask[torch.arange(len(queries), device=queries.device), queries] = False
    old = F.normalize(base.detach(), dim=1)
    new = F.normalize(combined, dim=1)
    old_logits = (old[queries] @ old.T / temperature).masked_fill(~mask, -1e9)
    new_logits = (new[queries] @ new.T / temperature).masked_fill(~mask, -1e9)
    return F.kl_div(F.log_softmax(new_logits, dim=1), F.softmax(old_logits, dim=1), reduction='batchmean')


def episodic_identity_loss(source, late, calibration, evaluation, *, temperature=0.05,
                           regularization=0.0, alignment_gradient='stop', baseline=None,
                           residual_weight=0.0):
    """Source→late cross entropy after calibration-only centered Procrustes.

    `stop` fixes the rotation for backprop; `svd` differentiates through it,
    optionally regularized. Official evaluation uses neither regularization
    nor this training loss. Repeated/zero singular values may make svd gradients
    unstable, especially when calibration size is less than the dimension.
    """
    if source.ndim != 2 or source.shape != late.shape:
        raise ValueError('source and late shapes must match')
    if not torch.isfinite(source).all() or not torch.isfinite(late).all():
        raise ValueError('non-finite vectors')
    if temperature <= 0 or regularization < 0 or residual_weight < 0:
        raise ValueError('invalid loss hyperparameters')
    if alignment_gradient not in ('stop', 'svd'):
        raise ValueError('alignment_gradient must be stop or svd')
    for indices in (calibration, evaluation):
        if indices.ndim != 1 or indices.dtype != torch.long or len(indices) < 2:
            raise ValueError('splits must be long vectors of at least two hosts')
        if torch.any(indices < 0) or torch.any(indices >= len(source)) or len(torch.unique(indices)) != len(indices):
            raise ValueError('indices out of range or duplicated')
    if torch.isin(calibration, evaluation).any():
        raise ValueError('calibration and evaluation must be disjoint')
    src = source.detach()
    ma, mb = src[calibration].mean(0), late[calibration].mean(0)
    cross = (late[calibration] - mb).T @ (src[calibration] - ma)
    if regularization:
        cross = cross + regularization * torch.eye(cross.shape[0], dtype=cross.dtype, device=cross.device)
    if alignment_gradient == 'stop':
        with torch.no_grad():
            u, _, vh = torch.linalg.svd(cross.detach(), full_matrices=False)
            rotation = u @ vh
    else:
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        rotation = u @ vh
    a = F.normalize(src[evaluation] - ma, dim=1)
    b = F.normalize((late[evaluation] - mb) @ rotation, dim=1)
    logits = a @ b.T / temperature
    loss = F.cross_entropy(logits, torch.arange(len(evaluation), device=logits.device))
    if residual_weight:
        if baseline is None or baseline.shape != late.shape:
            raise ValueError('baseline required for residual regularization')
        loss = loss + residual_weight * (late[evaluation] - baseline.detach()[evaluation]).square().sum(1).mean()
    return loss, logits
