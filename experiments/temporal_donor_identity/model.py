"""Frozen codebook with a zero-initialized direct late-vector correction."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    base_dim: int = 896
    context_dim: int = 64
    hidden_dim: int = 128
    correction: str = 'linear'
    identity_bias: bool = True

    def __post_init__(self):
        if min(self.base_dim, self.context_dim, self.hidden_dim) <= 0:
            raise ValueError('dimensions must be positive')
        if self.correction not in ('linear', 'mlp'):
            raise ValueError('correction must be linear or mlp')


def _shape(name, value, rows, width):
    if value.ndim != 2 or value.shape != (rows, width):
        raise ValueError(f'{name}: expected {(rows, width)}, got {tuple(value.shape)}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name}: non-finite input')


class TemporalDonorIdentity(nn.Module):
    """Residual adapter on observed traffic; no query-ID or candidate lookup.

    For extended source coordinates provide a separately trained instance with
    base_dim equal to the extended dimension and an appropriate late baseline.
    """
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        if config.correction == 'linear':
            self.identity = nn.Linear(config.base_dim + config.context_dim, config.base_dim, bias=config.identity_bias)
            last = self.identity
        else:
            self.identity = nn.Sequential(
                nn.Linear(config.base_dim + config.context_dim, config.hidden_dim),
                nn.GELU(), nn.Linear(config.hidden_dim, config.base_dim, bias=config.identity_bias),
            )
            last = self.identity[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def late_vectors(self, late_base, late_context, *, eta=1.0):
        if late_base.ndim != 2 or not len(late_base):
            raise ValueError('late_base must be a nonempty matrix')
        _shape('late_base', late_base, len(late_base), self.config.base_dim)
        _shape('late_context', late_context, len(late_base), self.config.context_dim)
        if not math.isfinite(eta) or eta < 0:
            raise ValueError('eta must be finite and nonnegative')
        baseline = late_base.detach()
        correction = self.identity(torch.cat((baseline, late_context), dim=1))
        # No normalization: exact numerical baseline matters to centered Procrustes.
        return baseline + eta * correction


def source_vectors(base, profiles, weights):
    """Concatenate L2-normalized history blocks with sqrt weights.

    weights=[base,days,recent,periodic] must sum to one. With baseline
    weights [1,0,0,0] return base unchanged, preserving dimensions and values.
    Zero rows of a profile remain zero; the final vector is not renormalized,
    because the evaluator normalizes it and zero blocks must be handled fairly.
    """
    if base.ndim != 2 or not len(base) or len(profiles) != 3 or len(weights) != 4:
        raise ValueError('expected base, three history profiles and four weights')
    if not torch.isfinite(base).all():
        raise ValueError('base contains non-finite values')
    weights = tuple(float(w) for w in weights)
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-8):
        raise ValueError('weights must be finite, nonnegative and sum to one')
    if weights == (1., 0., 0., 0.):
        return base.detach().clone()
    parts = [math.sqrt(weights[0]) * F.normalize(base.detach(), dim=1)]
    for i, (x, weight) in enumerate(zip(profiles, weights[1:])):
        if x.ndim != 2 or x.shape[0] != len(base) or x.shape[1] == 0 or not torch.isfinite(x).all():
            raise ValueError(f'invalid history profile {i}')
        parts.append(math.sqrt(weight) * F.normalize(x.detach(), dim=1))
    return torch.cat(parts, dim=1)
