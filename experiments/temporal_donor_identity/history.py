"""Construct causal historical contact profiles from precomputed count matrices."""
from dataclasses import dataclass
import math
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Observation:
    day: int
    hour: float
    end: float
    counts: sparse.spmatrix


def contact_profiles(observations, *, cutoff, target_hour, half_life_days=3., period_hours=24., vocabulary_idf=None):
    """Return day-presence, recency and hour-matched CSR profiles.

    All matrices must share an externally fitted host/partner vocabulary. The
    source02 window may be included if its end is before the training cutoff.
    `end` and `cutoff` use the same epoch/units. target_hour is known schedule,
    not target-window flows. For strict folds build matrices and IDF from that
    fold's prefix only. Count absence in a missing window is not observed absence.
    """
    if not observations or not math.isfinite(cutoff) or not 0 <= target_hour < period_hours:
        raise ValueError('invalid history, cutoff or target hour')
    if half_life_days <= 0 or period_hours <= 0:
        raise ValueError('half life and period must be positive')
    shape = observations[0].counts.shape
    if len(shape) != 2 or not all(shape):
        raise ValueError('history matrices must be nonempty')
    by_day = {}
    recent = sparse.csr_matrix(shape, dtype=np.float64)
    periodic = sparse.csr_matrix(shape, dtype=np.float64)
    latest = max(o.day for o in observations)
    for o in observations:
        if o.end > cutoff:
            raise ValueError('future observation after cutoff')
        if not 0 <= o.hour < period_hours or o.counts.shape != shape:
            raise ValueError('invalid hour or mismatched matrix shape')
        x = o.counts.tocsr().astype(np.float64)
        if not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError('counts must be finite and nonnegative')
        x.eliminate_zeros()
        present = x.copy(); present.data[:] = 1.
        by_day[o.day] = by_day.get(o.day, sparse.csr_matrix(shape)) + present
        recent += present * (2 ** (-(latest - o.day) / half_life_days))
        distance = abs(o.hour - target_hour)
        distance = min(distance, period_hours - distance)
        periodic += present * math.exp(-0.5 * (distance / 2.) ** 2)
    days = sparse.csr_matrix(shape, dtype=np.float64)
    for x in by_day.values():
        x.data[:] = 1.
        days += x
    idf = np.ones(shape[1]) if vocabulary_idf is None else np.asarray(vocabulary_idf, dtype=float)
    if idf.shape != (shape[1],) or not np.isfinite(idf).all() or (idf < 0).any():
        raise ValueError('invalid prefix-fitted IDF')
    return tuple(normalize(x.multiply(idf).tocsr(), norm='l2') for x in (days, recent, periodic))
