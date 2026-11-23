"""Evaluate a predeclared weight grid on an allowed development reference.

This reports actual partner-ranking metrics only; it does not select a final
model, optimize parameters, read any specific data, or calculate total score.
"""
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from evaluation.partners import partner_ranking
from .model import source_vectors


def evaluate_grid(base, profiles, host_ids, partners, active_hosts, weights_grid, *, k=10):
    """Use exact project scoring incl. exclusions and midrank for each weight.

    Inputs and hyperparameter grid are supplied explicitly. Any dimensionality
    reduction of sparse profiles must be fitted before this function using only
    the allowed training prefix; evaluation occurs after that projection.
    Dense allocation happens only for active hosts, not the entire cohort.
    """
    host_ids = pd.Index(host_ids, dtype=str)
    active = pd.Index(active_hosts, dtype=str)
    if host_ids.has_duplicates or active.has_duplicates:
        raise ValueError('host identifiers must be unique')
    pos = host_ids.get_indexer(active)
    if (pos < 0).any() or len(active) <= k:
        raise ValueError('active hosts must belong to the cohort and exceed k')
    if base.ndim != 2 or base.shape[0] != len(host_ids) or len(profiles) != 3:
        raise ValueError('base and history profiles must match the cohort')
    if any(x.shape[0] != len(host_ids) for x in profiles):
        raise ValueError('history profiles must match the cohort')
    b = torch.as_tensor(np.asarray(base)[pos], dtype=torch.float64)
    p = [torch.as_tensor(x[pos].toarray() if sparse.issparse(x) else np.asarray(x)[pos],
                         dtype=torch.float64) for x in profiles]
    results = []
    for weights in weights_grid:
        z = source_vectors(b, p, weights).numpy()
        emb = pd.DataFrame(z, index=active)
        result = partner_ranking(emb, partners, k=k)
        results.append({'weights': [float(w) for w in weights],
                        'dimension': z.shape[1],
                        'new_partners_mrr': result['new']['embedding']['mrr'],
                        'new_partners_n_hosts': result['new']['embedding']['n_hosts'],
                        'new_partners_popularity_mrr': result['new']['popularity']['mrr']})
    return results
