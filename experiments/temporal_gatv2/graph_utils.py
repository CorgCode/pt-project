"""Convert prepared graphs and construct a two-hop minibatch subgraph."""
import joblib
import numpy as np
import torch


def load_window(path):
    data = joblib.load(path)
    if data['node_features'].shape[1:] != (12, 8) or len(data['graphs']) != 12:
        raise ValueError('unexpected graph format')
    return data


def tensor_window(data):
    # Fixed, non-fitted rescaling; no future window contributes normalization statistics.
    x = torch.from_numpy(data['node_features'] / 10.)
    contacts = torch.from_numpy(data['contacts'])
    graphs = []
    for g in data['graphs']:
        item = {}
        for key, value in g.items():
            value = value.copy()
            if key.endswith('_attr'):
                value[:, :6] /= 10.
            item[key] = torch.from_numpy(value)
        graphs.append(item)
    return x, graphs, contacts


def subgraph(data, roots, *, hops=2, cap_per_direction=4):
    """Two-hop expansion across time; no labels enter graph selection.

    Uses the first `cap_per_direction` pre-ranked edges per query/slice to
    bound memory. Node totals/contact sketches are not sampled. Returned roots
    stay first and in the requested order. External nodes are not dropped.
    """
    roots = np.asarray(roots, dtype=np.int64)
    if len(np.unique(roots)) != len(roots) or (roots < 0).any() or (roots >= data['n_cohort']).any():
        raise ValueError('invalid cohort roots')
    n = len(data['nodes'])
    selected = np.zeros(n, dtype=bool); selected[roots] = True
    trimmed = []
    for graph in data['graphs']:
        part = {}
        for direction in ['out', 'in']:
            center = graph[f'{direction}_center']
            # Stored arrays are sorted by (part,center,count descending).
            if len(center):
                start = np.maximum.accumulate(np.where(np.r_[True, center[1:] != center[:-1]], np.arange(len(center)), 0))
                keep = np.arange(len(center)) - start < cap_per_direction
            else:
                keep = np.zeros(0, dtype=bool)
            for suffix in ['center', 'neighbor', 'attr']:
                part[f'{direction}_{suffix}'] = graph[f'{direction}_{suffix}'][keep]
        trimmed.append(part)
    for _ in range(hops):
        expanded = selected.copy()
        for graph in trimmed:
            for direction in ['out', 'in']:
                center, neighbor = graph[f'{direction}_center'], graph[f'{direction}_neighbor']
                expanded[neighbor[selected[center]]] = True
        selected = expanded
    other = np.flatnonzero(selected); other = other[~np.isin(other, roots)]
    positions = np.concatenate([roots, other])
    inverse = np.full(n, -1, dtype=np.int64); inverse[positions] = np.arange(len(positions))
    graphs = []
    for graph in trimmed:
        item = {}
        for direction in ['out', 'in']:
            center, neighbor = graph[f'{direction}_center'], graph[f'{direction}_neighbor']
            mask = selected[center] & selected[neighbor]
            item[f'{direction}_center'] = torch.from_numpy(inverse[center[mask]])
            item[f'{direction}_neighbor'] = torch.from_numpy(inverse[neighbor[mask]])
            attrs = graph[f'{direction}_attr'][mask].copy()
            attrs[:, :6] /= 10.
            item[f'{direction}_attr'] = torch.from_numpy(attrs)
        graphs.append(item)
    return (torch.from_numpy(data['node_features'][positions].copy() / 10.), graphs,
            torch.from_numpy(data['contacts'][positions].copy()), positions)
