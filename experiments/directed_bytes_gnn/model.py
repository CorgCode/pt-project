"""Train an inductive directed GNN on windows through day 12 only."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from .data import features


def tensor_graph(matrix):
    m = matrix.tocoo()
    return torch.sparse_coo_tensor(np.vstack([m.row, m.col]), m.data.astype(np.float32),
                                   m.shape, check_invariants=True).coalesce()


def load_observation(root, name, *, include_bytes=True):
    p = Path(root) / name
    totals = np.load(p / 'totals.npy')
    counts, byte_graph = sparse.load_npz(p / 'counts.npz'), sparse.load_npz(p / 'bytes.npz')
    x, graphs, active = features(totals, counts, byte_graph, include_bytes=include_bytes)
    return {'x': x, 'graphs': [tensor_graph(g) for g in graphs], 'active': active}


class Layer(nn.Module):
    def __init__(self, inputs, outputs, use_graph=True):
        super().__init__()
        self.self_map = nn.Linear(inputs, outputs)
        self.neighbor_maps = nn.ModuleList(nn.Linear(inputs, outputs, bias=False) for _ in range(4))
        self.gates = nn.Parameter(torch.full((4,), -1.0))
        self.norm = nn.LayerNorm(outputs)
        self.use_graph = use_graph

    def forward(self, x, graphs):
        out = self.self_map(x)
        if self.use_graph:
            for gate, fn, graph in zip(self.gates.sigmoid(), self.neighbor_maps, graphs):
                out = out + gate * fn(torch.sparse.mm(graph, x))
        return F.gelu(self.norm(out))


class DirectedByteGNN(nn.Module):
    def __init__(self, inputs=8, hidden=64, dim=64, *, use_graph=True):
        super().__init__()
        self.first = Layer(inputs, hidden, use_graph)
        self.second = Layer(hidden, dim, use_graph)
        self.skip = nn.Linear(inputs, dim, bias=False)
        self.forecast = nn.Sequential(nn.Linear(dim + 1, hidden), nn.GELU(), nn.Linear(hidden, inputs))

    def forward(self, x, graphs):
        h = self.first(x, graphs)
        h = self.second(h, graphs) + .2 * self.skip(x)
        return F.normalize(h, dim=1)


def scale(x, mean, std):
    return torch.from_numpy(np.clip((x - mean) / std, -8, 8).astype(np.float32))


def train(observations, *, mode='gnn', steps=80, seed=17):
    if mode not in ('gnn', 'no_graph', 'no_bytes'):
        raise ValueError('unsupported ablation')
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Train-only standardization; the day13 development window is never used here.
    fit = np.concatenate([v['x'][v['active'] >= 300] for v in observations.values()])
    mean, std = fit.mean(0), np.maximum(fit.std(0), .05)
    xs = {name: scale(v['x'], mean, std) for name, v in observations.items()}
    network = DirectedByteGNN(use_graph=mode != 'no_graph')
    optimizer = torch.optim.AdamW(network.parameters(), lr=8e-4, weight_decay=1e-4)
    # day02↔late adds a long temporal identity gap. Targets for the explicit
    # one/two-day prediction never extend beyond day12.
    pairs = [('day02', 'day10'), ('day02', 'day11'), ('day02', 'day12'),
             ('day10', 'day11'), ('day10', 'day12'), ('day11', 'day12')]
    targets = {('day10', 'day11'): 1., ('day10', 'day12'): 2., ('day11', 'day12'): 1.}
    valid = {p: np.flatnonzero((observations[p[0]]['active'] >= 300) &
                               (observations[p[1]]['active'] >= 300)) for p in pairs}
    if min(map(len, valid.values())) < 2:
        raise ValueError('not enough recurring training hosts')
    history = []
    for step in range(1, steps + 1):
        a, b = pairs[(step - 1) % len(pairs)]
        idx = torch.from_numpy(rng.choice(valid[a, b], min(384, len(valid[a, b])), replace=False))
        za = network(xs[a], observations[a]['graphs'])[idx]
        zb = network(xs[b], observations[b]['graphs'])[idx]
        logits = za @ zb.T / .1
        labels = torch.arange(len(idx))
        contrast = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        forecast = torch.tensor(0.)
        if (a, b) in targets:
            horizon = torch.full((len(idx), 1), targets[a, b] / 2)
            forecast = F.smooth_l1_loss(network.forecast(torch.cat([za, horizon], dim=1)), xs[b][idx])
        loss = contrast + .3 * forecast
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5); optimizer.step()
        history.append({'step': step, 'from': a, 'to': b, 'contrast': float(contrast.detach()),
                        'forecast': float(forecast.detach())})
        if step == 1 or step % 20 == 0 or step == steps:
            print(mode, 'step', step, 'pair', a, b, 'contrast', round(contrast.item(), 4),
                  'forecast', round(forecast.item(), 4), flush=True)
    return {'mode': mode, 'steps': steps, 'seed': seed, 'weights': network.state_dict(),
            'mean': mean, 'std': std, 'history': history,
            'train_windows': list(observations), 'forecast_pairs': [list(p) for p in targets]}


def infer(state, observation):
    network = DirectedByteGNN(use_graph=state['mode'] != 'no_graph')
    network.load_state_dict(state['weights']); network.eval()
    with torch.no_grad():
        return network(scale(observation['x'], state['mean'], state['std']), observation['graphs']).numpy()


def save_state(state, path):
    # Separate simple metadata from tensors; PyTorch loading is restricted to weights_only.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'mode': state['mode'], 'steps': state['steps'], 'seed': state['seed'],
                'weights': state['weights'], 'mean': torch.from_numpy(state['mean']),
                'std': torch.from_numpy(state['std'])}, path)
    path.with_suffix('.json').write_text(json.dumps({k: v for k, v in state.items() if k not in ('weights', 'mean', 'std')}, indent=2))


def load_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    state['mean'] = state['mean'].numpy(); state['std'] = state['std'].numpy()
    return state
