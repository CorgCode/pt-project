import os
"""Быстрый адаптер готовой v5: ограниченный gate и обученная на train проекция."""
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn

SIZES = (57, 128, 128, 384, 128)
OFFSETS = np.cumsum((0,) + SIZES)


def reliability(x):
    features = [x[:, :57]]
    for a, b in zip(OFFSETS[:-1], OFFSETS[1:]):
        square = x[:, a:b] ** 2
        energy = square.sum(1)
        participation = energy ** 2 / np.maximum((square ** 2).sum(1) * (b-a), 1e-12)
        features.extend([np.log1p(energy)[:, None], participation[:, None]])
    return np.hstack(features)


class Gate(nn.Module):
    def __init__(self, inputs=67):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Linear(32, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return 1 + 0.5 * torch.tanh(self.net(x))


def apply_gate(x, gate, scaler, alpha):
    if alpha == 0:
        return x.copy()
    gate.eval()
    with torch.no_grad():
        feature = np.clip(scaler.transform(reliability(x)), -8, 8).astype(np.float32)
        weights = gate(torch.from_numpy(feature)).numpy()
    out = x.copy()
    for k, (a, b) in enumerate(zip(OFFSETS[:-1], OFFSETS[1:])):
        out[:, a:b] *= (1 + alpha * (weights[:, k] - 1))[:, None]
    return out


def transform(x, folder, selected):
    folder = Path(folder)
    recipe = selected['recipe']
    if recipe['alpha'] == 0:
        out = x.copy()
    else:
        scaler = joblib.load(folder/'gate_scaler.joblib')
        gate = Gate()
        gate.load_state_dict(torch.load(folder/'gate.pt',map_location='cpu',weights_only=True))
        out = apply_gate(x, gate, scaler, recipe['alpha'])
    if recipe['dim']:
        projection = joblib.load(folder/f"projection_alpha{recipe['alpha']:g}.joblib")
        out = projection.transform(out)[:, :recipe['dim']]
    return out
