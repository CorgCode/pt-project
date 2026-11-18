"""Directed edge-aware message passing with a zero-initialized codebook residual."""
import torch
from torch import nn
from torch.nn import functional as F


class DirectedLayer(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.own = nn.Linear(width, width)
        self.messages = nn.ModuleList(nn.Linear(width + 7, width) for _ in range(2))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, edges, use_graph=True):
        value = self.own(h)
        if use_graph:
            for direction, transform in zip(('out', 'in'), self.messages):
                center = edges[direction + '_center']
                neighbor = edges[direction + '_neighbor']
                attr = edges[direction + '_attr']
                m = F.gelu(transform(torch.cat((h[neighbor], attr), dim=1)))
                aggregate = torch.zeros_like(h).index_add_(0, center, m)
                count = torch.bincount(center, minlength=len(h)).clamp_min(1).to(h.dtype)
                value = value + aggregate / count[:, None]
        return F.gelu(self.norm(value))


class CodebookDirectedGNN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.input = nn.Linear(11, width)
        self.layers = nn.ModuleList([DirectedLayer(width), DirectedLayer(width)])
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.hour = nn.Linear(width + 3, width)
        self.source_residual = nn.Linear(width, 896, bias=False)
        self.late_residual = nn.Linear(width, 896, bias=False)
        nn.init.zeros_(self.source_residual.weight)
        nn.init.zeros_(self.late_residual.weight)

    def graph(self, features, edges, *, n_output, use_graph=True):
        # Exact byte sums remain in the input artifact, never in model IDs.
        raw = features
        byte_total = raw[:, :, 0] + raw[:, :, 1]
        fraction = raw[:, :, 1] / byte_total.clamp_min(1.)
        active = (raw[:, :, 4] + raw[:, :, 5] > 0).to(raw.dtype)
        x = torch.cat((torch.log1p(raw) / 10., fraction[..., None],
                       active[..., None], (torch.log1p(byte_total) / 10.)[..., None]), dim=-1)
        chunks = []
        for t in range(6):
            h = F.gelu(self.input(x[:, t]))
            for layer in self.layers:
                h = layer(h, edges[t], use_graph=use_graph)
            chunks.append(h[:n_output])
        _, hidden = self.temporal(torch.stack(chunks, dim=1))
        summed = raw[:n_output, :, :2].sum(dim=1)
        total = summed.sum(dim=1, keepdim=True)
        direct = torch.cat((torch.log1p(summed) / 10., summed[:, 1:2] / total.clamp_min(1.)), dim=1)
        return F.gelu(self.hour(torch.cat((hidden[-1], direct), dim=1)))

    def source(self, base, graph):
        return base + self.source_residual(graph)

    def late(self, base, graph):
        return base + self.late_residual(graph)
