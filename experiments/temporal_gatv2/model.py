"""Directed GATv2 and masked hierarchical time encoder, independent of host ID.

GATv2 applies the attention vector *after* the nonlinear combination of query,
neighbor and edge features. See https://arxiv.org/abs/2105.14491. This
edge-aware directed version is an adaptation, not a TGAT implementation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def segment_softmax(logits, group, n_groups):
    """Stable softmax over edges for each (query node, attention head)."""
    if logits.ndim != 2 or group.ndim != 1 or len(logits) != len(group):
        raise ValueError('invalid segment shapes')
    if len(group) == 0:
        return logits
    index = group[:, None].expand_as(logits)
    maximum = logits.new_full((n_groups, logits.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, index, logits, reduce='amax', include_self=True)
    x = torch.exp(logits - maximum[group])
    denom = logits.new_zeros(maximum.shape)
    denom.scatter_add_(0, index, x)
    return x / denom[group].clamp_min(1e-12)


class EdgeGATv2(nn.Module):
    """Separate incoming/outgoing dynamic edge-aware attention aggregators."""
    def __init__(self, width, edge_dim, heads=4, dropout=.1):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by number of heads')
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.query = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.neighbor = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(2))
        self.edge = nn.ModuleList(nn.Linear(edge_dim, width, bias=False) for _ in range(2))
        self.attention = nn.Parameter(torch.empty(2, heads, self.head_dim))
        self.output = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.attention)

    def aggregate(self, x, center, neighbor, edge_features, direction):
        n = x.shape[0]
        if len(center) == 0:
            return x.new_zeros((n, self.width))
        q = self.query[direction](x[center]).view(-1, self.heads, self.head_dim)
        k = self.neighbor[direction](x[neighbor]).view(-1, self.heads, self.head_dim)
        e = self.edge[direction](edge_features).view(-1, self.heads, self.head_dim)
        logits = (F.leaky_relu(q + k + e, negative_slope=.2) * self.attention[direction]).sum(-1)
        alpha = F.dropout(segment_softmax(logits, center, n), p=self.dropout, training=self.training)
        messages = (k + e) * alpha.unsqueeze(-1)
        out = x.new_zeros((n, self.heads, self.head_dim))
        out.index_add_(0, center, messages)
        return out.reshape(n, self.width)

    def forward(self, x, graph, *, use_graph=True):
        if use_graph:
            outgoing = self.aggregate(x, graph['out_center'], graph['out_neighbor'], graph['out_attr'], 0)
            incoming = self.aggregate(x, graph['in_center'], graph['in_neighbor'], graph['in_attr'], 1)
        else:
            outgoing = incoming = torch.zeros_like(x)
        return self.norm(x + self.output(torch.cat([x, incoming, outgoing], dim=-1)))


class TimeEncoder(nn.Module):
    """Sinusoidal relative time encoding; explicitly an adaptation of TGAT ideas."""
    def __init__(self, width):
        super().__init__()
        if width % 2:
            raise ValueError('time encoding width must be even')
        frequencies = torch.exp(torch.linspace(0., -math.log(10000.), width // 2))
        self.register_buffer('frequencies', frequencies)

    def forward(self, relative_time):
        v = relative_time[..., None] * self.frequencies
        return torch.cat([v.sin(), v.cos()], dim=-1)


class MaskedTransformer(nn.Module):
    def __init__(self, width, heads=4, layers=2, dropout=.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout,
                                           batch_first=True, norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.time = TimeEncoder(width)

    def forward(self, x, present, relative_time):
        if x.ndim != 3 or present.shape != x.shape[:2]:
            raise ValueError('invalid temporal input')
        any_present = present.any(dim=1)
        # PyTorch attention is undefined for rows with all keys masked.
        safe = present.clone()
        safe[~any_present, 0] = True
        h = self.transformer(x + self.time(relative_time), src_key_padding_mask=~safe)
        weights = present.to(h.dtype)
        out = (h * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1.)
        return out * any_present.unsqueeze(-1)


class TemporalGATv2(nn.Module):
    """Shared GATv2 + within-hour and history Transformers; fixed output per host."""
    def __init__(self, node_dim=8, edge_dim=7, contact_dim=128, width=256, heads=4,
                 output_dim=512, dropout=.1):
        super().__init__()
        self.node = nn.Linear(node_dim, width)
        self.layers = nn.ModuleList(EdgeGATv2(width, edge_dim, heads, dropout) for _ in range(2))
        self.within_hour = MaskedTransformer(width, heads, layers=2, dropout=dropout)
        self.contacts = nn.Sequential(nn.Linear(contact_dim, width), nn.GELU(), nn.Linear(width, width))
        self.combine = nn.LayerNorm(width)
        self.history = MaskedTransformer(width, heads, layers=2, dropout=dropout)
        self.output = nn.Linear(width, output_dim)
        self.bytes_decoder = nn.Linear(width, 2)
        self.link_decoder = nn.Sequential(nn.Linear(2 * output_dim, width), nn.GELU(), nn.Linear(width, 1))

    def encode_window(self, node_features, graphs, contact_features, *, use_graph=True,
                      n_output=None, checkpoint_graph=False):
        """[N,parts,F], ragged directed edges, [N,C] → [N,width]."""
        n, parts, _ = node_features.shape
        if len(graphs) != parts:
            raise ValueError('graph parts and node features disagree')
        outputs = []
        for k, graph in enumerate(graphs):
            h = self.node(node_features[:, k])
            for layer in self.layers:
                if checkpoint_graph and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    h = checkpoint(lambda v, layer=layer, graph=graph: layer(v, graph, use_graph=use_graph),
                                   h, use_reentrant=False)
                else:
                    h = layer(h, graph, use_graph=use_graph)
            outputs.append(h if n_output is None else h[:n_output])
        h = torch.stack(outputs, dim=1)
        if n_output is not None:
            node_features, contact_features, n = node_features[:n_output], contact_features[:n_output], n_output
        present = node_features[:, :, 4:6].sum(-1) > 0
        offsets = torch.arange(parts, device=h.device, dtype=h.dtype).expand(n, -1) / max(parts - 1, 1)
        pooled = self.within_hour(h, present, offsets)
        return self.combine(pooled + self.contacts(contact_features)), present.any(1)

    def encode_history(self, values, present, offsets):
        h = self.history(values, present, offsets)
        return F.normalize(self.output(h), dim=1, eps=1e-8)

    def links(self, z, first, second):
        a, b = z[first], z[second]
        return self.link_decoder(torch.cat([(a - b).abs(), a * b], dim=-1)).squeeze(-1)
