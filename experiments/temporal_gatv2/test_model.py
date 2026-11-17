import numpy as np
import pandas as pd
import torch

from .data import prepare_window
from .model import EdgeGATv2, MaskedTransformer, TemporalGATv2, segment_softmax
from .objectives import episodic_identity, straight_through_votes, vote_rank_loss
from .train import mask_bytes


def test_exact_bidirectional_node_totals_and_external_nodes(tmp_path):
    data = pd.DataFrame([
        ['A', 'B', '6', '443', 100, 40, 10, 4, 0, 1],
        ['B', 'A', '6', '443', 30, 20, 3, 2, 301, 1],
        ['A', 'EXT', '17', '53', 7, 5, 1, 2, 600, 1],
        ['A', 'A', '6', '80', 999, 999, 9, 9, 0, 1],
    ], columns=['src', 'dst', 'proto', 'dst_port', 'bytes', 'resp_bytes', 'packets', 'resp_packets', 'ts', 'duration'])
    path = tmp_path / 'flows.csv'; data.to_csv(path, index=False)
    d = prepare_window(path, ['A', 'B'], 0, 3600, cap=10, chunk_size=2)
    assert d['nodes'] == ['A', 'B', 'EXT']
    np.testing.assert_array_equal(d['totals'][0, :, :4].sum(0), [127, 75, 13, 9])
    np.testing.assert_array_equal(d['totals'][1, :, :4].sum(0), [70, 120, 7, 12])
    np.testing.assert_array_equal(d['totals'][2, :, :4].sum(0), [5, 7, 2, 1])
    assert d['graphs'][0]['out_center'].tolist() == [0]
    assert d['graphs'][0]['in_center'].tolist() == [1]
    # Same flow has opposite per-endpoint edge byte values after log1p.
    np.testing.assert_allclose(d['graphs'][0]['out_attr'][0, 1:3], np.log1p([100, 40]))
    np.testing.assert_allclose(d['graphs'][0]['in_attr'][0, 1:3], np.log1p([40, 100]))


def test_segment_softmax_is_group_normalized_and_differentiable():
    scores = torch.tensor([[3., -2.], [0., 4.], [1., 2.]], requires_grad=True)
    group = torch.tensor([0, 0, 2])
    weights = segment_softmax(scores, group, 4)
    torch.testing.assert_close(weights[:2].sum(0), torch.ones(2))
    torch.testing.assert_close(weights[2], torch.ones(2))
    weights[0, 0].backward()
    assert scores.grad[0, 0] > 0 and scores.grad[1, 0] < 0


def test_graph_attention_and_time_mask():
    torch.manual_seed(7)
    n, width = 4, 16
    graph = {'out_center': torch.tensor([0, 0]), 'out_neighbor': torch.tensor([1, 2]),
             'out_attr': torch.rand(2, 7), 'in_center': torch.tensor([1, 2]),
             'in_neighbor': torch.tensor([0, 0]), 'in_attr': torch.rand(2, 7)}
    layer = EdgeGATv2(width, 7, heads=4, dropout=0.)
    x = torch.randn(n, width, requires_grad=True)
    actual = layer(x, graph)
    assert not torch.allclose(actual, layer(x, graph, use_graph=False))
    actual.square().sum().backward()
    assert layer.attention.grad.abs().sum() > 0
    assert layer.edge[0].weight.grad.abs().sum() > 0
    temporal = MaskedTransformer(width, dropout=0.)
    value = torch.randn(2, 3, width)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    out = temporal(value, mask, torch.zeros(2, 3))
    assert torch.isfinite(out).all() and (out[1] == 0).all()


def test_history_rejects_masked_data_invariance():
    torch.manual_seed(12)
    model = TemporalGATv2(width=16, heads=4, output_dim=8, dropout=0.).eval()
    x = torch.randn(3, 2, 16)
    mask = torch.tensor([[True, False], [True, True], [True, False]])
    times = torch.zeros(3, 2)
    y1 = model.encode_history(x, mask, times)
    x[~mask] = 1000
    y2 = model.encode_history(x, mask, times)
    torch.testing.assert_close(y1, y2)


def test_hard_vote_forward_and_procrustes_gradients():
    torch.manual_seed(5)
    n, p = 15, 8
    z = torch.randn(n, 12, requires_grad=True)
    counts = torch.randint(0, 5, (n, p)).float()
    active, q = torch.arange(n), torch.tensor([0, 3])
    votes, top = straight_through_votes(z, q, active, counts.T.to_sparse().coalesce())
    unit = torch.nn.functional.normalize(z, dim=1)
    sim = unit[q] @ unit.T
    expected = (sim.gather(1, top).clamp_min(0).unsqueeze(-1) * counts[top]).sum(1)
    torch.testing.assert_close(votes, expected, rtol=1e-5, atol=1e-5)
    valid = torch.ones((2, p), dtype=torch.bool)
    pos = torch.zeros_like(valid); pos[:, 0] = True
    loss = vote_rank_loss(votes, valid, pos) + episodic_identity(unit, unit.roll(1, 0))
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0


def test_masking_bytes_also_hides_incident_edges():
    x = torch.ones(4, 12, 8)
    graph = {'out_center': torch.tensor([0, 1, 2]), 'out_neighbor': torch.tensor([1, 2, 3]),
             'out_attr': torch.ones(3, 7), 'in_center': torch.tensor([1, 2, 3]),
             'in_neighbor': torch.tensor([0, 1, 2]), 'in_attr': torch.ones(3, 7)}
    masked_x, graphs = mask_bytes(x, [graph], torch.tensor([1]))
    assert (masked_x[1, :, :2] == 0).all() and (masked_x[0] == 1).all()
    for side in ['out', 'in']:
        assert (graphs[0][side + '_attr'][:2, 1:3] == 0).all()
        assert (graphs[0][side + '_attr'][2, 1:3] == 1).all()
        assert (graph[side + '_attr'] == 1).all()


def test_graph_layer_is_equivariant_to_host_order():
    torch.manual_seed(3)
    n = 5
    original = {'out_center': torch.tensor([0, 0, 3]), 'out_neighbor': torch.tensor([1, 2, 4]),
                'out_attr': torch.rand(3, 7), 'in_center': torch.tensor([1, 2, 4]),
                'in_neighbor': torch.tensor([0, 0, 3]), 'in_attr': torch.rand(3, 7)}
    layer = EdgeGATv2(16, 7, heads=4, dropout=0.).eval()
    x = torch.randn(n, 16)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(n)
    permuted = {key: (inverse[value] if key.endswith('_center') or key.endswith('_neighbor') else value)
                for key, value in original.items()}
    torch.testing.assert_close(layer(x[permutation], permuted), layer(x, original)[permutation])
