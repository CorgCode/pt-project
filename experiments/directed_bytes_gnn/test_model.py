import numpy as np
import pytest
torch = pytest.importorskip("torch")

from .model import DirectedByteGNN, infer, load_state, save_state


def test_graph_receives_gradients_and_input_changes_output():
    torch.manual_seed(17)
    n = 12
    idx = torch.arange(n)
    graph = torch.sparse_coo_tensor(torch.stack((idx, (idx + 1) % n)),
                                    torch.ones(n), (n, n)).coalesce()
    empty = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
                                    torch.empty(0), (n, n)).coalesce()
    x = torch.randn(n, 8)
    model = DirectedByteGNN()
    out = model(x, [graph, graph, graph, graph])
    altered = model(x, [empty, empty, empty, empty])
    assert not torch.allclose(out, altered)
    loss = (out[0] - out[1]).square().sum()
    loss.backward()
    assert model.first.neighbor_maps[0].weight.grad.abs().sum() > 0
    assert model.first.gates.grad.abs().sum() > 0
    torch.testing.assert_close(out.norm(dim=1), torch.ones(n))
    for channel in range(4):
        graphs = [empty] * 4
        graphs[channel] = graph
        assert not torch.allclose(model(x, graphs), altered), f'channel {channel} ignored'


def test_checkpoint_roundtrip_preserves_output(tmp_path):
    torch.manual_seed(1)
    n = 6
    idx = torch.arange(n)
    graph = torch.sparse_coo_tensor(torch.stack((idx, (idx + 1) % n)),
                                    torch.ones(n), (n, n)).coalesce()
    net = DirectedByteGNN()
    state = {'mode': 'gnn', 'steps': 1, 'seed': 17, 'weights': net.state_dict(),
             'mean': np.zeros(8, np.float32), 'std': np.ones(8, np.float32), 'history': []}
    obs = {'x': np.random.default_rng(1).normal(size=(n, 8)).astype(np.float32),
           'graphs': [graph] * 4}
    expected = infer(state, obs)
    path = tmp_path / 'model.pt'
    save_state(state, path)
    np.testing.assert_array_equal(infer(load_state(path), obs), expected)
