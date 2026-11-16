import numpy as np
from scipy import sparse
import torch

from experiments.source_vote_geometry.train import masks, votes_st
from experiments.temporal_donor_identity.objectives import first_positive_rank_loss


def test_hard_forward_and_gradient():
    rng = np.random.default_rng(8)
    n, partners = 15, 18
    counts = rng.integers(0, 4, size=(n, partners)).astype(np.float32)
    source = sparse.csr_matrix(counts)
    coo = source.tocoo()
    transpose = torch.sparse_coo_tensor(np.stack([coo.col, coo.row]), coo.data,
                                       (partners, n)).coalesce()
    ep = {'rows': np.arange(n), 'transpose': transpose}
    z = torch.tensor(rng.normal(size=(n, 9)).astype(np.float32), requires_grad=True)
    q = np.array([0, 4, 6])
    actual, top = votes_st(z, ep, q, .03)
    zn = torch.nn.functional.normalize(z, dim=1)
    sim = zn[q] @ zn.T
    expected = (sim.gather(1, top).clamp_min(0).unsqueeze(-1) * torch.tensor(counts)[top]).sum(1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0


def test_masks_enforce_old_and_self_exclusion():
    source = sparse.csr_matrix([[1, 0, 0, 0], [0, 0, 1, 0]])
    target = sparse.csr_matrix([[1, 1, 0, 1], [0, 1, 1, 0]])
    valid, positive = masks(source, target, np.array([3, 0]), [0, 1])
    np.testing.assert_array_equal(valid, [[False, True, True, False], [False, True, False, True]])
    np.testing.assert_array_equal(positive, [[False, True, False, False], [False, True, False, False]])
    loss = first_positive_rank_loss(torch.zeros((2, 4)), torch.tensor(valid), torch.tensor(positive))
    assert torch.isfinite(loss)
