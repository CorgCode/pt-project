import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
torch = pytest.importorskip("torch")
from torch.nn import functional as F

from evaluation.partners import partner_ranking
from .history import Observation, contact_profiles
from .model import ModelConfig, TemporalDonorIdentity, source_vectors
from .objectives import hard_votes, soft_votes, first_positive_rank_loss, episodic_identity_loss


@pytest.mark.parametrize('correction', ['linear', 'mlp'])
def test_initial_residual_exact_baseline_and_permutation(correction):
    torch.manual_seed(4)
    c = ModelConfig(base_dim=5, context_dim=2, hidden_dim=8, correction=correction)
    model = TemporalDonorIdentity(c)
    late, context = torch.randn(12, 5), torch.randn(12, 2)
    assert torch.equal(model.late_vectors(late, context), late)
    with torch.no_grad():
        last = model.identity if correction == 'linear' else model.identity[-1]
        last.weight.normal_()
    order = torch.randperm(12)
    assert torch.allclose(model.late_vectors(late[order], context[order]),
                          model.late_vectors(late, context)[order])


def test_source_baseline_is_exact_and_zero_profile_remains_zero():
    base = torch.tensor([[3., 4.], [0., 2.]])
    profiles = [torch.tensor([[0., 0.], [3., 4.]]) for _ in range(3)]
    assert torch.equal(source_vectors(base, profiles, [1., 0., 0., 0.]), base)
    output = source_vectors(base, profiles, [.7, .1, .1, .1])
    assert torch.equal(output[0, 2:], torch.zeros(6))
    assert output.shape == (2, 8)
    with pytest.raises(ValueError):
        source_vectors(base, profiles, [.7, .1, .1, .2])


def test_history_distinct_day_periodic_decay_and_cutoff():
    a = sparse.csr_matrix([[1., 0.], [0., 0.]])
    b = sparse.csr_matrix([[3., 2.], [0., 0.]])
    obs = [Observation(10, 12, 10, a), Observation(10, 13, 11, a), Observation(12, 12, 12, b)]
    days, recent, period = contact_profiles(obs, cutoff=12, target_hour=12, half_life_days=2)
    assert np.allclose(days.toarray()[0], np.array([2., 1.]) / np.sqrt(5))
    assert recent[0, 0] > recent[0, 1] > 0 and period[0, 0] > period[0, 1] > 0
    assert days[1].nnz == recent[1].nnz == period[1].nnz == 0
    with pytest.raises(ValueError, match='future'):
        contact_profiles(obs, cutoff=11, target_hour=12)


def test_episodic_procrustes_matches_scipy_and_baseline_gradient():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(20, 5)); b = rng.normal(size=(20, 5))
    ai, ei = np.arange(10), np.arange(10, 20)
    a_t = torch.tensor(a, dtype=torch.float64); b_t = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    loss, logits = episodic_identity_loss(a_t, b_t, torch.tensor(ai), torch.tensor(ei), temperature=.2)
    ma, mb = a[ai].mean(0), b[ai].mean(0)
    rotation, _ = orthogonal_procrustes(b[ai] - mb, a[ai] - ma)
    x, y = a[ei] - ma, (b[ei] - mb) @ rotation
    expected = (x / np.linalg.norm(x, axis=1, keepdims=True)) @ (y / np.linalg.norm(y, axis=1, keepdims=True)).T / .2
    assert np.allclose(logits.detach().numpy(), expected, atol=1e-10)
    loss.backward(); assert torch.isfinite(b_t.grad).all()
    with pytest.raises(ValueError, match='disjoint'):
        episodic_identity_loss(a_t, b_t, torch.tensor(ai), torch.tensor(ai))


def test_hard_vote_matches_official_partner_mrr_with_ties():
    rng = np.random.default_rng(4)
    n = 16; names = [f'h{i:02}' for i in range(n)]
    z = rng.normal(size=(n, 5)); counts = rng.integers(0, 3, size=(n, 7))
    src = [dict(host_id=names[i], partner=f'p{p}', count=int(counts[i, p]), window='source')
           for i in range(n) for p in range(7) if counts[i, p]]
    tgt = [dict(host_id=names[i], partner=f'p{p}', count=1, window='target')
           for i in range(n) for p in range(7) if not counts[i, p]]
    official = partner_ranking(pd.DataFrame(z, index=names), pd.DataFrame(src + tgt), k=10)
    votes, _, _ = hard_votes(torch.tensor(z), torch.tensor(counts, dtype=torch.float64),
                             torch.arange(n), torch.ones(n, dtype=torch.bool), k=10)
    rr = []
    for i, row in enumerate(votes.numpy()):
        valid = counts[i] == 0
        if not valid.any(): continue
        best = row[valid].max()
        rr.append(1 / (np.sum(valid & (row > best)) + (np.sum(valid & (row == best)) + 1) / 2))
    assert np.mean(rr) == pytest.approx(official['new']['embedding']['mrr'])


def test_soft_votes_gradients():
    torch.manual_seed(7)
    z = torch.randn(12, 5, requires_grad=True); counts = torch.rand(12, 6)
    active = torch.ones(12, dtype=torch.bool); active[-1] = False
    votes = soft_votes(z, counts, torch.tensor([0, 1]), active, k=3)
    positive = torch.tensor([[True, False, False, False, False, False], [False, True, False, False, False, False]])
    loss = first_positive_rank_loss(votes, torch.ones_like(positive), positive); loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_grid_uses_exact_evaluator_and_baseline_dimension():
    from .donor_grid import evaluate_grid
    rng = np.random.default_rng(9); n = 16
    names = [f'h{i:02}' for i in range(n)]; base = rng.normal(size=(n, 5))
    profiles = [sparse.csr_matrix(rng.normal(size=(n, 3))) for _ in range(3)]
    rows = []
    for i, h in enumerate(names):
        rows.append({'host_id': h, 'partner': 'p' + str(i % 4), 'count': i + 1, 'window': 'source'})
        rows.append({'host_id': h, 'partner': 'p' + str((i + 1) % 4), 'count': 1, 'window': 'target'})
    partners = pd.DataFrame(rows)
    actual = evaluate_grid(base, profiles, names, partners, names,
                           [[1, 0, 0, 0], [.7, .1, .1, .1]])
    expected = partner_ranking(pd.DataFrame(base, index=names), partners)['new']['embedding']['mrr']
    assert actual[0]['dimension'] == 5 and actual[1]['dimension'] == 14
    assert actual[0]['new_partners_mrr'] == pytest.approx(expected)


def test_regularized_procrustes_svd_gradients_are_finite():
    torch.manual_seed(42)
    source = torch.randn(24, 6, dtype=torch.float64)
    late = torch.randn(24, 6, dtype=torch.float64, requires_grad=True)
    loss, _ = episodic_identity_loss(source, late, torch.arange(12), torch.arange(12, 24),
                                     alignment_gradient='svd', regularization=.01,
                                     baseline=late.detach().clone(), residual_weight=.1)
    loss.backward()
    assert torch.isfinite(late.grad).all()


def test_no_bias_mode_exact_baseline_and_backward_compatibility():
    config = ModelConfig(base_dim=5, context_dim=2, hidden_dim=8, identity_bias=False)
    model = TemporalDonorIdentity(config)
    x, context = torch.randn(5, 5), torch.randn(5, 2)
    assert model.identity.bias is None
    assert torch.equal(model.late_vectors(x, context), x)
    assert ModelConfig(**{'base_dim':5, 'context_dim':2, 'hidden_dim':8, 'correction':'linear'}).identity_bias
