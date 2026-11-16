"""Train source geometry on days 03–12; day 13 is development, never test.

Historical targets and the already-fitted base are in-sample. No test data is
read. The straight-through gradient is biased; exact evaluation is separate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.nn import functional as F

from evaluation.partners import partner_ranking
from experiments.temporal_donor_identity.objectives import first_positive_rank_loss

ROOT = Path('PT_PROJECT_ML')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def masks(source, target, own, queries):
    valid = np.ones((len(queries), source.shape[1]), dtype=bool)
    positive = np.zeros_like(valid)
    for k, i in enumerate(queries):
        valid[k, source.indices[source.indptr[i]:source.indptr[i + 1]]] = False
        if own[i] >= 0:
            valid[k, own[i]] = False
        ids = target.indices[target.indptr[i]:target.indptr[i + 1]]
        positive[k, ids] = valid[k, ids]
    return valid, positive


def episode(day, host_ids, vocab, source):
    folder = ROOT / ('fingerprint_aligned' if day <= 9 else 'fingerprint_codebook_recent/history')
    rows = np.flatnonzero(np.load(folder / f'future_active{day:02}.npy'))
    target = sparse.load_npz(folder / f'future{day:02}.npz').tocsr()[rows]
    counts = source[rows].tocsr()
    # The evaluator's candidate universe is source partners of active hosts.
    cols = np.flatnonzero(np.asarray(counts.getnnz(axis=0)).ravel())
    counts = counts[:, cols].tocsr().astype(np.float32)
    target = target[:, cols].tocsr()
    own = pd.Index(vocab[cols]).get_indexer(host_ids[rows])
    eligible = []
    for i in range(len(rows)):
        v, p = masks(counts, target, own, [i])
        if p.any():
            eligible.append(i)
    coo = counts.tocoo()
    st = torch.sparse_coo_tensor(np.stack([coo.col, coo.row]), coo.data,
                                 (counts.shape[1], counts.shape[0])).coalesce()
    return {'day': day, 'rows': rows, 'counts': counts, 'target': target,
            'own': own, 'eligible': np.asarray(eligible), 'transpose': st}


def votes_st(z, ep, queries, temp):
    """Exact forward top10, sigmoid top10 approximate backward; weighted counts."""
    rows = torch.as_tensor(ep['rows'], dtype=torch.long)
    q = torch.as_tensor(queries, dtype=torch.long)
    x = F.normalize(z[rows], dim=1)
    sim = x[q] @ x.T
    allowed = torch.ones_like(sim, dtype=torch.bool)
    allowed[torch.arange(len(q)), q] = False
    masked = sim.masked_fill(~allowed, -1e6)
    top = masked.topk(10, dim=1).indices
    hard = torch.zeros_like(sim).scatter(1, top, 1.0)
    # Numerical bisection sets soft mass to 10; implicit derivative preserves it.
    with torch.no_grad():
        lo = masked.masked_fill(~allowed, float('inf')).min(1, keepdim=True).values - 1
        hi = masked.max(1, keepdim=True).values + 1
        for _ in range(24):
            mid = (lo + hi) / 2
            mass = (torch.sigmoid((masked - mid) / temp) * allowed).sum(1, keepdim=True)
            lo, hi = torch.where(mass > 10, mid, lo), torch.where(mass > 10, hi, mid)
        threshold = (lo + hi) / 2
        p = torch.sigmoid((masked - threshold) / temp) * allowed
        derivative = p * (1 - p)
    implicit = (sim * derivative).sum(1, keepdim=True) / derivative.sum(1, keepdim=True).clamp_min(1e-12)
    threshold = threshold + implicit - implicit.detach()
    soft = torch.sigmoid((masked - threshold) / temp) * allowed
    gate = hard + soft - soft.detach()
    weights = gate * sim.clamp_min(0)
    return torch.sparse.mm(ep['transpose'], weights.T).T, top


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--lr', type=float, default=.002)
    ap.add_argument('--temperature', type=float, default=.03)
    ap.add_argument('--regularization', type=float, default=.5)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    hosts = pd.Index(json.loads((ROOT / 'fingerprint_aligned/hosts.json').read_text()), dtype=str)
    vocab = pd.Index(json.loads((ROOT / 'fingerprint_transfer/vocabulary.json').read_text()), dtype=str)
    source_path = ROOT / 'fingerprint_transfer/source0.npz'
    source = sparse.load_npz(source_path).tocsr()
    base_path = ROOT / 'fingerprint_codebook_recent/candidates/codebook1_e400/embeddings.csv'
    base = pd.read_csv(base_path, dtype={'host_id': str}).set_index('host_id').loc[hosts]
    base_t = F.normalize(torch.tensor(base.to_numpy(np.float32)), dim=1).detach()
    delta = torch.nn.Parameter(torch.zeros_like(base_t))
    opt = torch.optim.Adam([delta], lr=args.lr)
    ref = ROOT / 'fingerprint_recent/reference_validation/validation'
    partners = pd.read_csv(ref / 'partners.csv', dtype={'host_id': str, 'partner': str, 'window': str})
    active = pd.Index(pd.read_csv(ref / 'eval_hosts.csv', dtype=str).host_id)
    active_pos = hosts.get_indexer(active)
    # Ref source counts must match the history source matrix exactly.
    src = partners[partners.window == 'source']
    hi, pi = hosts.get_indexer(src.host_id), vocab.get_indexer(src.partner)
    if (hi < 0).any() or (pi < 0).any() or not np.array_equal(np.asarray(source[hi, pi]).ravel(), src['count'].to_numpy()):
        raise ValueError('source counts do not match official day13 reference')
    config = vars(args).copy(); config['output'] = str(args.output)
    config.update({'train_days': list(range(3, 13)), 'development_day': 13,
                   'base_in_sample': True, 'test_read': False,
                   'source_sha256': digest(source_path), 'base_sha256': digest(base_path),
                   'reference_sha256': digest(ref / 'partners.csv'), 'script_sha256': digest(Path(__file__))})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    episodes = [episode(day, hosts, vocab, source) for day in range(3, 13)]
    print('episodes', [(e['day'], len(e['rows']), len(e['eligible'])) for e in episodes], flush=True)
    assessments = []

    def assess(step):
        z = F.normalize(base_t + delta, dim=1).detach().numpy()
        frame = pd.DataFrame(z[active_pos].astype(np.float64), index=active)
        value = partner_ranking(frame, partners, k=10)['new']['embedding']
        row = {'step': step, 'mrr': value['mrr'], 'n_hosts': value['n_hosts'],
               'mean_delta_norm': float(delta.detach().norm(dim=1).mean())}
        assessments.append(row)
        np.save(args.output / f'source_step{step:04}.npy', z)
        (args.output / 'development.json').write_text(json.dumps(assessments, indent=2))
        print('development', row, flush=True)

    assess(0)
    for step in range(1, args.steps + 1):
        ep = episodes[(step - 1) % len(episodes)]
        q = rng.choice(ep['eligible'], min(args.batch, len(ep['eligible'])), replace=False)
        valid, positive = masks(ep['counts'], ep['target'], ep['own'], q)
        z = base_t + delta
        vote, _ = votes_st(z, ep, q, args.temperature)
        # Per-query scale is held constant for backprop, avoiding overwhelming
        # the rank surrogate with raw counts spanning several orders of magnitude.
        scale = vote.detach().amax(dim=1, keepdim=True).clamp_min(1.)
        loss_vote = first_positive_rank_loss(vote / scale, torch.from_numpy(valid),
                                             torch.from_numpy(positive), temperature=.05)
        reg = delta.square().sum(1).mean()
        loss = loss_vote + args.regularization * reg
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([delta], 1.); opt.step()
        if step % 10 == 0 or step == 1:
            print('step', step, 'day', ep['day'], 'loss', round(loss_vote.item(), 5), 'reg', round(reg.item(), 5), flush=True)
        if step in {args.steps // 2, args.steps}:
            assess(step)
    (args.output / 'completed.json').write_text(json.dumps({'steps': args.steps, 'selection': 'none; report all checkpoints',
                                                            'development': assessments}, indent=2))


if __name__ == '__main__':
    main()
