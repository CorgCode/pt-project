"""Source02 voting data and causal historical target windows."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch


def sparse_tensor(matrix):
    x = matrix.tocoo()
    return torch.sparse_coo_tensor(np.vstack([x.row, x.col]), x.data.astype(np.float32),
                                   x.shape, check_invariants=True).coalesce()


class Episode:
    def __init__(self, data_root, target_day, target_active):
        root = Path(data_root)
        self.hosts = pd.Index(json.loads((root/'fingerprint_aligned/hosts.json').read_text()), dtype=str)
        vocab = pd.Index(json.loads((root/'fingerprint_transfer/vocabulary.json').read_text()), dtype=str)
        if self.hosts.has_duplicates or vocab.has_duplicates:raise ValueError('Duplicate cohort/vocabulary IDs')
        self.order_sha256=hashlib.sha256(json.dumps([self.hosts.tolist(),vocab.tolist()]).encode()).hexdigest()
        if np.asarray(target_active).shape!=(len(self.hosts),):raise ValueError('Wrong activity mask')
        source = sparse.load_npz(root/'fingerprint_transfer/source0.npz').tocsr()
        folder = root/('fingerprint_aligned' if target_day <= 9 else 'fingerprint_codebook_recent/history')
        target = sparse.load_npz(folder/f'future{target_day:02}.npz').tocsr()
        if source.shape != target.shape or source.shape != (len(self.hosts), len(vocab)):
            raise ValueError('cohort/vocabulary order mismatch')
        self.active = np.flatnonzero(target_active)
        active_counts = source[self.active]
        columns = np.flatnonzero(np.asarray(active_counts.getnnz(axis=0)).ravel())
        self.columns = columns
        self.vocabulary = vocab[columns]
        self.source = source[:, columns].tocsr()
        self.target = target[:, columns].tocsr()
        self.own = self.vocabulary.get_indexer(self.hosts)
        self.transposed = sparse_tensor(self.source[self.active].T)
        self.eligible = []
        for q in self.active:
            pos = self.target.indices[self.target.indptr[q]:self.target.indptr[q+1]]
            old = self.source.indices[self.source.indptr[q]:self.source.indptr[q+1]]
            positive = np.setdiff1d(pos, old, assume_unique=False)
            positive = positive[positive != self.own[q]]
            if len(positive):
                self.eligible.append(q)
        self.eligible = np.asarray(self.eligible, dtype=np.int64)
        # Maps observed source-partner IDs to cohort candidates for auxiliary link loss.
        self.partner_host = self.hosts.get_indexer(self.vocabulary)
        self.host_partner = self.vocabulary.get_indexer(self.hosts)

    def masks(self, queries):
        valid = np.ones((len(queries), len(self.columns)), dtype=bool)
        positive = np.zeros_like(valid)
        for r, q in enumerate(queries):
            valid[r, self.source.indices[self.source.indptr[q]:self.source.indptr[q+1]]] = False
            if self.own[q] >= 0:
                valid[r, self.own[q]] = False
            p = self.target.indices[self.target.indptr[q]:self.target.indptr[q+1]]
            positive[r, p] = valid[r, p]
        return torch.from_numpy(valid), torch.from_numpy(positive)

    def link_examples(self, queries, rng):
        first, second, labels = [], [], []
        active_set = set(self.active.tolist())
        for q in queries:
            p = self.target.indices[self.target.indptr[q]:self.target.indptr[q+1]]
            eligible = [int(i) for i in self.partner_host[p] if i >= 0 and i != q and i in active_set]
            if not eligible:
                continue
            positive = int(rng.choice(eligible))
            forbidden = set(p.tolist())
            negatives = [int(i) for i in rng.choice(self.active, min(32, len(self.active)), replace=False)
                         if i != q and (self.host_partner[i] < 0 or self.host_partner[i] not in forbidden)]
            if not negatives:
                continue
            first.extend([int(q), int(q)]); second.extend([positive, negatives[0]]); labels.extend([1., 0.])
        return torch.tensor(first, dtype=torch.long), torch.tensor(second, dtype=torch.long), torch.tensor(labels)
