"""Extract directional node totals and sparse directed graphs from a flow window."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def add_flow_chunk(totals, counts, byte_graph, chunk, hosts):
    """Accumulate traffic; bytes/resp_bytes travel src→dst and dst→src.

    Self-flows are excluded. Node totals include outside-cohort endpoints;
    sparse graphs contain only edges between hosts in the fixed cohort.
    """
    chunk = chunk.loc[chunk.src != chunk.dst]
    n = len(hosts)
    src = hosts.get_indexer(chunk.src)
    dst = hosts.get_indexer(chunk.dst)
    fwd = chunk['bytes'].to_numpy(dtype=np.float64)
    rev = chunk['resp_bytes'].to_numpy(dtype=np.float64)
    pf = chunk['packets'].to_numpy(dtype=np.float64)
    pr = chunk['resp_packets'].to_numpy(dtype=np.float64)
    if not np.isfinite(np.column_stack((fwd, rev, pf, pr))).all() or min(fwd.min(initial=0), rev.min(initial=0), pf.min(initial=0), pr.min(initial=0)) < 0:
        raise ValueError('negative or missing byte/packet counts')
    for ids, values in ((src, (fwd, rev, pf, pr)), (dst, (rev, fwd, pr, pf))):
        mask = ids >= 0
        for col, value in enumerate(values):
            totals[:, col] += np.bincount(ids[mask], weights=value[mask], minlength=n)
    for col, ids in ((4, src), (5, dst)):
        mask = ids >= 0
        totals[:, col] += np.bincount(ids[mask], minlength=n)
    both = (src >= 0) & (dst >= 0)
    a, b = src[both], dst[both]
    counts += sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    byte_graph += sparse.coo_matrix((fwd[both], (a, b)), shape=(n, n)).tocsr()
    byte_graph += sparse.coo_matrix((rev[both], (b, a)), shape=(n, n)).tocsr()
    return totals, counts, byte_graph


def extract(path, host_ids, *, chunksize=250_000, start=None, end=None):
    hosts = pd.Index(host_ids, dtype=str)
    if hosts.has_duplicates:
        raise ValueError('duplicate hosts')
    totals = np.zeros((len(hosts), 6), dtype=np.float64)
    counts = sparse.csr_matrix((len(hosts), len(hosts)), dtype=np.float64)
    byte_graph = counts.copy()
    columns = ['src', 'dst', 'bytes', 'resp_bytes', 'packets', 'resp_packets']
    if start is not None or end is not None:
        if start is None or end is None or start >= end:
            raise ValueError('both valid time bounds are required')
        columns += ['ts', 'duration']
    for chunk in pd.read_csv(path, usecols=columns, dtype={'src': str, 'dst': str}, chunksize=chunksize):
        if start is not None:
            chunk = chunk[(chunk.ts >= start) & (chunk.ts < end) & (chunk.ts + chunk.duration < end)]
        totals, counts, byte_graph = add_flow_chunk(totals, counts, byte_graph, chunk, hosts)
    counts.eliminate_zeros(); byte_graph.eliminate_zeros()
    return totals, counts, byte_graph


def row_norm(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    den = np.asarray(matrix.sum(axis=1)).ravel()
    return (sparse.diags(1 / np.maximum(den, 1e-12)) @ matrix).tocsr()


def features(totals, counts, byte_graph, *, include_bytes=True):
    """Host features and out/in message matrices for flows and byte volume."""
    unique_out = np.diff(counts.indptr)
    unique_in = np.diff(counts.tocsc().indptr)
    numeric = np.log1p(np.column_stack((totals, unique_out, unique_in))).astype(np.float32)
    if not include_bytes:
        numeric[:, :2] = 0
    adjacency = counts.copy(); adjacency.data = np.log1p(adjacency.data)
    byte_adjacency = byte_graph.copy(); byte_adjacency.data = np.log1p(byte_adjacency.data)
    if not include_bytes:
        byte_adjacency = sparse.csr_matrix(byte_adjacency.shape, dtype=np.float32)
    graphs = [row_norm(adjacency), row_norm(adjacency.T),
              row_norm(byte_adjacency), row_norm(byte_adjacency.T)]
    active = totals[:, 4] + totals[:, 5]
    return numeric, graphs, active


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--flows', type=Path, required=True)
    ap.add_argument('--hosts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start', type=float, required=True)
    ap.add_argument('--end', type=float, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    hosts = pd.read_csv(args.hosts, dtype=str).host_id.tolist()
    args.out.mkdir(parents=True)
    totals, counts, byte_graph = extract(args.flows, hosts, start=args.start, end=args.end)
    np.save(args.out / 'totals.npy', totals)
    sparse.save_npz(args.out / 'counts.npz', counts)
    sparse.save_npz(args.out / 'bytes.npz', byte_graph)
    (args.out / 'manifest.json').write_text(json.dumps({'flows': str(args.flows), 'flows_sha256': digest(args.flows),
        'hosts_sha256': digest(args.hosts), 'start': args.start, 'end': args.end, 'n_hosts': len(hosts),
        'count_edges': counts.nnz, 'byte_edges': byte_graph.nnz,
        'active_300': int(((totals[:, 4] + totals[:, 5]) >= 300).sum()),
        'tx_rx_definition': 'src tx+=bytes rx+=resp_bytes; dst tx+=resp_bytes rx+=bytes'}, indent=2))
    print('prepared', args.out, 'edges', counts.nnz, 'active', int(((totals[:, 4]+totals[:, 5])>=300).sum()), flush=True)


if __name__ == '__main__':
    main()
