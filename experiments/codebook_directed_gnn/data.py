"""Aggregate bidirectional LANL rows into 6 directed graph slices.

Pass 1 includes cohort hosts and every observed direct partner, including
partners outside the cohort. Pass 2 counts all traffic of those selected
nodes (also traffic to nodes outside the graph). Attention edges have both
endpoints in the selected graph and are capped *after* exact aggregation.
No fitted vocabulary or target host ID enters the neural encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def temporal_mask(chunk, start, end):
    return ((chunk.ts >= start) & (chunk.ts < end) &
            (chunk.ts + chunk.duration < end) & (chunk.src != chunk.dst))


def discover(flows, hosts, start, end, chunk_size):
    hostset, neighbors = set(hosts), set()
    for chunk in pd.read_csv(flows, usecols=['src', 'dst', 'ts', 'duration'],
                             dtype={'src': str, 'dst': str}, chunksize=chunk_size):
        f = chunk.loc[temporal_mask(chunk, start, end)]
        neighbors.update(f.loc[f.src.isin(hostset), 'dst'].unique())
        neighbors.update(f.loc[f.dst.isin(hostset), 'src'].unique())
    return list(hosts) + sorted(neighbors - hostset)


def add_nodes(totals, part, src, dst, fwd, back, pkt_fwd, pkt_back, duration):
    """Accumulate exact node sums, no doubled bidirectional bytes."""
    for indices, arrays in ((src, (fwd, back, pkt_fwd, pkt_back)),
                            (dst, (back, fwd, pkt_back, pkt_fwd))):
        mask = indices >= 0
        for j, values in enumerate(arrays):
            np.add.at(totals[:, :, j], (indices[mask], part[mask]), values[mask])
    for col, indices in ((4, src), (5, dst)):
        mask = indices >= 0
        np.add.at(totals[:, :, col], (indices[mask], part[mask]), 1)
        np.add.at(totals[:, :, 6 + (col - 4)], (indices[mask], part[mask]), duration[mask])


def contact_hash(values, n_bins):
    return (pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64) % np.uint64(n_bins)).astype(np.int64)


def make_attention_edges(edges, center, neighbor, cap):
    """Highest-count contacts per node and interval, independent of labels."""
    e = edges.sort_values(['part', center, 'count', neighbor, 'service'],
                          ascending=[True, True, False, True, True], kind='stable')
    e = e.groupby(['part', center], sort=False).head(cap)
    graphs = []
    for part in range(6):
        g = e[e.part == part]
        forward = center == 'src_idx'
        columns = (['src_bytes', 'dst_bytes', 'src_packets', 'dst_packets'] if forward else
                   ['dst_bytes', 'src_bytes', 'dst_packets', 'src_packets'])
        raw = g[['count'] + columns + ['duration']].to_numpy(dtype=np.float32)
        attr = np.concatenate([np.log1p(raw), g[['service']].to_numpy(np.float32) / 63.], axis=1)
        graphs.append({'center': g[center].to_numpy(np.int64),
                       'neighbor': g[neighbor].to_numpy(np.int64), 'attr': attr})
    return graphs


def prepare_window(flows, hosts, start, end, *, cap=16, chunk_size=250_000):
    if end - start != 3600 or cap < 1:
        raise ValueError('expected one-hour interval and positive neighbor cap')
    nodes = discover(flows, hosts, start, end, chunk_size)
    index = pd.Index(nodes, dtype=str)
    n = len(nodes)
    totals = np.zeros((n, 6, 8), dtype=np.float64)
    contacts = np.zeros((n, 128), dtype=np.float32)
    fragments = []
    columns = ['src', 'dst', 'proto', 'dst_port', 'bytes', 'resp_bytes',
               'packets', 'resp_packets', 'ts', 'duration']
    types = {k: str for k in ('src', 'dst', 'proto', 'dst_port')}
    for chunk in pd.read_csv(flows, usecols=columns, dtype=types, chunksize=chunk_size):
        f = chunk.loc[temporal_mask(chunk, start, end)]
        if f.empty:
            continue
        src, dst = index.get_indexer(f.src), index.get_indexer(f.dst)
        part = ((f.ts.to_numpy() - start) // 600).astype(np.int64)
        byte, rev = f['bytes'].to_numpy(float), f.resp_bytes.to_numpy(float)
        pkt, rpkt, dur = f.packets.to_numpy(float), f.resp_packets.to_numpy(float), f.duration.to_numpy(float)
        if not np.isfinite(np.column_stack([byte, rev, pkt, rpkt, dur])).all() or (np.minimum.reduce([byte, rev, pkt, rpkt, dur]) < 0).any():
            raise ValueError('missing or negative counters')
        add_nodes(totals, part, src, dst, byte, rev, pkt, rpkt, dur)
        ok = src >= 0
        np.add.at(contacts, (src[ok], contact_hash(f.dst[ok], 64)), 1)
        ok = dst >= 0
        np.add.at(contacts, (dst[ok], 64 + contact_hash(f.src[ok], 64)), 1)
        inside = (src >= 0) & (dst >= 0)
        service = contact_hash(f.proto[inside] + '/' + f.dst_port[inside], 64)
        g = pd.DataFrame({'part': part[inside], 'src_idx': src[inside], 'dst_idx': dst[inside],
                          'service': service, 'src_bytes': byte[inside], 'dst_bytes': rev[inside],
                          'src_packets': pkt[inside], 'dst_packets': rpkt[inside],
                          'duration': dur[inside], 'count': 1})
        fragments.append(g.groupby(['part', 'src_idx', 'dst_idx', 'service'], sort=False, as_index=False).sum())
    if not fragments:
        raise ValueError('no eligible edges')
    all_edges = pd.concat(fragments, ignore_index=True).groupby(
        ['part', 'src_idx', 'dst_idx', 'service'], sort=False, as_index=False).sum()
    outgoing = make_attention_edges(all_edges, 'src_idx', 'dst_idx', cap)
    incoming = make_attention_edges(all_edges, 'dst_idx', 'src_idx', cap)
    graphs = [{'out_center': out['center'], 'out_neighbor': out['neighbor'], 'out_attr': out['attr'],
               'in_center': inc['center'], 'in_neighbor': inc['neighbor'], 'in_attr': inc['attr']}
              for out, inc in zip(outgoing, incoming)]
    contact_features = np.log1p(contacts)
    contact_features /= np.maximum(np.linalg.norm(contact_features, axis=1, keepdims=True), 1e-8)
    return {'nodes': nodes, 'n_cohort': len(hosts), 'totals': totals,
            'node_features': np.log1p(totals).astype(np.float32),
            'contacts': contact_features, 'graphs': graphs,
            'n_all_edges': len(all_edges), 'cap': cap}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--flows', type=Path, required=True); ap.add_argument('--hosts', type=Path, required=True)
    ap.add_argument('--start', type=int, required=True); ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--cap', type=int, default=16)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hosts = pd.read_csv(args.hosts, dtype=str).host_id.tolist()
    if len(hosts) != len(set(hosts)):
        raise ValueError('duplicate cohort host IDs')
    result = prepare_window(args.flows, hosts, args.start, args.end, cap=args.cap)
    result['manifest'] = {'start': args.start, 'end': args.end, 'flows_sha256': sha(args.flows),
                          'hosts_sha256': sha(args.hosts), 'self_flows': 'excluded', 'parts': 6,
                          'neighbor_cap_per_part_and_direction': args.cap,
                          'nodes': len(result['nodes']), 'cohort': len(hosts), 'full_edge_rows': result['n_all_edges'],
                          'byte_semantics': 'src tx=bytes,rx=resp_bytes; dst tx=resp_bytes,rx=bytes',
                          'bin_assignment': 'flow timestamp; no within-flow temporal byte allocation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, args.output, compress=1)
    print('prepared', args.output, result['manifest'], flush=True)


if __name__ == '__main__':
    main()
