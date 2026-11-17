"""Causal end-to-end GATv2 pilot on source02 and historical days10–12.

Trains one fixed encoder; source vectors are generated from a prefix and
late vectors from a subsequent training window. Day13/test are never read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .episode import Episode
from .graph_utils import load_window, tensor_window
from .model import TemporalGATv2
from .objectives import episodic_identity, relation_loss, straight_through_votes, vote_rank_loss


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def mask_bytes(x, graphs, cohort_indices):
    """Hide selected node byte inputs and their incident byte edge inputs."""
    x = x.clone()
    flag = torch.zeros(len(x), dtype=torch.bool); flag[cohort_indices] = True
    x[cohort_indices, :, :2] = 0
    result = []
    for g in graphs:
        item = dict(g)
        for direction in ['out', 'in']:
            attr = g[f'{direction}_attr'].clone()
            hidden = flag[g[f'{direction}_center']] | flag[g[f'{direction}_neighbor']]
            attr[hidden, 1:3] = 0
            item[f'{direction}_attr'] = attr
        result.append(item)
    return x, result


def historical_embedding(model, encoded, days, cutoff):
    values = torch.stack([encoded[d][0] for d in days], dim=1)
    present = torch.stack([encoded[d][1] for d in days], dim=1)
    offset = torch.tensor([d-cutoff for d in days], dtype=values.dtype).expand(len(values), -1)
    return model.encode_history(values, present, offset)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--windows', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--steps', type=int, default=12)
    p.add_argument('--width', type=int, default=32); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--no-checkpoint-graph', action='store_true')
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--disable-graph', action='store_true')
    p.add_argument('--disable-bytes', action='store_true')
    p.add_argument('--disable-vote', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.batch < 8 or args.width % 4 or not 0 <= args.warmup_steps <= args.steps:
        raise ValueError('invalid steps, batch or attention width')
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed); torch.set_num_threads(2); rng = np.random.default_rng(args.seed)
    cohort_file = args.data/'late_test/processed/cohort.csv'
    cohort = pd.read_csv(cohort_file, dtype=str).host_id.tolist()
    prepared = {day: load_window(args.windows/f'day{day:02}.joblib') for day in [2,10,11,12]}
    spans = {2:(126000,129600),10:(822600,826200),11:(909000,912600),12:(995400,999000)}
    for day, graph in prepared.items():
        if graph['nodes'][:len(cohort)] != cohort or graph['manifest']['hosts_sha256'] != digest(cohort_file):
            raise ValueError(f'day{day}: cohort/order mismatch')
        if (graph['manifest']['start'], graph['manifest']['end']) != spans[day]:
            raise ValueError('unexpected train window')
        if day in [10,11,12]:
            active = graph['totals'][:len(cohort), :, 4:6].sum((1, 2)) >= 300
            cached = np.load(args.data/f'fingerprint_codebook_recent/history/future_active{day:02}.npy')
            if not np.array_equal(active, cached):
                raise ValueError(f'day{day}: activity differs from target data')
    tensors = {day: tensor_window(graph) for day, graph in prepared.items()}
    if args.disable_bytes:
        for x, graphs, _ in tensors.values():
            x[:, :, :2] = 0
            for g in graphs:
                g['out_attr'][:, 1:3] = 0
                g['in_attr'][:, 1:3] = 0
    episodes = {day: Episode(args.data, day, prepared[day]['totals'][:len(cohort),:,4:6].sum((1,2)) >= 300)
                for day in [11,12]}
    if episodes[11].hosts.tolist() != cohort or episodes[12].hosts.tolist() != cohort:
        raise ValueError('target/cohort host order mismatch')
    model = TemporalGATv2(width=args.width, output_dim=512, dropout=.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    config = {'train_windows':[2,10,11,12], 'episodes':[
        {'source':[2,10], 'target':11}, {'source':[2,10], 'target':12}, {'source':[2,10,11], 'target':12}],
        'development':13, 'benchmark':[14,15,16], 'development_or_benchmark_used_for_fitting':False,
        'architecture':'two directed edge-aware GATv2 layers, 12-slice + history Transformers',
        'width':args.width,'heads':4,'output_dim':512,'steps':args.steps,'batch':args.batch,'seed':args.seed,
        'loss_weights':{'identity':1.,'vote':.15,'links':.2,'behavior':.2,'masked_bytes':.3},
        'graph_cap_per_node_and_slice':16,'procrustes':'batch stop-gradient, rank often < 512',
        'disable_graph':args.disable_graph,'disable_bytes':args.disable_bytes,'disable_vote':args.disable_vote,
        'warmup_steps':args.warmup_steps,
        'not_full_history_02_to_12':True,
        'train_window_sha256':{str(d):prepared[d]['manifest']['flows_sha256'] for d in prepared}}
    (args.output/'protocol.json').write_text(json.dumps(config,indent=2))
    plan = [([2,10],11),([2,10],12),([2,10,11],12)]
    history = []
    for step in range(1, args.steps+1):
        beginning = time.monotonic()
        sources, target = plan[(step-1)%len(plan)]; ep = episodes[target]
        if len(ep.eligible) < args.batch:
            raise ValueError('not enough valid queries')
        queries = rng.choice(ep.eligible,args.batch,replace=False)
        q = torch.from_numpy(queries); active = torch.from_numpy(ep.active)
        masked = torch.from_numpy(rng.choice(queries,max(1,args.batch//5),replace=False))
        model.train(); optimizer.zero_grad(set_to_none=True)
        encoded = {}
        for day in sources:
            x,g,c = tensors[day]
            encoded[day] = model.encode_window(x,g,c,n_output=len(cohort),use_graph=not args.disable_graph,
                                               checkpoint_graph=not args.no_checkpoint_graph)
        source = historical_embedding(model,encoded,sources,max(sources))
        x,g,c = tensors[target]
        mx, mg = mask_bytes(x,g,masked)
        late_window, present = model.encode_window(mx,mg,c,n_output=len(cohort),use_graph=not args.disable_graph,
                                                   checkpoint_graph=not args.no_checkpoint_graph)
        late = model.encode_history(late_window[:,None,:], present[:,None], torch.zeros(len(cohort),1))
        id_loss = episodic_identity(source[q],late[q])
        if args.disable_vote or step <= args.warmup_steps:
            vote_loss = source.new_zeros(())
        else:
            votes,_ = straight_through_votes(source,q,active,ep.transposed)
            valid,positive = ep.masks(queries)
            vote_loss = vote_rank_loss(votes,valid,positive)
        behavior = relation_loss(source[q], tensors[sources[-1]][0][q], tensors[sources[-1]][2][q])
        first,second,labels = ep.link_examples(queries,rng)
        if not len(first):
            raise ValueError('no training links')
        links = F.binary_cross_entropy_with_logits(model.links(source,first,second),labels)
        raw_bytes = prepared[target]['totals'][queries, :, :2].sum(axis=1)
        lookup = {int(host): i for i,host in enumerate(queries)}
        positions = [lookup[int(i)] for i in masked]
        byte_target = torch.from_numpy(np.log1p(raw_bytes[positions]).astype(np.float32)/10.)
        byte_loss = source.new_zeros(()) if args.disable_bytes else F.smooth_l1_loss(model.bytes_decoder(late_window[masked]),byte_target)
        loss = id_loss + .15*vote_loss + .2*links + .2*behavior + .3*byte_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.); optimizer.step()
        row={'step':step,'source':sources,'target':target,'loss':float(loss.detach()),
             'identity':float(id_loss.detach()),'vote':float(vote_loss.detach()),'links':float(links.detach()),
             'behavior':float(behavior.detach()),'bytes':float(byte_loss.detach()),
             'seconds':round(time.monotonic()-beginning,2),
             'maxrss_mb':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,1)}
        history.append(row); print('training',row,flush=True)
        torch.save({'model':model.state_dict(),'config':config,'step':step}, args.output/'checkpoint.pt')
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
    (args.output/'completed.json').write_text(json.dumps({'steps':args.steps,'sha256':digest(args.output/'checkpoint.pt'),
        'test_used':False,'development_used':False,'maxrss_mb':history[-1]['maxrss_mb']},indent=2))
    print('training complete',flush=True)


if __name__ == '__main__':
    main()
