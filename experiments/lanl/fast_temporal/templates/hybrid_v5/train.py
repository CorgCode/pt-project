"""Многовидовой графовый энкодер с временным контрастивным обучением на train."""
from collections import Counter
from datetime import datetime
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ["PT_REPO_ROOT"])
WORK = Path(os.environ["PT_RUN_ROOT"])
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize, StandardScaler
from pipeline.data import ProcessedData
from evaluation.io import load_embeddings
from evaluation.scoreboard import headline_table

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('hybrid_core', WORK / 'hybrid_v4/train.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.OUT = OUT
core.STATUS = {'state': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'pid': os.getpid()}
log, dump = core.log, core.dump
DATA = core.DATA
PREVIOUS = core.PREVIOUS
SEED = 17
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


class Encoder(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128))
        self.shortcut = nn.Linear(inputs, 128, bias=False)

    def forward(self, x):
        return F.normalize(self.net(x) + 0.2 * self.shortcut(x), dim=1)


def matrix(counter, vocab, hosts):
    cols = {t: i for i, t in enumerate(vocab)}
    rows = {h: i for i, h in enumerate(hosts)}
    triples = [(rows[h], cols[t], n) for (h, t), n in counter.items() if t in cols]
    if not triples:
        return sp.csr_matrix((len(hosts), len(vocab)))
    r, c, v = zip(*triples)
    return sp.csr_matrix((v, (r, c)), shape=(len(hosts), len(vocab)), dtype=float)


def collect(window, hosts, vocabularies=None, split_train=False):
    groups = ('a', 'b') if split_train else (window,)
    counters = {group: {kind: Counter() for kind in ('peers', 'services', 'context')} for group in groups}
    stats = {group: np.zeros((len(hosts), 12), dtype=float) for group in groups}
    hindex = pd.Index(hosts)
    wanted = set(hosts)
    midpoint = 127800.0
    read = 0
    columns = ['src', 'dst', 'proto', 'dst_port', 'ts', 'bytes', 'packets', 'resp_bytes', 'resp_packets', 'duration']
    for chunk in pd.read_csv(DATA / 'processed/flows' / f'{window}.csv', usecols=columns,
                             dtype={'src': str, 'dst': str, 'proto': str, 'dst_port': str}, chunksize=150_000):
        segments = [(g, chunk.loc[chunk.ts < midpoint] if g == 'a' else chunk.loc[chunk.ts >= midpoint]) for g in groups] if split_train else [(window, chunk)]
        for group, flows in segments:
            if flows.empty:
                continue
            svc = flows.proto + '/' + flows.dst_port
            for direction, endpoints, peers in [('o', flows.src, flows.dst), ('i', flows.dst, flows.src)]:
                mask = endpoints.isin(wanted)
                for kind, tokens in [('peers', direction + ':' + peers),
                                     ('services', ('u:' if direction == 'o' else 's:') + svc),
                                     ('context', direction + ':' + peers + '|' + svc)]:
                    use = mask
                    if vocabularies is not None:
                        use = mask & tokens.isin(vocabularies[kind])
                    counts = pd.DataFrame({'h': endpoints[use], 't': tokens[use]}).groupby(['h', 't']).size()
                    counters[group][kind].update(counts.to_dict())
                idx = hindex.get_indexer(endpoints[mask])
                offset = 0 if direction == 'o' else 6
                values = [np.ones(mask.sum()), flows.loc[mask, 'bytes'].to_numpy(), flows.loc[mask, 'packets'].to_numpy(),
                          flows.loc[mask, 'resp_bytes'].to_numpy(), flows.loc[mask, 'resp_packets'].to_numpy(), flows.loc[mask, 'duration'].to_numpy()]
                for j, value in enumerate(values):
                    stats[group][:, offset+j] += np.bincount(idx, weights=value, minlength=len(hosts))
        read += len(chunk)
        log('collect/' + window, rows=read)
    if split_train:
        counters['train'] = {k: counters['a'][k] + counters['b'][k] for k in counters['a']}
        stats['train'] = stats['a'] + stats['b']
        context_df = Counter()
        for host, token in counters['train']['context']:
            context_df[token] += 1
        candidates = [t for t, n in context_df.most_common() if n >= 2][:32768]
        vocabularies = dict(VOCABS, context=pd.Index(candidates))
    result = {}
    for group in counters:
        result[group] = {'counts': {k: matrix(c, vocabularies[k], hosts) for k, c in counters[group].items()},
                         'stats': stats[group]}
        counters[group].clear()
    del counters
    gc.collect()
    return result, vocabularies


def logcounts(x):
    x = x.copy()
    x.data = np.log1p(x.data)
    return x


def adjacencies(counts, vocab, hosts):
    # Исходящие контакты внутри когорты; веса больших хабов ослабляются.
    col_to_host = pd.Index(hosts).get_indexer([t[2:] if t.startswith('o:') else '' for t in vocab])
    mask = col_to_host >= 0
    selected = counts[:, mask].tocoo()
    a = sp.csr_matrix((np.log1p(selected.data), (selected.row, col_to_host[mask][selected.col])), shape=(len(hosts), len(hosts)))
    a.setdiag(0)
    a.eliminate_zeros()
    def norm(a):
        degree = np.asarray(a.sum(0)).ravel()
        a = a @ sp.diags(1 / np.sqrt(np.maximum(degree, 1)))
        return normalize(a, norm='l1', axis=1)
    return norm(a), norm(a.T.tocsr())


def features(view, hosts, models, vocabs):
    z = {}
    for kind in ('peers', 'services', 'context'):
        tfidf, svd = models[kind]
        z[kind] = svd.transform(normalize(tfidf.transform(logcounts(view['counts'][kind]))))
    out, inn = adjacencies(view['counts']['peers'], vocabs['peers'], hosts)
    # Направленные одношаговые и двухшаговые сообщения о сервисных ролях.
    outgoing, incoming = out @ z['services'], inn @ z['services']
    hop2 = 0.5 * (out @ incoming + inn @ outgoing)
    z['diffusion'] = np.hstack([outgoing, incoming, hop2])
    z['activity'] = np.log1p(view['stats'] / view.get('seconds', 1800))
    z['input'] = np.hstack([z[k] for k in ('peers', 'services', 'context', 'diffusion', 'activity')]).astype(np.float32)
    return z


def infer(net, x):
    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(x[i:i+512])).numpy() for i in range(0, len(x), 512)])


def main():
    global VOCABS
    cache = joblib.load(WORK / 'hybrid_v4/models.joblib')
    data = ProcessedData(DATA / 'processed')
    hosts = data.cohort
    if cache['hosts'] != hosts:
        raise ValueError('v4 model cohort differs from SSD cohort')
    VOCABS = cache['vocabs']
    dump(OUT / 'plan.json', {'architecture': 'Temporal contrastive residual MLP + directed multiscale graph encoder',
                            'fit_windows': ['train'], 'positive_pairs': 'same host in first and second halves of train',
                            'context_vocabulary': 'peer + protocol/service + direction, min_df=2, max_features=32768',
                            'graph': 'hub-normalized outgoing, incoming and two-hop service messages',
                            'encoder': [512, 256, 128], 'epochs': 60, 'checkpoints': [20, 60],
                            'loss': 'symmetric InfoNCE + behavioral/service relational distillation',
                            'selection': 'validation only, original recipe included; test only for selected recipe',
                            'seed': SEED, 'device': 'cpu', 'threads': 2})
    raw, vocabs = collect('train', hosts, split_train=True)
    raw['train']['seconds'] = 3600
    context_tfidf = TfidfTransformer().fit(logcounts(raw['train']['counts']['context']))
    context_x = normalize(context_tfidf.transform(logcounts(raw['train']['counts']['context'])))
    log('fit_context', features=context_x.shape[1])
    context_svd = TruncatedSVD(min(128, context_x.shape[1]-1, context_x.shape[0]-1), random_state=SEED).fit(context_x)
    models = {kind: cache['models'][kind, 128] for kind in ('peers', 'services')}
    models['context'] = context_tfidf, context_svd
    del cache
    values = {group: features(view, hosts, models, vocabs) for group, view in raw.items()}
    active = (raw['a']['stats'][:, [0, 6]].sum(1) >= 20) & (raw['b']['stats'][:, [0, 6]].sum(1) >= 20)
    del raw
    validation_raw, _ = collect('validation', hosts, vocabs)
    values['validation'] = features(validation_raw['validation'], hosts, models, vocabs)
    del validation_raw
    scaler = StandardScaler().fit(values['train']['input'])
    inputs = {k: np.clip(scaler.transform(v['input']), -8, 8).astype(np.float32) for k, v in values.items()}
    joblib.dump({'models': models, 'vocabs': vocabs, 'scaler': scaler, 'hosts': hosts}, OUT / 'preprocessing.joblib')
    gc.collect()
    blocks = {}
    for window in ('train', 'validation'):
        hs = hosts if window == 'train' else data.evaluable(window)
        rows = pd.Index(hosts).get_indexer(hs)
        blocks[window] = {k: core.block(values[window][k][rows]) for k in ('peers', 'services', 'context', 'diffusion')}
        blocks[window]['behavioral'] = core.block(load_embeddings(PREVIOUS / 'behavioral' / f'{window}.csv').loc[hs].to_numpy())
    window_hosts = {'train': hosts, 'validation': data.evaluable('validation')}
    records = []
    def frame(recipe, window):
        return pd.DataFrame(np.hstack([blocks[window][key] * weight for key, weight in recipe.items()]),
                            index=pd.Index(window_hosts[window], name='host_id')).add_prefix('emb_')
    def trial(recipe, checkpoint=0):
        name = f'v5_candidate_{len(records):02d}'
        log('validation', candidate=len(records)+1, recipe=recipe, checkpoint=checkpoint)
        score, result = core.evaluate(name, frame(recipe, 'train'), frame(recipe, 'validation'), 'validation')
        records.append({'name': name, 'recipe': recipe, 'checkpoint': checkpoint, 'validation_score': score})
        dump(OUT / 'search_results.json', records)
        log('validation_done', candidate=len(records), score=score, best=max(r['validation_score'] for r in records))
    base = {'behavioral': 2.0, 'peers': 1.0, 'services': 1.0}
    # Фиксированная абляция отличает вклад контекста и графа от простой склейки лидера.
    for recipe in [base, {**base, 'context': 0.5}, {**base, 'context': 1.0},
                   {**base, 'diffusion': 0.5}, {**base, 'context': 0.5, 'diffusion': 0.25}]:
        trial(recipe)
    neural_base = dict(max(records, key=lambda r: r['validation_score'])['recipe'])
    net = Encoder(inputs['train'].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
    a, b = torch.from_numpy(inputs['a']), torch.from_numpy(inputs['b'])
    eligible = np.flatnonzero(active)
    rng = np.random.default_rng(SEED)
    behavior = normalize(load_embeddings(PREVIOUS / 'behavioral/train.csv').loc[hosts].to_numpy()).astype(np.float32)
    roles = normalize(values['train']['services']).astype(np.float32)
    teacher = torch.from_numpy(np.hstack([behavior, roles]) / np.sqrt(2))
    started = time.monotonic()
    for epoch in range(1, 61):
        net.train()
        order = rng.permutation(eligible)
        losses = []
        for offset in range(0, len(order), 256):
            ix = torch.from_numpy(order[offset:offset+256])
            if len(ix) < 2:
                continue
            za, zb = net(a[ix]), net(b[ix])
            logits = za @ zb.T / 0.15
            labels = torch.arange(len(ix))
            contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            target = teacher[ix] @ teacher[ix].T
            relational = 0.5 * (F.mse_loss(za @ za.T, target) + F.mse_loss(zb @ zb.T, target))
            loss = contrastive + 0.5 * relational
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        log('neural_training', epoch=epoch, epochs=60, loss=float(np.mean(losses)),
            neural_percent=round(100 * epoch / 60, 1), elapsed_seconds=round(time.monotonic()-started),
            eta_seconds=round((time.monotonic()-started) * (60-epoch)/epoch))
        if epoch in (20, 60):
            torch.save({'state_dict': net.state_dict(), 'inputs': inputs['train'].shape[1], 'epoch': epoch}, OUT / f'encoder_epoch{epoch}.pt')
            for window in ('train', 'validation'):
                hs = window_hosts[window]
                rows = pd.Index(hosts).get_indexer(hs)
                blocks[window]['neural'] = core.block(infer(net, inputs[window])[rows])
            for wb in (2.0, 3.0):
                for weight in (0.25, 0.5, 1.0):
                    trial({**neural_base, 'behavioral': wb, 'neural': weight}, epoch)
    winner = max(records, key=lambda r: r['validation_score'])
    dump(OUT / 'selected.json', winner)
    log('test_preparation', selected=winner)
    if not (PREVIOUS / 'behavioral/test.csv').exists():
        behavior_model = joblib.load(WORK / 'behavioral_model.joblib')
        behavior_model.embed(data, 'test', data.evaluable('test')).to_csv(PREVIOUS / 'behavioral/test.csv')
    test_raw, _ = collect('test', hosts, vocabs)
    test_values = features(test_raw['test'], hosts, models, vocabs)
    del test_raw
    window_hosts['test'] = data.evaluable('test')
    rows = pd.Index(hosts).get_indexer(window_hosts['test'])
    blocks['test'] = {k: core.block(test_values[k][rows]) for k in ('peers', 'services', 'context', 'diffusion')}
    blocks['test']['behavioral'] = core.block(load_embeddings(PREVIOUS / 'behavioral/test.csv').loc[window_hosts['test']].to_numpy())
    if winner['checkpoint']:
        checkpoint = torch.load(OUT / f"encoder_epoch{winner['checkpoint']}.pt", map_location='cpu', weights_only=True)
        net.load_state_dict(checkpoint['state_dict'])
        for window in ('train', 'validation'):
            rows_window = pd.Index(hosts).get_indexer(window_hosts[window])
            blocks[window]['neural'] = core.block(infer(net, inputs[window])[rows_window])
        test_input = np.clip(scaler.transform(test_values['input']), -8, 8).astype(np.float32)
        blocks['test']['neural'] = core.block(infer(net, test_input)[rows])
    log('test_evaluation', selected=winner)
    score, result = core.evaluate('hybrid_temporal_graph_v5', frame(winner['recipe'], 'train'), frame(winner['recipe'], 'test'), 'test', 200)
    dump(OUT / 'test_result.json', result)
    baseline = json.loads((ROOT / 'scoreboard/results/hybrid_tfidf.json').read_text())
    (OUT / 'comparison.md').write_text('# Temporal graph encoder v5\n\n' + headline_table([baseline, result]) + '\nSelected recipe:\n' + json.dumps(winner, indent=2))
    log('completed', state='completed', test_score=score, finished_at=datetime.now().astimezone().isoformat())


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        log('failed', state='failed', error=repr(error))
        raise
