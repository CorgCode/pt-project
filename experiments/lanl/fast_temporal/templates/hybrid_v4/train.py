"""Подбор гибрида на validation, однократная оценка победителя на test."""
from collections import Counter
from datetime import datetime
import gc
import itertools
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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from evaluation.evaluate import run
from evaluation.identity import identity_retrieval
from evaluation.io import load_embeddings
from evaluation.scoreboard import overall_score, headline_table
from pipeline.data import ProcessedData

OUT = Path(__file__).resolve().parent
DATA = Path(os.environ["PT_LANL_DIR"])
PREVIOUS = WORK / 'base_embeddings'
STATUS = {'state': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'pid': os.getpid()}


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    os.replace(temp, path)


def log(phase, **kwargs):
    STATUS.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **kwargs)
    dump(OUT / 'status.json', STATUS)
    print(f'[{datetime.now():%H:%M:%S}] {phase}: {kwargs}', flush=True)


def aggregate(window, hosts, vocabularies=None):
    counters = {'peers': Counter(), 'services': Counter()}
    seen = 0
    wanted = set(hosts)
    for flows in pd.read_csv(DATA / 'processed/flows' / f'{window}.csv',
                             usecols=['src', 'dst', 'proto', 'dst_port'], dtype=str, chunksize=200_000):
        svc = flows['proto'] + '/' + flows['dst_port']
        for kind, out, inn in [('peers', 'o:' + flows['dst'], 'i:' + flows['src']),
                               ('services', 'u:' + svc, 's:' + svc)]:
            for endpoints, tokens in [(flows['src'], out), (flows['dst'], inn)]:
                mask = endpoints.isin(wanted)
                if vocabularies is not None:
                    mask &= tokens.isin(vocabularies[kind])
                grouped = pd.DataFrame({'h': endpoints[mask], 't': tokens[mask]}).groupby(['h', 't']).size()
                counters[kind].update(grouped.to_dict())
        seen += len(flows)
        log(f'counts/{window}', rows=seen)
    matrices, vocab = {}, {}
    for kind, counter in counters.items():
        if vocabularies is None:
            df = Counter()
            for host, token in counter:
                df[token] += 1
            vocab[kind] = pd.Index(sorted(token for token, count in df.items() if count >= 2))
        else:
            vocab[kind] = vocabularies[kind]
        cols = {token: i for i, token in enumerate(vocab[kind])}
        rows = {host: i for i, host in enumerate(hosts)}
        entries = [(rows[h], cols[t], np.log1p(n)) for (h, t), n in counter.items() if t in cols]
        ri, ci, values = zip(*entries)
        matrices[kind] = sp.csr_matrix((values, (ri, ci)), shape=(len(hosts), len(cols)))
        sp.save_npz(OUT / f'counts_{kind}_{window}.npz', matrices[kind])
        counter.clear()
        del entries
    gc.collect()
    return matrices, vocab


def block(x):
    x = x - x.mean(axis=0)
    return x / max(1e-12, np.sqrt(np.square(x).sum(axis=1).mean()))


def evaluate(name, train, later, window, bootstrap=0):
    path = OUT / 'candidates' / name
    path.mkdir(parents=True, exist_ok=True)
    train.to_csv(path / 'embeddings.csv', float_format='%.8g')
    later.to_csv(path / f'{window}.csv', float_format='%.8g')
    result = run(path / 'embeddings.csv', DATA / 'reference' / window, name,
                 n_boot=bootstrap, seed=0,
                 fit_reference_dir=DATA / 'reference/validation' if window == 'test' else None)
    identity = identity_retrieval(train.loc[later.index], later, alignment='procrustes', seed=0, n_boot=bootstrap)
    result['metrics']['identity_top1'] = identity['top1']
    result['metrics_ci']['identity_top1'] = identity.get('top1_ci')
    result['metric_status']['identity'] = 'ok'
    result['metric_details']['identity'] = identity
    result['summary'].append({'group': 'Узнаваемость', 'metric': 'identity_top1', 'title': 'Узнаваемость, Top-1',
                              'value': identity['top1'], 'chance': identity['chance_top1'], 'ci': identity.get('top1_ci')})
    score = float(overall_score([result], {s['metric']: s['title'] for s in result['summary']})[name])
    dump(path / f'{window}_result.json', result)
    return score, result


def main():
    data = ProcessedData(DATA / 'processed')
    hosts = {'train': data.cohort, 'validation': data.evaluable('validation'), 'test': data.evaluable('test')}
    train_counts, vocabs = aggregate('train', hosts['train'])
    val_counts, _ = aggregate('validation', hosts['validation'], vocabs)
    blocks = {window: {'behavioral': block(load_embeddings(PREVIOUS / 'behavioral' / f'{window}.csv').loc[hs].to_numpy())}
              for window, hs in hosts.items() if window != 'test'}
    models = {}
    for kind in ('peers', 'services'):
        tfidf = TfidfTransformer().fit(train_counts[kind])
        for dim in (128, 256):
            log('fit_svd', kind=kind, dim=dim)
            effective = min(dim, train_counts[kind].shape[0] - 1, train_counts[kind].shape[1] - 1)
            svd = TruncatedSVD(effective, random_state=0).fit(normalize(tfidf.transform(train_counts[kind])))
            models[kind, dim] = (tfidf, svd)
            for window, counts in [('train', train_counts), ('validation', val_counts)]:
                blocks[window][f'{kind}{dim}'] = block(svd.transform(normalize(tfidf.transform(counts[kind]))))
    joblib.dump({'models': models, 'vocabs': vocabs, 'hosts': hosts['train']}, OUT / 'models.joblib')
    del train_counts, val_counts
    gc.collect()
    for window in blocks:
        for kind in ('node2vec', 'graph_spectral'):
            blocks[window][kind] = block(load_embeddings(PREVIOUS / kind / f'{window}.csv').loc[hosts[window]].to_numpy())
    def frame(recipe, window):
        return pd.DataFrame(np.hstack([blocks[window][key] * weight for key, weight in recipe.items()]),
                            index=pd.Index(hosts[window], name='host_id')).add_prefix('emb_')
    # Сетка зафиксирована до чтения test. Исходный рецепт (128, 2, 1, 1) включён.
    recipes = [{'behavioral': float(wb), f'peers{dim}': 1.0, f'services{dim}': float(ws)}
               for dim, wb, ws in itertools.product((128, 256), (1, 2, 3, 4), (0.5, 1, 2))]
    dump(OUT / 'search_plan.json', {'base_candidates': recipes, 'second_stage': 'best + node2vec or spectral, weights 0.25, 0.5, 1',
                                   'selection': 'validation overall score including procrustes identity',
                                   'test_policy': 'evaluate selected recipe once; historical test already known from project'})
    records = []
    def trial(recipe):
        name = f'candidate_{len(records):02d}'
        log('validation', candidate=len(records) + 1, planned_candidates=30, recipe=recipe)
        score, result = evaluate(name, frame(recipe, 'train'), frame(recipe, 'validation'), 'validation')
        records.append({'name': name, 'recipe': recipe, 'validation_score': score})
        dump(OUT / 'search_results.json', records)
        log('validation_done', candidate=len(records), score=score, best=max(r['validation_score'] for r in records))
    for recipe in recipes:
        trial(recipe)
    winner = max(records, key=lambda row: row['validation_score'])
    best_base = dict(winner['recipe'])
    for kind, weight in itertools.product(('node2vec', 'graph_spectral'), (0.25, 0.5, 1.0)):
        trial({**best_base, kind: weight})
    winner = max(records, key=lambda row: row['validation_score'])
    dump(OUT / 'selected.json', winner)
    log('test_counts', selected=winner)
    test_counts, _ = aggregate('test', hosts['test'], vocabs)
    blocks['test'] = {'behavioral': block(load_embeddings(PREVIOUS / 'behavioral/test.csv').loc[hosts['test']].to_numpy())}
    for key in winner['recipe']:
        if key.startswith(('peers', 'services')):
            kind = 'peers' if key.startswith('peers') else 'services'
            dim = int(key[len(kind):])
            tfidf, svd = models[kind, dim]
            blocks['test'][key] = block(svd.transform(normalize(tfidf.transform(test_counts[kind]))))
        elif key != 'behavioral':
            blocks['test'][key] = block(load_embeddings(PREVIOUS / key / 'test.csv').loc[hosts['test']].to_numpy())
    log('test_evaluation', selected=winner)
    score, result = evaluate('hybrid_tfidf_v4', frame(winner['recipe'], 'train'), frame(winner['recipe'], 'test'), 'test', 200)
    dump(OUT / 'test_result.json', result)
    baseline = json.loads((ROOT / 'scoreboard/results/hybrid_tfidf.json').read_text())
    (OUT / 'comparison.md').write_text('# Hybrid v4 vs existing leader\n\n' + headline_table([baseline, result]) + '\nSelected recipe:\n' + json.dumps(winner, indent=2))
    log('completed', state='completed', test_score=score, finished_at=datetime.now().astimezone().isoformat())


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        log('failed', state='failed', error=repr(error))
        raise
