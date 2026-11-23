"""Select a nonzero GNN weight on day13 and compare matched ablations."""
import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from experiments.temporal_graph_mixer.common.evaluate import geometry, with_identity


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frame(values, hosts):
    return pd.DataFrame(values, index=pd.Index(hosts, name='host_id')).add_prefix('emb_')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = run / 'development'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    training = run / 'training'
    proto = json.loads((training / 'protocol.json').read_text())
    if proto['train'] != ['day02', 'day10', 'day11', 'day12']:
        raise ValueError('unexpected training history')
    cohort = pd.read_csv('PT_PROJECT_ML/late_test/processed/cohort.csv', dtype=str).host_id.tolist()
    source_path = Path('runs/similarity_graph_recent/refine01/source.csv')
    late_path = run / 'baseline_day13.csv'
    source = pd.read_csv(source_path, dtype={'host_id': str}).set_index('host_id')
    late = pd.read_csv(late_path, dtype={'host_id': str}).set_index('host_id')
    if source.index.tolist() != cohort or late.index.tolist() != cohort:
        raise ValueError('baseline and graph windows have different host order')
    base, late_base = source.to_numpy(float), late.to_numpy(float)
    # This locally calibrated reference has the original CSVs and no absent AppleDouble entries.
    ref = Path('runs/similarity_graph_recent/select01/references/day13_development')
    dev_manifest = json.loads((run / 'windows/day13/manifest.json').read_text())
    if (dev_manifest['start'], dev_manifest['end']) != (1081800, 1085400):
        raise ValueError('unexpected development window')
    totals = np.load(run / 'windows/day13/totals.npy')
    observed = {cohort[i] for i in np.flatnonzero(totals[:, 4] + totals[:, 5] >= 300)}
    reference = set(pd.read_csv(ref / 'eval_hosts.csv', dtype=str).host_id)
    if observed != reference:
        raise ValueError('development active hosts differ from the reference')
    fit = Path('PT_PROJECT_ML/late_test/reference/validation')
    cache = run / 'eval_cache'
    rows = []

    def evaluate(name, mode=None, weight=0.):
        if mode is None:
            a, b = base, late_base
        else:
            a = np.hstack([base, weight * np.load(training / f'{mode}_source.npy')])
            b = np.hstack([late_base, weight * np.load(training / f'{mode}_day13.npy')])
        # Evaluate exactly the values reloaded by the project CSV reader.
        source_file, late_file = out / f'{name}_source.csv', out / f'{name}_day13.csv'
        frame(a, cohort).to_csv(source_file); frame(b, cohort).to_csv(late_file)
        a = pd.read_csv(source_file, dtype={'host_id': str}).set_index('host_id')
        b = pd.read_csv(late_file, dtype={'host_id': str}).set_index('host_id')
        core = geometry(a, ref, fit, cache)
        result = with_identity(a, b, ref, core)
        (out / f'{name}.json').write_text(json.dumps(result, indent=2))
        row = {'name': name, 'mode': mode, 'weight': weight, 'score': result['score'],
               'metrics': result['metrics'], 'source_sha256': sha(source_file), 'late_sha256': sha(late_file)}
        rows.append(row); print('development', name, row['score'], row['metrics'], flush=True)
        (out / 'results.json').write_text(json.dumps(rows, indent=2))
        return row

    baseline = evaluate('baseline')
    candidates = [evaluate(f'gnn_w{int(100*w):03d}', 'gnn', w) for w in (.25, .5, 1.)]
    best = max(candidates, key=lambda r: (r['score'], -r['weight']))
    controls = [evaluate(f'{mode}_w{int(100*best["weight"]):03d}', mode, best['weight'])
                for mode in ['no_graph', 'no_bytes']]
    result = {'selected': best, 'baseline': baseline, 'controls': controls,
              'chosen_on': 'previously studied development day13 only',
              'weights_tested': [.25, .5, 1.], 'gnn_weight_must_be_nonzero': True,
              'no_bytes_scope': 'byte inputs removed only in added neural block; fixed base may still use bytes',
              'test_not_used': True, 'seed_count': 1, 'checkpoint_sha256': sha(training / 'gnn.pt'),
              'base_source_sha256': sha(source_path), 'base_day13_sha256': sha(late_path)}
    (out / 'frozen_selection.json').write_text(json.dumps(result, indent=2))
    print('FROZEN', json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
