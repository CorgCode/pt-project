"""Apply frozen day13-selected graph model to previously studied benchmark windows."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate_dev import frame, sha
from .model import infer, load_observation, load_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    selection = json.loads((run / 'development/frozen_selection.json').read_text())
    if not selection['test_not_used'] or selection['selected']['mode'] != 'gnn' or selection['selected']['weight'] <= 0:
        raise ValueError('nonzero GNN must have been frozen before test')
    checkpoint = run / 'training/gnn.pt'
    if sha(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('checkpoint changed after selection')
    source = run / f"development/{selection['selected']['name']}_source.csv"
    if sha(source) != selection['selected']['source_sha256']:
        raise ValueError('source changed after selection')
    out = run / 'benchmark_submission'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copyfile(source, out / 'embeddings.csv')
    hosts_file = Path('PT_PROJECT_ML/late_test/processed/cohort.csv')
    hosts = pd.read_csv(hosts_file, dtype=str).host_id.tolist()
    state = load_state(checkpoint)
    windows = [('day14', 'test', 'test14.csv', 1168200, 1171800),
               ('day15', 'test_day15', 'test15.csv', 1254600, 1258200),
               ('day16', 'test_day16', 'test16.csv', 1341000, 1344600)]
    baseline_sha = {
        'test': '566269aab2d2db4a3411659f0cfac98f2ebefa10ff83137abc715bb629143311',
        'test_day15': '5a17204b83345a44103c05352444e7f4a0df563c506f760ed9be69f017ac7a25',
        'test_day16': '4f0488b6399cb920ac7b1f720ecc266535f67da806a694153a6b37ae13130e03',
    }
    baseline_dir = Path('runs/similarity_graph_recent')
    results = {}
    for day, key, old_file, start, end in windows:
        manifest = json.loads((run / 'windows' / day / 'manifest.json').read_text())
        if (manifest['start'], manifest['end']) != (start, end) or manifest['hosts_sha256'] != sha(hosts_file):
            raise ValueError('mismatched benchmark window or host order')
        obs = load_observation(run / 'windows', day)
        actual_active = {hosts[i] for i in np.flatnonzero(obs['active'] >= 300)}
        active_ref = set(pd.read_csv(Path('PT_PROJECT_ML/late_test/reference') / key / 'eval_hosts.csv', dtype=str).host_id)
        if actual_active != active_ref:
            raise ValueError(f'{day}: active cohort does not match official reference')
        if sha(baseline_dir / old_file) != baseline_sha[key]:
            raise ValueError(f'{day}: baseline vector changed')
        baseline = pd.read_csv(baseline_dir / old_file, dtype={'host_id': str}).set_index('host_id')
        if baseline.index.tolist() != hosts:
            raise ValueError('baseline late host order mismatch')
        z = infer(state, obs)
        values = np.hstack([baseline.to_numpy(float), selection['selected']['weight'] * z])
        file = out / f'{key}.csv'
        frame(values, hosts).to_csv(file)
        results[key] = {'n_active': len(actual_active), 'base_sha256': sha(baseline_dir / old_file),
                        'late_sha256': sha(file), 'flow_sha256': manifest['flows_sha256']}
        print('inference', key, results[key], flush=True)
    (out / 'provenance.json').write_text(json.dumps({'source_sha256': sha(out / 'embeddings.csv'),
        'checkpoint_sha256': sha(checkpoint), 'chosen_weight': selection['selected']['weight'],
        'test_not_used_for_selection': True, 'previously_studied_benchmark': True, 'windows': results}, indent=2))
    print('benchmark vectors ready', out, flush=True)


if __name__ == '__main__':
    main()
