"""Evaluate baseline and a fixed pilot checkpoint on studied development day13."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.masked_multiday.common.evaluate import geometry, with_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    root = Path('PT_PROJECT_ML')
    ref = args.run / 'reference_day13'
    if not ref.exists():
        shutil.copytree(root / 'fingerprint_recent/reference_validation/validation', ref)
        manifest = json.loads((ref / 'manifest.json').read_text())
        manifest['neighbor_ceiling'] = manifest['score_neighbors_ideal']
        (ref / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    fit = root / 'late_test/reference/validation'  # day02 12:00–12:30
    hosts = pd.Index(json.loads((root / 'fingerprint_aligned/hosts.json').read_text()), dtype=str)
    original = pd.read_csv(root / 'fingerprint_codebook_recent/candidates/codebook1_e400/embeddings.csv',
                           dtype={'host_id': str}).set_index('host_id').loc[hosts]
    later = pd.read_csv(root / 'fingerprint_codebook_recent/candidates/codebook1_e400/later.csv',
                        dtype={'host_id': str}).set_index('host_id')
    candidate = pd.DataFrame(np.load(args.run / f'source_step{args.step:04}.npy').astype(np.float64),
                             index=hosts, columns=original.columns)
    for name, source in [('baseline', original), (f'step{args.step:04}', candidate)]:
        core = geometry(source, ref, fit, args.run / 'cache')
        result = with_identity(source, later, ref, core)
        (args.run / f'dev_{name}.json').write_text(json.dumps(result, indent=2))
        print(name, 'score', result['score'], 'metrics', result['metrics'], flush=True)


if __name__ == '__main__':
    main()
