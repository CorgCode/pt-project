"""Prepare input graphs without reading later windows before selection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--part', choices=['train', 'development', 'benchmark'], required=True)
    p.add_argument('--cap', type=int, default=16); p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--frozen-selection', type=Path,
                   help='required for benchmark; development decision must already be frozen')
    a = p.parse_args()
    if a.part == 'benchmark':
        if not a.frozen_selection or not a.frozen_selection.exists():
            raise ValueError('benchmark requires a frozen development selection')
        decision = json.loads(a.frozen_selection.read_text())
        if decision.get('test_used') is not False:
            raise ValueError('selection does not declare test isolation')
    root = a.data
    hosts = root / 'late_test/processed/cohort.csv'
    configs = {'train': [
        ('day02', root/'late_test/processed/flows/train.csv', 126000,129600),
        ('day10', root/'fingerprint_memory/day10/flows/test.csv',822600,826200),
        ('day11', root/'fingerprint_memory/day11/flows/test.csv',909000,912600),
        ('day12', root/'fingerprint_expanded/day12/flows/test.csv',995400,999000)],
        'development': [('day13', root/'fingerprint_recent/day13/flows/test.csv',1081800,1085400)],
        'benchmark': [('day14', root/'late_test/processed/flows/test.csv',1168200,1171800),
                      ('day15', root/'late_test/processed/flows/test_day15.csv',1254600,1258200),
                      ('day16', root/'late_test/processed/flows/test_day16.csv',1341000,1344600)]}
    a.output.mkdir(parents=True, exist_ok=True)
    for name, flows, start, end in configs[a.part]:
        out = a.output / f'{name}.joblib'
        if out.exists() and a.skip_existing:
            # Do not silently accept a different data source, interval, or cap.
            import joblib
            from .data import sha
            m = joblib.load(out)['manifest']
            if (m['start'], m['end'], m['neighbor_cap_per_part_and_direction']) != (start, end, a.cap) or m['flows_sha256'] != sha(flows) or m['hosts_sha256'] != sha(hosts):
                raise ValueError(f'cached window {name} differs')
            continue
        print('preparing', name, flush=True)
        subprocess.run([sys.executable, '-m', 'experiments.temporal_gatv2.data', '--flows', str(flows),
                        '--hosts', str(hosts), '--start', str(start), '--end', str(end),
                        '--output', str(out), '--cap', str(a.cap)], check=True)


if __name__ == '__main__':
    main()
