"""Prepare chronological LANL windows; benchmark is a separate explicit stage."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['development', 'benchmark'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('PT_PROJECT_ML')
    hosts = root / 'late_test/processed/cohort.csv'
    windows = {
        'development': [
            ('day02', root / 'late_test/processed/flows/train.csv', 126000, 129600),
            ('day10', root / 'fingerprint_memory/day10/flows/test.csv', 822600, 826200),
            ('day11', root / 'fingerprint_memory/day11/flows/test.csv', 909000, 912600),
            ('day12', root / 'fingerprint_expanded/day12/flows/test.csv', 995400, 999000),
            ('day13', root / 'fingerprint_recent/day13/flows/test.csv', 1081800, 1085400),
        ],
        'benchmark': [
            ('day14', root / 'late_test/processed/flows/test.csv', 1168200, 1171800),
            ('day15', root / 'late_test/processed/flows/test_day15.csv', 1254600, 1258200),
            ('day16', root / 'late_test/processed/flows/test_day16.csv', 1341000, 1344600),
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, path, start, end in windows[args.part]:
        dest = args.output / name
        print('preparing', name, flush=True)
        subprocess.run([sys.executable, '-m', 'experiments.directed_bytes_gnn.data', '--flows', str(path),
                        '--hosts', str(hosts), '--out', str(dest), '--start', str(start), '--end', str(end)], check=True)


if __name__ == '__main__':
    main()
