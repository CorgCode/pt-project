"""Prepare only source and historical train or the separate development window."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--part', choices=['train', 'dev'], required=True)
    p.add_argument('--cap', type=int, default=16)
    a = p.parse_args()
    d = a.data
    entries = {'train': [('day02', d/'late_test/processed/flows/train.csv', 126000),
                         ('day10', d/'fingerprint_memory/day10/flows/test.csv', 822600),
                         ('day11', d/'fingerprint_memory/day11/flows/test.csv', 909000),
                         ('day12', d/'fingerprint_expanded/day12/flows/test.csv', 995400)],
               'dev': [('day13', d/'fingerprint_recent/day13/flows/test.csv', 1081800)]}
    a.output.mkdir(parents=True, exist_ok=True)
    for name, path, start in entries[a.part]:
        subprocess.run([sys.executable, '-m', 'experiments.codebook_directed_gnn.data',
                        '--flows', str(path), '--hosts', str(d/'late_test/processed/cohort.csv'),
                        '--start', str(start), '--end', str(start+3600), '--cap', str(a.cap),
                        '--output', str(a.output/(name+'.joblib'))], check=True)


if __name__ == '__main__':
    main()
