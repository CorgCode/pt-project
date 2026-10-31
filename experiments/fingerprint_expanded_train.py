"""Обучение головы на32 исторических часах, без доступа к validation/test."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from experiments.fingerprint_bag import BagConfig, train_head
from experiments.fingerprint_expanded_data import WIDTH


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--dim', required=True, type=int, choices=[128, 256])
    args = ap.parse_args()
    work, dim = args.work, args.dim
    config = BagConfig(width=WIDTH, dim=dim, temperature=0.1 if dim == 128 else 0.07,
                       margin=0.1 if dim == 128 else 0.15, dropout=0.2)
    train_head(sp.load_npz(work / 'training.npz'), json.loads((work / 'hosts.json').read_text()),
               np.load(work / 'teacher.npy'), config, work / f'bag{dim}', checkpoints=(120, 300))


if __name__ == '__main__':
    main()
