"""Freeze a specific already-written step for a matched-budget pilot comparison."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from .evaluate_dev import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    destination=a.output/'checkpoint.pt'
    if destination.exists():raise FileExistsError(destination)
    shutil.copyfile(a.training/'checkpoint.pt',destination)
    state=torch.load(destination,map_location='cpu',weights_only=True)
    if state['step'] != a.step:
        raise ValueError(f"captured step {state['step']} instead of {a.step}")
    (a.output/'completed.json').write_text(json.dumps({'steps':a.step,'sha256':sha(destination),
        'test_used':False,'development_used':False,'scope':'intermediate step snapshot, no development selection'},indent=2))
    print('snapshot',a.step,sha(destination),flush=True)


if __name__=='__main__':main()
