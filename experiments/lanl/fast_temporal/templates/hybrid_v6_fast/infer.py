import os
"""Применить выбранный быстрый адаптер к готовым векторам v5 (825 координат)."""
import argparse
from pathlib import Path
import sys
import json
ROOT = Path(os.environ["PT_REPO_ROOT"])
WORK = Path(os.environ["PT_RUN_ROOT"])
sys.path.insert(0,str(ROOT))
import pandas as pd
from evaluation.io import load_embeddings
from model import transform

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
folder=Path(__file__).resolve().parent
selected=json.loads((folder/'selected.json').read_text())
x=load_embeddings(a.input)
if x.shape[1]!=825:
    raise ValueError('Expected 825 v5 coordinates, ordered behavioral/peers/services/diffusion/neural')
y=transform(x.to_numpy(),folder,selected)
a.output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(y,index=x.index).add_prefix('emb_').to_csv(a.output,float_format='%.8g')
print(f'{len(x)} hosts, {x.shape[1]} -> {y.shape[1]} dimensions: {a.output}')
