"""Apply a trained rebuilt-plus delivery checkpoint without fitting or late-ID lookup."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import Window, read_window, signature
from .plus_experiment import frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','source','flows','cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--start',type=float,required=True)
    parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--day',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    model=joblib.load(args.checkpoint)
    if args.start<model.base.training_end: raise ValueError('late window precedes training cutoff')
    before=signature(args.source)
    source=pd.read_csv(args.source,index_col=0)
    if source.index.astype(str).tolist()!=model.hosts: raise ValueError('source host order differs')
    np.testing.assert_allclose(source.to_numpy(),model.source,rtol=1e-7,atol=1e-9)
    model.base.features.cache=str(args.cache)
    observation=read_window(Window('inference',args.day,args.start,args.end,str(args.flows)),model.hosts,args.cache)
    values=model.predict(observation)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame(values,model.hosts).to_csv(args.output)
    if signature(args.source)!=before: raise RuntimeError('source changed during inference')
    print(args.output)


if __name__=='__main__':
    main()
