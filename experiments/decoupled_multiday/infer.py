"""Apply a frozen Decoupled Multiday checkpoint, without fitting any component."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from experiments.similarity_graph.data import Window,read_window,signature
from .model import frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['checkpoint','source','flows','cache','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--start',type=float,required=True);parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--day',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    model=joblib.load(args.checkpoint)
    if args.start<model.training_end:raise ValueError('late window precedes training cutoff')
    source=pd.read_csv(args.source,index_col=0)
    if source.index.tolist()!=model.hosts:raise ValueError('source cohort/order differs')
    np.testing.assert_allclose(source.to_numpy(),model.source,rtol=1e-6,atol=1e-8)
    before=signature(args.source);model.set_cache(args.cache)
    obs=read_window(Window('inference',args.day,args.start,args.end,str(args.flows)),model.hosts,args.cache)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame(model.predict(obs),model.hosts).to_csv(args.output)
    if signature(args.source)!=before:raise RuntimeError('inference changed frozen source')
    print(args.output,model.source.shape)


if __name__=='__main__':main()
