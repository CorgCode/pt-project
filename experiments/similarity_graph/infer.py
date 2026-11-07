"""Apply a saved model to traffic with optional arbitrary target-node order; never calls fit."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import Window,read_window,signature
from .run import frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','source','flows','cache','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--start',type=float,required=True);parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--day',type=int,required=True);parser.add_argument('--target-cohort',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    model=joblib.load(args.checkpoint)
    if getattr(model.matcher,'joint_context',False):model.matcher.context_cache=str(args.cache)
    if args.start<model.training_end:raise ValueError('late window precedes training cutoff')
    before=signature(args.source)
    source=pd.read_csv(args.source,index_col=0)
    if source.index.astype(str).tolist()!=model.hosts:raise ValueError('source cohort/order mismatch')
    np.testing.assert_allclose(source.to_numpy(),model.source,rtol=1e-10,atol=1e-12)
    hosts=pd.read_csv(args.target_cohort,dtype=str).host_id.tolist() if args.target_cohort else model.hosts
    if len(hosts)!=len(set(hosts)):raise ValueError('duplicate target nodes')
    obs=read_window(Window('inference',args.day,args.start,args.end,str(args.flows)),hosts,args.cache)
    values,info=model.predict(obs,hosts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame(values,hosts).to_csv(args.output)
    args.output.with_suffix('.matching.json').write_text(json.dumps(info,indent=2)+'\n')
    if signature(args.source)!=before:raise RuntimeError('source changed during inference')
    print(args.output,info)


if __name__=='__main__':main()
