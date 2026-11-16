"""Encode a later flow window with the frozen pilot adapter; never train on queries."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.feature_extraction import FeatureHasher
from .common.data import Window,read_window


def main():
    ap=argparse.ArgumentParser()
    for key in ['checkpoint','flows','cache','output']:ap.add_argument('--'+key,type=Path,required=True)
    for key in ['day','start','end']:ap.add_argument('--'+key,type=int,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.start>=args.end or args.day<=12:raise ValueError('a later window after training day12 is required')
    saved=torch.load(args.checkpoint,weights_only=False);hosts=saved['hosts'];state=saved['state']
    obs=read_window(Window('query',args.day,args.start,args.end,str(args.flows)),hosts,args.cache)
    x=FeatureHasher(n_features=1024,input_type='dict',alternate_sign=False).transform([{k:np.log1p(v) for k,v in r.items()} for r in obs.counts]).toarray().astype('float32')
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    z=F.normalize(F.linear(torch.from_numpy(x),state['adapter.weight'],state['adapter.bias']),dim=1).numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(z,index=pd.Index(hosts,name='host_id')).add_prefix('emb_').to_csv(args.output)

if __name__=='__main__':main()
