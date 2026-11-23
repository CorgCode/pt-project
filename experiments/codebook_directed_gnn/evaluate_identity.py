"""Применение сохранённой late-поправки и оценка development13 по CSV."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from .base import sha
from .model import CodebookDirectedGNN
from .train_identity import load_graph
from experiments.temporal_graph_mixer.common.evaluate import geometry,with_identity


def apply(checkpoint,graph_path,old,hosts=None):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=CodebookDirectedGNN();model.load_state_dict(state['state_dict']);model.eval()
    if hosts is None:
        graph=joblib.load(graph_path);hosts=graph['nodes'][:graph['n_cohort']]
    positions=pd.Index(hosts).get_indexer(old.index)
    if (positions<0).any():raise ValueError('Поздний хост вне исходной когорты')
    x,edges=load_graph(graph_path,hosts,
        directionless_bytes=state['protocol'].get('directionless_bytes',False))
    torch.set_num_threads(3)
    with torch.no_grad():
        g=model.graph(x,edges,n_output=len(hosts),use_graph=state['protocol']['use_graph'])
        values=model.late(torch.from_numpy(old.to_numpy()),g[positions]).numpy()
    return pd.DataFrame(values,index=old.index,columns=old.columns)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','windows','training','reference','output','zero']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cp=a.training/'model.pt';completed=json.loads((a.training/'completed.json').read_text())
    if sha(cp)!=completed['checkpoint_sha256'] or completed['dev_used'] or completed['test_used']:
        raise ValueError('Неверный checkpoint')
    source=pd.read_csv(a.zero/'source.csv',index_col=0)
    old=pd.read_csv(a.zero/'late.csv',index_col=0,float_precision='round_trip')
    if not source.index.equals(old.index):raise ValueError('Разный порядок хостов')
    late=apply(cp,a.windows/'day13.joblib',old);a.output.mkdir(parents=True)
    late.to_csv(a.output/'late.csv');late=pd.read_csv(a.output/'late.csv',index_col=0)
    result=with_identity(source,late,a.reference,
        geometry(source,a.reference,a.data/'late_test/reference/validation',a.output/'cache'))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps({'score':result['score'],'metrics':result['metrics'],
        'checkpoint_sha256':sha(cp),'source_unchanged':True,'test_used':False},indent=2)+'\n')
    print(result['score'])


if __name__=='__main__':main()
