"""Evaluate a completed frozen stage-one checkpoint on the development window."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from .model import CodebookDirectedGNN
from .train_identity import load_graph,sha
from experiments.masked_multiday.common.evaluate import geometry,with_identity


def main():
    p=argparse.ArgumentParser()
    for n in ('data','windows','training','reference','output','zero'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    complete=json.loads((a.training/'completed.json').read_text())
    cp=a.training/'model.pt'
    if sha(cp)!=complete['checkpoint_sha256'] or complete['dev_used'] or complete['test_used']:
        raise ValueError('invalid provenance')
    state=torch.load(cp,map_location='cpu',weights_only=True)
    model=CodebookDirectedGNN();model.load_state_dict(state['state_dict']);model.eval()
    hosts=pd.read_csv(a.data/'late_test/processed/cohort.csv',dtype=str).host_id.tolist()
    host_index=pd.Index(hosts)
    x,edges=load_graph(a.windows/'day13.joblib',hosts,
        directionless_bytes=state['protocol'].get('directionless_bytes',False))
    source=pd.read_csv(a.zero/'source.csv',dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    old=pd.read_csv(a.zero/'late.csv',dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    pos=host_index.get_indexer(old.index)
    if (pos<0).any():raise ValueError('unknown late host')
    torch.set_num_threads(3)
    with torch.no_grad():
        g=model.graph(x,edges,n_output=len(hosts),use_graph=state['protocol']['use_graph'])
        late=model.late(torch.from_numpy(old.to_numpy()),g[torch.from_numpy(pos)]).numpy()
    a.output.mkdir(parents=True,exist_ok=False)
    dest=a.output/'late.csv'
    pd.DataFrame(late,index=old.index,columns=old.columns).to_csv(dest)
    reloaded=pd.read_csv(dest,dtype={'host_id':str},float_precision='round_trip').set_index('host_id')
    core=geometry(source,a.reference,a.data/'late_test/reference/validation',a.zero/'cache')
    result=with_identity(source,reloaded,a.reference,core)
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    summary={'score':result['score'],'metrics':result['metrics'],'checkpoint_sha256':sha(cp),
             'output_sha256':sha(dest),'steps':complete['steps'],'test_used':False,
             'source_unchanged':True,'day13_is_development':True}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
