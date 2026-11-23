"""Evaluate a completed checkpoint on day13, using fixed prefix source embeddings."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph_utils import load_window, tensor_window
from .model import TemporalGATv2
from experiments.temporal_graph_mixer.common.evaluate import geometry, with_identity


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def frame(values, hosts):
    return pd.DataFrame(values,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    completion = json.loads((args.training/'completed.json').read_text())
    ckpt = args.training/'checkpoint.pt'
    if sha(ckpt)!=completion['sha256'] or completion['development_used'] or completion['test_used']:
        raise ValueError('training checkpoint provenance mismatch')
    state = torch.load(ckpt,map_location='cpu',weights_only=True)
    config = state['config']
    if config['train_windows']!=[2,10,11,12] or state['step']!=completion['steps']:
        raise ValueError('unexpected prefix or incomplete model')
    hosts_file = args.data/'late_test/processed/cohort.csv'
    hosts = pd.read_csv(hosts_file,dtype=str).host_id.tolist()
    torch.set_num_threads(2)
    model = TemporalGATv2(width=config['width'],output_dim=512,dropout=.1)
    model.load_state_dict(state['model']);model.eval()
    args.output.mkdir(parents=True)
    encoded,presence = {},{}
    with torch.no_grad():
        for day in [2,10,11,12,13]:
            graph = load_window(args.windows/f'day{day:02}.joblib')
            if graph['nodes'][:len(hosts)] != hosts or graph['manifest']['hosts_sha256'] != sha(hosts_file):
                raise ValueError(f'day{day}: host order mismatch')
            if day==13:
                if (graph['manifest']['start'],graph['manifest']['end']) != (1081800,1085400):
                    raise ValueError('incorrect development window')
                observed={hosts[i] for i in np.flatnonzero(graph['totals'][:len(hosts),:,4:6].sum((1,2))>=300)}
                reference=set(pd.read_csv(args.reference/'eval_hosts.csv',dtype=str).host_id)
                if observed != reference:raise ValueError('development active cohort differs')
            x,g,c=tensor_window(graph)
            if config.get('disable_bytes',False):
                x[:,:,:2]=0
                for edge in g:
                    edge['out_attr'][:,1:3]=0;edge['in_attr'][:,1:3]=0
            encoded[day],presence[day]=model.encode_window(x,g,c,n_output=len(hosts),use_graph=not config.get('disable_graph',False))
            print('encoded',day,flush=True)
            del graph,x,g,c
        h=torch.stack([encoded[d] for d in [2,10,11,12]],dim=1)
        mask=torch.stack([presence[d] for d in [2,10,11,12]],dim=1)
        offset=torch.tensor([-10.,-2.,-1.,0.]).expand(len(hosts),-1)
        source=model.encode_history(h,mask,offset).numpy()
        late=model.encode_history(encoded[13][:,None,:],presence[13][:,None],torch.zeros(len(hosts),1)).numpy()
    source_file, late_file=args.output/'embeddings.csv',args.output/'validation.csv'
    frame(source,hosts).to_csv(source_file);frame(late,hosts).to_csv(late_file)
    # Match the project parser, including its treatment of decimal CSVs.
    src=pd.read_csv(source_file,dtype={'host_id':str}).set_index('host_id')
    dst=pd.read_csv(late_file,dtype={'host_id':str}).set_index('host_id')
    core=geometry(src,args.reference,args.data/'late_test/reference/validation',args.output/'cache')
    result=with_identity(src,dst,args.reference,core)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    summary={'development_score':result['score'],'metrics':result['metrics'],'checkpoint_sha256':sha(ckpt),
        'source_sha256':sha(source_file),'late_sha256':sha(late_file),'training_steps':state['step'],
        'source_history':[2,10,11,12], 'source_is_fixed_across_test':True,
        'scope':'day13 development, already studied', 'test_used':False,
        'model_inference_uses_own_host_id':False}
    (args.output/'frozen_selection.json').write_text(json.dumps(summary,indent=2))
    print('development',json.dumps(summary),flush=True)


if __name__=='__main__':main()
