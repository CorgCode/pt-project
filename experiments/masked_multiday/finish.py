"""Resume development evaluation of an already trained pilot without retraining."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.feature_extraction import FeatureHasher
from .common.data import read_window,Window
from .common.evaluate import geometry,with_identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(a.run/'model.pt',weights_only=False);state=saved['state'];hosts=saved['hosts']
    root=Path('PT_PROJECT_ML');day=root/'fingerprint_recent/day13'
    start,end=json.loads((day/'manifest.json').read_text())['config']['windows']['test']
    obs=read_window(Window('day13',13,start,end,str(day/'flows/test.csv')),hosts,a.run/'cache')
    x=FeatureHasher(n_features=1024,input_type='dict',alternate_sign=False).transform([{k:np.log1p(v) for k,v in row.items()} for row in obs.counts]).toarray().astype('float32')
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    late=F.normalize(F.linear(torch.from_numpy(x),state['adapter.weight'],state['adapter.bias']),dim=1).numpy()
    source=F.normalize(state['source'],dim=1).numpy()
    def frame(x):return pd.DataFrame(x,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')
    reference=root/'fingerprint_recent/reference_validation/validation';local=a.run/'reference_development'
    if not local.exists():shutil.copytree(reference,local)
    meta=json.loads((reference/'manifest.json').read_text())
    meta['neighbor_ceiling']=json.loads((root/'fingerprint_recent/candidates/ema3/result.json').read_text())['score_neighbors_ideal']
    (local/'manifest.json').write_text(json.dumps(meta,indent=2))
    core=geometry(frame(source),local,root/'late_test/reference/validation',a.run/'cache')
    result=with_identity(frame(source),frame(late),local,core)
    frame(late).to_csv(a.run/'validation.csv')
    (a.run/'development.json').write_text(json.dumps(result,indent=2))
    status={'development_score':result['score'],'target_reached':result['score']>=85,'test_read':False,'scope':'previously examined day13, local references','decision':'stop pilot before tests' if result['score']<75 else 'needs rolling validation'}
    (a.run/'STATUS.json').write_text(json.dumps(status,indent=2));print(status,flush=True)

if __name__=='__main__':main()
