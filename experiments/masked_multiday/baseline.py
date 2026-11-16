"""Reproduce the fixed seed17 pilot control on identical historical data."""
from pathlib import Path
import json,numpy as np,pandas as pd,torch
from sklearn.feature_extraction import FeatureHasher
from torch.nn import functional as F
from experiments.masked_multiday.common.data import Window,read_window
from experiments.masked_multiday.common.evaluate import geometry,with_identity
out=Path('runs/masked_multiday_recent_seed17');root=Path('PT_PROJECT_ML');torch.set_num_threads(2)
saved=torch.load(out/'model.pt',weights_only=False);hosts=saved['hosts'];pca=saved['pca']
hasher=FeatureHasher(n_features=1024,input_type='dict',alternate_sign=False)
def x(day,path,split):
 m=json.loads((path/'manifest.json').read_text());start,end=m['config']['windows'][split]
 o=read_window(Window(f'day{day}',day,start,end,str(path/f'flows/{split}.csv')),hosts,out/'cache')
 a=hasher.transform([{k:np.log1p(v) for k,v in r.items()} for r in o.counts]).toarray().astype('float32');return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
a=x(2,root/'multiday/processed','train');b=x(11,root/'fingerprint_memory/day11','test');c=x(12,root/'fingerprint_expanded/day12','test')
anchor=(a+b+c)/3
z=F.normalize(torch.from_numpy(pca.transform(anchor)),dim=1)
w=torch.linalg.lstsq(torch.from_numpy(anchor),z).solution
late=F.normalize(torch.from_numpy(x(13,root/'fingerprint_recent/day13','test'))@w,dim=1)
def frame(z):return pd.DataFrame(z.numpy(),index=pd.Index(hosts,name='host_id')).add_prefix('emb_')
core=geometry(frame(z),out/'reference_development',root/'late_test/reference/validation',out/'cache')
r=with_identity(frame(z),frame(late),out/'reference_development',core)
(out/'baseline_development.json').write_text(json.dumps(r,indent=2));print('baseline',r['score'],r['metrics'])
