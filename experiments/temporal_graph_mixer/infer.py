"""Inference замороженной модели для произвольного списка хостов."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .data import extract,sha
from .evaluate import checkpoint,frame
from .model import encode,plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['trained','selection','flows','cohort','output']:p.add_argument('--'+key,type=Path,required=True)
    for key in ['start','end']:p.add_argument('--'+key,type=float,required=True)
    p.add_argument('--day',type=int,required=True)
    a=p.parse_args();f=json.loads((a.trained/'frozen_training.json').read_text());selected=json.loads(a.selection.read_text())
    if selected['training_sha256']!=sha(a.trained/'frozen_training.json'):raise ValueError('model changed since selection')
    if a.start<f['training_end']:raise ValueError('query precedes training cutoff')
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.trained/'transforms.joblib')!=f['transforms_sha256']:raise ValueError('transforms changed')
    transforms=joblib.load(a.trained/'transforms.joblib');hosts=pd.read_csv(a.cohort,dtype=str).host_id.tolist()
    if len(hosts)!=len(set(hosts)):raise ValueError('duplicate query nodes')
    obs=extract(dict(name='inference',day=a.day,start=a.start,end=a.end,flows=str(a.flows)),hosts)
    x=transforms.transform(obs);label=selected['selected']['label']
    if label!='plain' and sha(a.trained/f'{label}.pt')!=f['candidates'][label]['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
    values=plain(x) if label=='plain' else encode(checkpoint(a.trained/f'{label}.pt'),x,f['candidates'][label].get('weights',(2.,1.,.5)))
    a.output.parent.mkdir(parents=True,exist_ok=True);frame(values,hosts).to_csv(a.output,float_format='%.17g')
    print(a.output,'hosts',len(hosts),'dim',values.shape[1])


if __name__=='__main__':main()
