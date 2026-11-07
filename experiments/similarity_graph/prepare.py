"""Train S from declared historical flows only; source is always the explicit day02 hour."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from .data import Window,read_window,contacts,check_schedule,signature
from .features import behavior
from .metrics import contact_matrix,WindowTask
from .matrix import initialize,optimize
from .reference import build


def config_file(path):
    def expand(v):
        if isinstance(v,str):return os.path.expandvars(v)
        if isinstance(v,list):return [expand(x) for x in v]
        if isinstance(v,dict):return {k:expand(x) for k,x in v.items()}
        return v
    return expand(json.loads(Path(path).read_text()))


def environment():
    result={}
    for package in ['numpy','pandas','scipy','scikit-learn','joblib','torch']:
        try:result[package]=version(package)
        except PackageNotFoundError:result[package]=None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();config=config_file(args.config)
    code_at_start={p.name:signature(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}
    if args.output.exists():raise FileExistsError(args.output)
    windows=sorted([Window(**w) for w in config['train']],key=lambda w:w.start)
    dev=[Window(**w) for w in config['development']];tests=[Window(**w) for w in config.get('test',[])]
    check_schedule(windows,dev,tests)
    args.output.mkdir(parents=True)
    hosts=pd.read_csv(config['cohort'],dtype=str).host_id.tolist()
    source=read_window(Window(**config['source_window']),hosts,config['cache'])
    if not any(w.start<=source.window.start and w.end>=source.window.end for w in windows):
        raise ValueError('canonical source must lie within training history')
    observations=[]
    for w in windows:
        print('training observation',w.name,flush=True)
        observations.append(read_window(w,hosts,config['cache']))
    source_rows=contacts(source)
    tasks=[]
    for observation in observations:
        if observation.window.day<=source.window.day:continue
        ref=args.output/'training_references'/observation.window.name
        build(source,observation,hosts,ref,config['cache'])
        tasks.append(WindowTask(source_rows,observation,hosts,ref))
        print('training references ready',observation.window.name,flush=True)
    fitted=behavior(source.window,hosts,config['cache'])[1]
    values=np.zeros_like(behavior(source.window,hosts,config['cache'],fitted)[0]);total=np.zeros(len(hosts))
    cutoff=max(w.day for w in windows)
    peer_rows=[{} for _ in hosts]
    from collections import Counter
    peer_rows=[Counter() for _ in hosts]
    for obs in observations:
        weight=2**(-(cutoff-obs.window.day)/3.)
        active=obs.events>=300
        values+=behavior(obs.window,hosts,config['cache'],fitted)[0]*active[:,None]*weight
        total+=active*weight
        for row,new in zip(peer_rows,contacts(obs)):
            for p,count in new.items():row[p]+=weight*np.log1p(count*3600/(obs.window.end-obs.window.start))
    values/=np.maximum(total[:,None],1e-12)
    vocabulary=sorted(set().union(*(set(r) for r in peer_rows)))
    structural=TfidfTransformer().fit_transform(contact_matrix(peer_rows,vocabulary))
    initial=initialize(values,structural)
    np.save(args.output/'initial.npy',initial)
    learned,history=optimize(initial,tasks,**config.get('matrix_training',{}))
    np.save(args.output/'learned.npy',learned)
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    freeze={'config':config,'hosts':hosts,'training_end':max(w.end for w in windows),
            'versions':environment(),
            'train':[asdict(w) for w in windows],'test_or_development_read':False,
            'source_window':asdict(source.window),
            'matrix_sha256':{n:signature(args.output/f'{n}.npy') for n in ['initial','learned']},
            'training_files':{w.name:{'bytes':Path(w.flows).stat().st_size,'mtime_ns':Path(w.flows).stat().st_mtime_ns} for w in windows},
            'code_sha256':code_at_start}
    (args.output/'frozen_training.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print('matrix training complete',args.output,flush=True)


if __name__=='__main__':main()
