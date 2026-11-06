"""Train a standalone plus reconstruction; no development/test traffic is read here."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from .codebook import RebuiltPlus
from .data import Window, read_window, check_schedule, signature
from .protocol import record
from .run import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(args.config)
    train=sorted([Window(**w) for w in config['train']],key=lambda w:w.start)
    dev=[Window(**w) for w in config['development']]
    tests=[Window(**w) for w in config.get('test',[])]
    check_schedule(train,dev,tests)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    record(args.output,config)
    torch.set_num_threads(2)
    hosts=pd.read_csv(config['cohort'],dtype=str)['host_id'].tolist()
    if len(hosts)!=len(set(hosts)): raise ValueError('duplicate cohort IDs')
    observations=[]
    for w in train:
        print('training inputs',w.name,flush=True)
        observations.append(read_window(w,hosts,config['cache']))
    source_window=Window(**config.get('source_window',asdict(train[0])))
    if not any(w.start<=source_window.start and w.end>=source_window.end for w in train):
        raise ValueError('source_window must lie within declared training history')
    source_obs=read_window(source_window,hosts,config['cache'])
    options=config.get('plus_training',{})
    model=RebuiltPlus().fit(observations,hosts,config['cache'],source_observation=source_obs,**options)
    joblib.dump(model,args.output/'plus.joblib',compress=3)
    pd.DataFrame(model.source,index=pd.Index(hosts,name='host_id')).add_prefix('emb_').to_csv(args.output/'source.csv')
    freeze={'source_sha256':signature(args.output/'source.csv'),
            'checkpoint_sha256':signature(args.output/'plus.joblib'),
            'training_end':model.training_end,'train':[asdict(w) for w in train],
            'source_window':asdict(source_window),
            'reconstruction_not_original_checkpoint':True,'development_or_test_read':False,
            'recipe_source':'submission card 70347a3 + historical codebook/listwise 86d709d',
            'base_dim':model.codes.shape[1],'dim':model.source.shape[1],
            'source_provenance':'all feature fits, source codes, partner targets and graph use train only'}
    (args.output/'frozen_training.json').write_text(json.dumps(freeze,indent=2)+'\n')
    (args.output/'training_log.json').write_text(json.dumps(model.training_log,indent=2)+'\n')
    print('trained',args.output,freeze,flush=True)


if __name__=='__main__':
    main()
