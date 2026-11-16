"""Train independent blocks on allowed historical windows; never read development/test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from experiments.similarity_graph.prepare import config_file,environment
from experiments.similarity_graph.data import Window,read_window,signature,check_schedule,contacts
from experiments.similarity_graph.features import with_context
from .base import RebuiltPlus,graph_block
from .collaborative import ImplicitALS,interaction_matrix,PartnerTask,refine_factors,direct_validation,unit
from .fingerprint import ContrastiveFingerprint
from .model import FrozenCodebook,frame


def load_base(config,train,hosts):
    options=config['base']
    if 'checkpoint' in options:
        path=Path(options['checkpoint'])
        if signature(path)!=options['sha256']:raise ValueError('base checkpoint fingerprint changed')
        base=joblib.load(path)
        if not isinstance(base,FrozenCodebook):raise ValueError('checkpoint must implement FrozenCodebook contract')
    else:
        path=Path(options['source'])
        if signature(path)!=options['sha256']:raise ValueError('base CSV fingerprint changed')
        data=pd.read_csv(path,index_col=0)
        if set(data.index)!=set(hosts):raise ValueError('base/cohort differ')
        base=FrozenCodebook(hosts,data.loc[hosts].to_numpy(),options['training_end'],
                            provenance=options.get('provenance','external frozen 896'))
    if base.hosts!=hosts:raise ValueError('base/cohort order differs')
    if base.training_end>max(w.end for w in train):raise ValueError('base used a later training cutoff')
    return base


def train_blocks(config,base,observations,source,seed,output):
    output.mkdir(parents=True)
    fingerprint=ContrastiveFingerprint(seed=seed,**config.get('identity',{})).fit(observations)
    context=[with_context(o,base.hosts,config['cache']) for o in observations if o.window.day>source.window.day]
    source_rows=contacts(source);cf=config.get('collaborative',{})
    through=cf.get('train_through_day',max(o.window.day for o in context)-1)
    before=[o for o in context if o.window.day<=through]
    validation=[o for o in context if o.window.day in cf.get('validation_days',[]) or o.window.day>through]
    scores=[]
    if 'fixed_regularization' in cf:
        selected=float(cf['fixed_regularization'])
        scores=[{'regularization':selected,'selection':'predeclared fixed control, no historical validation'}]
    else:
        if not before or not validation:raise ValueError('CF selection requires earlier fit days and separate historical validation days')
        matrix,vocabulary=interaction_matrix(before)
        for regularization in cf.get('regularizations',[10.,100.]):
            model=ImplicitALS(dim=cf.get('dim',32),regularization=regularization,steps=cf.get('als_steps',8),seed=seed).fit(matrix)
            reports=[direct_validation(model,vocabulary,o,source_rows,base.hosts) for o in validation]
            means=[r['mrr'] for r in reports if r['mrr'] is not None]
            if not means:raise ValueError('CF historical validation has no eligible queries')
            scores.append({'regularization':regularization,'validation_mrr':float(np.mean(means)),'reports':reports})
            print('CF selection',seed,scores[-1],flush=True)
        selected=max(scores,key=lambda r:r['validation_mrr'])['regularization']
    # Hyperparameters are now fixed. Final refit may use ALL allowed historical days.
    matrix,vocabulary=interaction_matrix(context)
    final=ImplicitALS(dim=cf.get('dim',32),regularization=selected,steps=cf.get('als_steps',8),seed=seed).fit(matrix)
    initial=unit(final.users)
    graph=graph_block(observations,base.hosts,dim=config.get('graph_dim',64))
    tasks=[PartnerTask(source_rows,o,base.hosts) for o in observations if o.window.day>source.window.day]
    partner,log=refine_factors(initial,base.codes,fingerprint.bank,graph,tasks,
                            steps=cf.get('listwise_steps',200),seed=seed)
    payload={'base':base,'fingerprint':fingerprint,'partner':partner,'cf_initial':initial,'graph':graph,
             'seed':seed,'cf_selection':scores,'cf_regularization':selected,'partner_log':log,
             'history_days':sorted({o.window.day for o in observations}),
             'training_end':max(o.window.end for o in observations)}
    joblib.dump(payload,output/'blocks.joblib',compress=3)
    (output/'training.json').write_text(json.dumps({k:v for k,v in payload.items() if k in
        ['seed','cf_selection','cf_regularization','partner_log','history_days','training_end']},indent=2)+'\n')
    return {'seed':seed,'checkpoint_sha256':signature(output/'blocks.joblib')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-base',action='store_true')
    args=parser.parse_args();config=config_file(args.config);torch.set_num_threads(2)
    windows=sorted([Window(**w) for w in config['train']],key=lambda w:w.start)
    dev=[Window(**w) for w in config['development']];test=[Window(**w) for w in config.get('test',[])]
    check_schedule(windows,dev,test)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    hosts=pd.read_csv(config['cohort'],dtype=str).host_id.tolist()
    observations=[read_window(w,hosts,config['cache']) for w in windows]
    sw=Window(**config['source_window'])
    if not any(w.start<=sw.start and sw.end<=w.end for w in windows):raise ValueError('source must be in training history')
    source=read_window(sw,hosts,config['cache'])
    if args.train_base:
        model=RebuiltPlus().fit(observations,hosts,config['cache'],source_observation=source,**config.get('base_training',{}))
        base=FrozenCodebook(hosts,model.codes,model.training_end,model,'local reconstruction, not original codebook weights')
    else:base=load_base(config,windows,hosts)
    joblib.dump(base,args.output/'base.joblib',compress=3);frame(base.codes,hosts).to_csv(args.output/'base.csv')
    freeze={'config':config,'versions':environment(),'train':[asdict(w) for w in windows],
            'base_sha256':signature(args.output/'base.csv'),'base_checkpoint_sha256':signature(args.output/'base.joblib'),
            'base_provenance':base.provenance,'hosts':hosts,'training_end':max(w.end for w in windows),
            'development_test_read':False,'seeds':[],
            'training_files':{w.name:{'bytes':Path(w.flows).stat().st_size,'mtime_ns':Path(w.flows).stat().st_mtime_ns} for w in windows},
            'code_sha256':{p.name:signature(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}}
    for seed in config.get('seeds',[17,29,43,71,101]):
        print('training seed',seed,flush=True)
        freeze['seeds'].append(train_blocks(config,base,observations,source,seed,args.output/f'seed{seed}'))
        (args.output/'frozen_training.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print('all blocks frozen',args.output,flush=True)


if __name__=='__main__':main()
