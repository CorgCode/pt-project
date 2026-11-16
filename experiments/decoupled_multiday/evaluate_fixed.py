"""History ablation at an ALREADY frozen recipe; no weight/retrieval search.

Prefix bases and all learned blocks must be fitted from that prefix independently.
This command evaluates development only; it cannot consume a closed test window.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import joblib
import numpy as np
from experiments.similarity_graph.prepare import config_file
from experiments.similarity_graph.data import Window,read_window,contacts,signature
from experiments.similarity_graph.reference import build,resolve
from experiments.similarity_graph.evaluate import geometry,with_identity
from .model import DecoupledModel,frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','trained','selection','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();config=config_file(args.config)
    freeze=json.loads((args.trained/'frozen_training.json').read_text())
    selection=json.loads(args.selection.read_text());recipe=selection['recipe'];hosts=freeze['hosts']
    train=sorted([Window(**w) for w in config['train']],key=lambda w:w.start)
    if freeze['train']!=[asdict(w) for w in train]:raise ValueError('prefix training schedule differs')
    if freeze['training_end']>min(w['start'] for w in config['development']):raise ValueError('prefix base leaks development')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output/'frozen_recipe.json').write_text(json.dumps({'recipe':recipe,
        'primary_selection_sha256':signature(args.selection),'prefix_training_end':freeze['training_end'],
        'prefix_seed_inputs':freeze['seeds'],'no_development_read_before_freeze':True},indent=2)+'\n')
    read=lambda w:read_window(Window(**w),hosts,config['cache'])
    source_obs=read(config['source_window']);report={'recipe':recipe,'results':{},'is_independent_holdout':False}
    for w in config['development']:
        obs=read(w);output=args.output/'references'/obs.window.name
        if obs.window.name in config.get('references',{}):
            ref=resolve(config['references'][obs.window.name],source_obs,obs,hosts,output,config)
        else:build(source_obs,obs,hosts,output,config['cache']);ref=output
        report['results'][obs.window.name]={}
        for stamp in freeze['seeds']:
            seed=stamp['seed'];path=args.trained/f'seed{seed}'/'blocks.joblib'
            if signature(path)!=stamp['checkpoint_sha256']:raise ValueError('prefix weights changed')
            p=joblib.load(path)
            model=DecoupledModel(p['base'],p['fingerprint'],p[recipe['partner_kind']],p['graph'],
                                recipe['weights'],recipe['k'],recipe['temperature'])
            source=frame(model.source,hosts);late=frame(model.predict(obs),hosts)
            core=geometry(source,ref,config['probe_reference'],config['cache'])
            result=with_identity(source,late,ref,core)
            report['results'][obs.window.name][str(seed)]=result
            print('fixed history',seed,obs.window.name,result['score'],flush=True)
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('fixed history evaluation complete',args.output,flush=True)


if __name__=='__main__':main()
