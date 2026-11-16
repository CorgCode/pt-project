"""Select listwise budget on earlier/later historical days, then refit allowed history.

This is conditional tuning above a FROZEN base/fingerprint. It is not an honest
rolling retraining of those components: use rolling.py with prefix bases for that.
Development and final test traffic are never opened by this command.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from evaluation.partners import partner_ranking
from experiments.similarity_graph.prepare import config_file
from experiments.similarity_graph.data import Window,read_window,contacts,signature
from experiments.similarity_graph.features import with_context
from experiments.similarity_graph.metrics import partner_frame
from .collaborative import interaction_matrix,ImplicitALS,PartnerTask,refine_factors,unit
from .model import assemble,frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','trained','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();config=config_file(args.config);torch.set_num_threads(2)
    freeze=json.loads((args.trained/'frozen_training.json').read_text());hosts=freeze['hosts']
    declared=sorted([Window(**w) for w in config['train']],key=lambda w:w.start)
    if freeze['train']!=[asdict(w) for w in declared]:raise ValueError('historical schedule changed')
    for w in declared:
        stamp=freeze['training_files'][w.name];stat=Path(w.flows).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(stamp['bytes'],stamp['mtime_ns']):raise ValueError('historical flow file changed')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for name,key in [('base.joblib','base_checkpoint_sha256'),('base.csv','base_sha256')]:
        if signature(args.trained/name)!=freeze[key]:raise ValueError('base artifact changed')
        import shutil
        shutil.copyfile(args.trained/name,args.output/name)
    observations=[read_window(Window(**w),hosts,config['cache']) for w in config['train']]
    source=read_window(Window(**config['source_window']),hosts,config['cache']);old=contacts(source)
    settings=config.get('collaborative',{});through=settings['train_through_day']
    before=[o for o in observations if source.window.day<o.window.day<=through]
    validation=[o for o in observations if o.window.day>through]
    if not before or not validation:raise ValueError('need distinct head-fit and historical validation days')
    joint=[with_context(o,hosts,config['cache']) for o in before]
    matrix,vocabulary=interaction_matrix(joint)
    all_tasks=[PartnerTask(old,o,hosts) for o in observations if o.window.day>source.window.day]
    fit_tasks=[PartnerTask(old,o,hosts) for o in before]
    budgets=settings.get('listwise_budgets',[0,100,200]);new_seeds=[]
    for stamp in freeze['seeds']:
        seed=stamp['seed'];path=args.trained/f'seed{seed}'/'blocks.joblib'
        if signature(path)!=stamp['checkpoint_sha256']:raise ValueError('block artifact changed')
        p=joblib.load(path);base=p['base'];fp=p['fingerprint'];graph=p['graph']
        model=ImplicitALS(dim=settings.get('dim',32),regularization=p['cf_regularization'],
                          steps=settings.get('als_steps',8),seed=seed).fit(matrix)
        initial=unit(model.users);reports=[]
        def assess(step,values):
            if step not in budgets:return
            source_vectors=assemble(base.codes,fp.bank,values,graph,(1.,.15,.5,.1))
            scores=[]
            for obs in validation:
                population=np.flatnonzero(obs.events>=300)
                partners=partner_frame(old,contacts(obs),hosts,population)
                result=partner_ranking(frame(source_vectors[population],[hosts[i] for i in population]),partners)
                scores.append(result['new']['embedding']['mrr'])
            reports.append({'steps':step,'benchmark_mrr':float(np.mean(scores))})
            print('historical listwise budget',seed,reports[-1],flush=True)
        assess(0,initial)
        refine_factors(initial,base.codes,fp.bank,graph,fit_tasks,steps=max(budgets),seed=seed,on_step=assess)
        selected=max(reports,key=lambda r:(r['benchmark_mrr'],-r['steps']))['steps']
        p['partner'],p['partner_log']=refine_factors(p['cf_initial'],base.codes,fp.bank,graph,all_tasks,
                                                    steps=selected,seed=seed)
        p['historical_listwise_validation']={'fit_days':[o.window.day for o in before],
            'validation_days':[o.window.day for o in validation],'budgets':reports,'selected_steps':selected,
            'conditional_on_frozen_base_and_fingerprint':True,'refit_all_allowed_train_days':True}
        folder=args.output/f'seed{seed}';folder.mkdir()
        joblib.dump(p,folder/'blocks.joblib',compress=3)
        (folder/'historical_selection.json').write_text(json.dumps(p['historical_listwise_validation'],indent=2)+'\n')
        new_seeds.append({'seed':seed,'checkpoint_sha256':signature(folder/'blocks.joblib')})
    freeze.update(seeds=new_seeds,head_tuned_from=str(args.trained),
                  head_selection='historical actual benchmark MRR, followed by refit; conditional on frozen base/fingerprint')
    (args.output/'frozen_training.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print('historical selection complete',args.output,flush=True)


if __name__=='__main__':main()
