"""Fixed source: compare soft/hard partial assignments on development, freeze, then test."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .prepare import config_file,environment
from .data import Window,read_window,signature
from .matching import late_vectors,harden_assignment,GraphMatcher
from .model import SimilarityGraphModel
from .evaluate import geometry,with_identity
from .reference import resolve as resolve_reference


def frame(values,hosts):return pd.DataFrame(values,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','checkpoint','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--joint-context',action='store_true')
    args=parser.parse_args();config=config_file(args.config)
    if args.output.exists():raise FileExistsError(args.output)
    original=joblib.load(args.checkpoint);hosts=original.hosts;source=frame(original.source,hosts)
    args.output.mkdir(parents=True)
    dev=[Window(**w) for w in config['development']];tests=[Window(**w) for w in config.get('test',[])]
    if original.training_end>min(w.start for w in dev):raise ValueError('model overlaps development')
    if tests and max(w.end for w in dev)>min(w.start for w in tests):raise ValueError('development overlaps test')
    banks={'original':original.matcher}
    if args.joint_context:
        history=[read_window(Window(**w),hosts,config['cache']) for w in config['train']]
        if max(o.window.end for o in history)>original.training_end:raise ValueError('new profiles exceed fixed-source cutoff')
        print('fitting joint peer/service traffic profiles from train only',flush=True)
        banks['joint_context']=GraphMatcher().fit(history,hosts,joint_context=True,cache=config['cache'])
    rows=[]
    diagnostics={}
    source_obs=read_window(Window(**config['source_window']),hosts,config['cache'])
    def reference(w,obs):
        path=Path(config.get('references',{}).get(w.name,args.checkpoint.parent/'references'/w.name))
        return resolve_reference(path,source_obs,obs,hosts,args.output/'references'/w.name,config)
    for w in dev:
        obs=read_window(w,hosts,config['cache']);ref=reference(w,obs)
        core=geometry(source,ref,config['probe_reference'],config['cache'])
        for bank,matcher in banks.items():
            candidate,_=matcher.unary(obs,k=30);active=obs.events>=300;truth=np.arange(len(hosts))
            diagnostics[f'{w.name}/{bank}']={'candidate_recall30':float(np.any(candidate[active]==truth[active,None],axis=1).mean()),
                'unary_top1':float((candidate[active,0]==truth[active]).mean()),'identity_labels_used_only_for_reporting':True}
            for weight in config.get('graph_weights',[0,.1,.25]):
                P,dust,info=matcher.assign(obs,graph_weight=weight,iterations=original.iterations)
                for mode in ['soft','hard']:
                    assignment,unknown=harden_assignment(P,dust) if mode=='hard' else (P,dust)
                    result=with_identity(source,frame(late_vectors(assignment,unknown,original.source),hosts),ref,core)
                    rows.append({'window':w.name,'profile_bank':bank,'graph_weight':weight,'assignment_mode':mode,'score':result['score'],'result':result})
                    print('matching development',w.name,bank,weight,mode,result['score'],flush=True)
    choices=[]
    for bank in banks:
        for weight in config.get('graph_weights',[0,.1,.25]):
            for mode in ['soft','hard']:
                choices.append({'profile_bank':bank,'graph_weight':weight,'assignment_mode':mode,
                                'score':float(np.mean([r['score'] for r in rows if r['profile_bank']==bank and r['graph_weight']==weight and r['assignment_mode']==mode]))})
    best=min(choices,key=lambda r:(-r['score'],r['profile_bank']!='original',r['assignment_mode']!='soft',r['graph_weight']))
    model=SimilarityGraphModel(original.source,banks[best['profile_bank']],best['graph_weight'],original.iterations,best['assignment_mode'])
    joblib.dump(model,args.output/'model.joblib',compress=3);source.to_csv(args.output/'source.csv')
    frozen={'fixed_source_checkpoint_sha256':signature(args.checkpoint),'source_sha256':signature(args.output/'source.csv'),
            'checkpoint_sha256':signature(args.output/'model.joblib'),'choice':best,'test_not_read_before_freeze':True,
            'config':config,'versions':environment(),
            'code_sha256':{p.name:signature(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')},
            'training_files':{w['name']:{'bytes':Path(w['flows']).stat().st_size,'mtime_ns':Path(w['flows']).stat().st_mtime_ns} for w in config['train']}}
    (args.output/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    report={'selection':frozen,'development':rows,'candidate_diagnostics':diagnostics,'test':{},'official_leaderboard_result':False}
    for w in tests:
        obs=read_window(w,hosts,config['cache']);ref=reference(w,obs)
        late,info=model.predict(obs);core=geometry(source,ref,config['probe_reference'],config['cache'],100)
        result=with_identity(source,frame(late,hosts),ref,core,100);result['matching']=info
        report['test'][w.name]=result;frame(late,hosts).to_csv(args.output/f'{w.name}.csv')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('matching refinement complete',{k:v['score'] for k,v in report['test'].items()},flush=True)


if __name__=='__main__':main()
