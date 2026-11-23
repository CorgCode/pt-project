"""Select exact/compact S models and orthogonal Links rotation by FULL development score."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .data import Window,read_window,contacts,signature,check_schedule
from .prepare import config_file,environment
from .matching import late_vectors
from .metrics import WindowTask
from .model import SimilarityGraphModel
from .evaluate import geometry,with_identity
from .rotation import fit_rotation,apply_rotation
from .reference import build as build_reference,resolve as resolve_reference


def frame(x,hosts):return pd.DataFrame(x,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def preservation(S,Z,task):
    before=task.summary(S)
    cosine=normalize(Z)@normalize(Z).T
    after=task.summary(cosine)
    overlap=np.mean([len(set(task.neighbors(S,q)[0])&set(task.neighbors(cosine,q)[0]))/min(10,len(task.population)-1)
                     for q in task.population])
    return {'matrix':before,'vectors':after,'top10_overlap':float(overlap),
            'new_partners_mrr_difference':after['new_partners_mrr']-before['new_partners_mrr']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','trained','built','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();config=config_file(args.config)
    if args.output.exists():raise FileExistsError(args.output)
    train=[Window(**w) for w in config['train']];dev=[Window(**w) for w in config['development']]
    tests=[Window(**w) for w in config.get('test',[])];check_schedule(train,dev,tests)
    built=json.loads((args.built/'frozen_build.json').read_text())
    training=json.loads((args.trained/'frozen_training.json').read_text())
    if training['train']!=[asdict(w) for w in sorted(train,key=lambda w:w.start)]:raise ValueError('different training schedule')
    if built['training_end']>min(w.start for w in dev):raise ValueError('training overlaps development')
    for w in train:
        stat=Path(w.flows).stat();record=training['training_files'][w.name]
        if (stat.st_size,stat.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):raise ValueError('training flows changed')
    args.output.mkdir(parents=True)
    hosts=built['hosts'];matcher=joblib.load(args.built/'matcher.joblib')
    if signature(args.built/'matcher.joblib')!=built['matcher_sha256']:raise ValueError('matcher changed')
    read=lambda w:read_window(w,hosts,config['cache'])
    source_obs=read(Window(**config['source_window']))
    development=[read(w) for w in dev]
    refs={}
    def reference(obs):
        path=config.get('references',{}).get(obs.window.name)
        if path:
            return resolve_reference(path,source_obs,obs,hosts,args.output/'references'/obs.window.name,config)
        path=args.output/'references'/obs.window.name
        build_reference(source_obs,obs,hosts,path,config['cache']);return path
    for obs in development:refs[obs.window.name]=reference(obs)
    weights=config.get('graph_weights',[0.,.1,.25])
    assignments={};assignment_info={}
    for obs in development:
        for weight in weights:
            print('joint matching development',obs.window.name,weight,flush=True)
            P,dust,info=matcher.assign(obs,graph_weight=weight,iterations=config.get('matching_iterations',5))
            assignments[obs.window.name,weight]=(P,dust);assignment_info[obs.window.name,weight]=info
    records=[];preserved={}
    for label,metadata in built['candidates'].items():
        path=args.built/f'{label}.npy'
        if signature(path)!=metadata['sha256']:raise ValueError('vector candidate changed')
        vectors=np.load(path);source=frame(vectors,hosts)
        S=np.load(args.trained/f"{metadata['matrix']}.npy")
        cores={}
        for obs in development:
            task=WindowTask(contacts(source_obs),obs,hosts,refs[obs.window.name])
            preserved[label,obs.window.name]=preservation(S,vectors,task)
            cores[obs.window.name]=geometry(source,refs[obs.window.name],config['probe_reference'],config['cache'])
        for weight in weights:
            results={}
            for obs in development:
                P,dust=assignments[obs.window.name,weight]
                late=frame(late_vectors(P,dust,vectors),hosts)
                print('evaluating Identity',label,weight,flush=True)
                results[obs.window.name]=with_identity(source,late,refs[obs.window.name],cores[obs.window.name])
            row={'candidate':label,'graph_weight':weight,'rotated':False,
                 'score':float(np.mean([r['score'] for r in results.values()])),
                 'results':results,'dim':vectors.shape[1]}
            records.append(row);print('development score',label,weight,row['score'],flush=True)
            (args.output/'development.json').write_text(json.dumps(records,indent=2)+'\n')
        del vectors,S,source,cores
    best=min(records,key=lambda r:(-r['score'],r['dim'],r['graph_weight']))
    vectors=np.load(args.built/f"{best['candidate']}.npy")
    # Train an orthogonal rotation only on declared historical target edges.
    edges=pd.concat([pd.read_csv(p,dtype={'host_a':str,'host_b':str})
                     for p in (args.trained/'training_references').glob('*/edges.csv')],ignore_index=True)
    rotation=fit_rotation(vectors,hosts,edges,steps=config.get('rotation_steps',150))
    rotated=apply_rotation(vectors,rotation)
    if np.max(np.abs(vectors[:64]@vectors.T-rotated[:64]@rotated.T))>1e-10:
        raise FloatingPointError('orthogonal rotation changed source cosines')
    results={}
    for obs in development:
        source=frame(rotated,hosts);P,dust=assignments[obs.window.name,best['graph_weight']]
        core=geometry(source,refs[obs.window.name],config['probe_reference'],config['cache'])
        results[obs.window.name]=with_identity(source,frame(late_vectors(P,dust,rotated),hosts),refs[obs.window.name],core)
    rotated_row={**best,'rotated':True,'score':float(np.mean([r['score'] for r in results.values()])),'results':results}
    records.append(rotated_row)
    if rotated_row['score']>best['score']+1e-10:best=rotated_row;vectors=rotated
    model=SimilarityGraphModel(vectors,matcher,best['graph_weight'],config.get('matching_iterations',5))
    joblib.dump(model,args.output/'model.joblib',compress=3)
    frame(vectors,hosts).to_csv(args.output/'source.csv')
    frozen={'candidate':best['candidate'],'graph_weight':best['graph_weight'],'rotated':best['rotated'],
            'selected_build':built['candidates'][best['candidate']],
            'versions':environment(),
            'development_score':best['score'],'dim':vectors.shape[1],
            'training_end':training['training_end'],'checkpoint_sha256':signature(args.output/'model.joblib'),
            'source_sha256':signature(args.output/'source.csv'),'test_not_read_before_freeze':True,
            'test_policy':config['test_policy'],'config':config,
            'code_sha256':{p.name:signature(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}}
    (args.output/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    report={'selection':frozen,'development':records,
            'preservation':{f'{k[0]}/{k[1]}':v for k,v in preserved.items()},
            'matching':{f'{k[0]}/{k[1]}':v for k,v in assignment_info.items()},
            'rotation_training':rotation['history'],'test':{},'official_leaderboard_result':False}
    for w in tests:
        obs=read(w)  # Test processing begins only after the checkpoint and selection are frozen.
        ref=reference(obs);late,info=model.predict(obs)
        source=frame(vectors,hosts)
        core=geometry(source,ref,config['probe_reference'],config['cache'],100)
        r=with_identity(source,frame(late,hosts),ref,core,100)
        r['matching']=info;report['test'][w.name]=r
        frame(late,hosts).to_csv(args.output/f'{w.name}.csv')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('completed',args.output,{k:v['score'] for k,v in report['test'].items()},flush=True)


if __name__=='__main__':main()
