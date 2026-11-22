"""Выбор на дне 13, фиксация source и оценка на канонических днях 14–16."""
from .config import load as load_config
import argparse,json,shutil
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from evaluation import scoreboard
from .data import prepare,sha
from .model import TemporalMixer,encode,plain,pooled_source
from .common.evaluate import geometry,with_identity
from .fit import code_shas


def frame(x,hosts):return pd.DataFrame(np.asarray(x,np.float64),index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=TemporalMixer(saved['dim'],saved['mode'],saved['hidden']);model.load_state_dict(saved['state']);model.eval();return model


def reference(window,observation,source,output):
    ref=Path(window['reference']);meta=json.loads((ref/'manifest.json').read_text())
    spans=meta.get('inputs')
    if spans is not None:
        declared_source,declared_target=spans['source'],spans['target']
    else:
        declared_source=meta['windows'][meta.get('source_window','train')]
        declared_target=meta['windows'][meta['target_window']]
    for declared,observed in [(declared_source,source['window']),(declared_target,observation['window'])]:
        if (declared['start'],declared['end'])!=(observed['start'],observed['end']):raise ValueError('reference window bounds differ')
    hosts=observation['hosts'];active=[hosts[i] for i in np.flatnonzero(observation['events']>=300)]
    if set(active)!=set(pd.read_csv(ref/'eval_hosts.csv',dtype=str).host_id):raise ValueError('active reference cohort differs')
    for name,digest in meta['files_sha256'].items():
        if name.startswith('._'):continue
        if sha(ref/name)!=digest:raise ValueError('reference CSV changed')
    # С эталоном сверяются полные часовые частоты, до quarter-фильтра.
    partners=pd.read_csv(ref/'partners.csv',dtype={'host_id':str,'partner':str});active_set=set(active)
    for label,obs in [('source',source),('target',observation)]:
        observed={(hosts[i],str(p)):int(v) for i in range(len(hosts)) if hosts[i] in active_set for p,v in obs['contacts'][i].items()}
        expected={(str(h),str(p)):int(v) for h,p,v in partners.loc[partners.window==label,['host_id','partner','count']].itertuples(index=False,name=None)}
        if observed!=expected:raise ValueError(f'{label} frequencies differ from reference')
    if 'neighbor_ceiling' not in meta:
        key=window['name']
        canonical=Path(__file__).resolve().parents[2]/'scoreboard/data/reference'/key/'manifest.json'
        if not canonical.exists() or meta['files_sha256']!=json.loads(canonical.read_text())['files_sha256']:raise ValueError('no canonical/local ceiling')
        meta['neighbor_ceiling']=scoreboard.NEIGHBORS_IDEAL[key]
    output.mkdir(parents=True,exist_ok=False)
    for p in ref.glob('*.csv'):
        if not p.name.startswith('._'):shutil.copyfile(p,output/p.name)
    meta['files_sha256']={p.name:sha(p) for p in output.glob('*.csv')}
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');return output


def evaluate_candidate(source,late,hosts,ref,config,output):
    output.mkdir(parents=True,exist_ok=False)
    a,b=output/'source.csv',output/'late.csv';frame(source,hosts).to_csv(a,float_format='%.17g');frame(late,hosts).to_csv(b,float_format='%.17g')
    # Оцениваются точные координаты после экспорта и чтения CSV.
    sa=pd.read_csv(a,index_col=0);sb=pd.read_csv(b,index_col=0)
    core=geometry(sa,ref,config['probe_reference'],config.get('eval_cache',output.parent/'eval_cache'))
    result=with_identity(sa,sb,ref,core)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','trained','output']:parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--part',choices=['development','benchmark'],required=True)
    args=parser.parse_args();c=load_config(args.config);frozen=json.loads((args.trained/'frozen_training.json').read_text())
    if args.output.exists():raise FileExistsError(args.output)
    if not frozen.get('development_test_read') is False:raise ValueError('training freeze missing')
    args.output.mkdir(parents=True);hosts=frozen['hosts']
    if frozen['train']!=c['train'] or frozen['training_end']>c['development']['start']:raise ValueError('history/cutoff differ')
    if sha(args.trained/'transforms.joblib')!=frozen['transforms_sha256']:raise ValueError('transforms changed')
    transforms=joblib.load(args.trained/'transforms.joblib')
    source_obs=joblib.load(prepare(c['train'][0],hosts,c['feature_dir']));xs=np.load(args.trained/f"{c['train'][0]['name']}.npy")
    def infer(label,obs):
        x=transforms.transform(obs)
        if label=='plain':return plain(xs),plain(x)
        model=checkpoint(args.trained/f'{label}.pt');weights=frozen['candidates'][label].get('weights',(2.,1.,.5))
        if frozen['candidates'][label].get('source_strategy')=='history_pool':
            history=[np.load(args.trained/f"{w['name']}.npy") for w in c['train']]
            presence=[joblib.load(prepare(w,hosts,c['feature_dir']))['events']>=300 for w in c['train']]
            source=pooled_source(model,history,[w['day'] for w in c['train']],weights=weights,presence=presence)
        else:source=encode(model,xs,weights)
        return source,encode(model,x,weights)
    if args.part=='development':
        obs=joblib.load(prepare(c['development'],hosts,c['feature_dir']));ref=reference(c['development'],obs,source_obs,args.output/'reference')
        rows=[]
        for label,meta in frozen['candidates'].items():
            if sha(args.trained/f'{label}_source.csv')!=meta['source_sha256']:raise ValueError('source candidate changed')
            if label!='plain' and sha(args.trained/f'{label}.pt')!=meta['checkpoint_sha256']:raise ValueError('candidate model changed')
            a,b=infer(label,obs);result=evaluate_candidate(a,b,hosts,ref,c,args.output/label)
            rows.append({'label':label,'score':result['score'],'metrics':result['metrics']});print('development',rows[-1],flush=True)
            (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        baseline=rows[0];eligible=[r for r in rows if r['metrics']['consistency@10_mean']>=.95*baseline['metrics']['consistency@10_mean']]
        best=max(eligible,key=lambda r:r['score'])
        selected={'selected':best,'candidates':rows,'chosen_on':'previously studied day13 development only','selection_uses_development_metrics_only':True,
                  'independent_holdout':False,'benchmark_previously_studied':True,
                  'trained':str(args.trained),'training_sha256':sha(args.trained/'frozen_training.json'),
                  'source_sha256':sha(args.output/best['label']/'source.csv'),'code_sha256':code_shas()}
        (args.output/'frozen_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    else:
        selection_path=Path(c['selection']);selected=json.loads(selection_path.read_text())
        if selected['training_sha256']!=sha(args.trained/'frozen_training.json'):raise ValueError('training changed since selection')
        label=selected['selected']['label'];meta=frozen['candidates'][label]
        if label!='plain' and sha(args.trained/f'{label}.pt')!=meta['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
        selected_source=Path(c['selection']).parent/label/'source.csv'
        if sha(selected_source)!=selected['source_sha256']:raise ValueError('selected source changed')
        controls=['plain'] if label!='plain' else []
        paired=(label.replace('no_graph','graph',1) if label.startswith('no_graph') else label.replace('graph','no_graph',1) if label.startswith('graph') else None)
        if paired in frozen['candidates'] and paired not in controls:controls.append(paired)
        plan={'selected':label,'controls':controls,'frozen_selection_sha256':sha(selection_path),'code_sha256':code_shas(),
              'plan_written_before_this_invocation_benchmark_reads':True,'independent_holdout':False,'benchmark_previously_studied':True}
        for candidate in [label,*controls]:
            meta=frozen['candidates'][candidate]
            if candidate!='plain' and sha(args.trained/f'{candidate}.pt')!=meta['checkpoint_sha256']:raise ValueError('control checkpoint changed')
        (args.output/'benchmark_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        results={};control_results={control:{} for control in controls}
        for w in c['benchmark']:
            obs=joblib.load(prepare(w,hosts,c['feature_dir']));ref=reference(w,obs,source_obs,args.output/'references'/w['name'])
            a,b=infer(label,obs);results[w['name']]=evaluate_candidate(a,b,hosts,ref,c,args.output/w['name'])
            print('benchmark',label,w['name'],results[w['name']]['score'],flush=True)
            for control in controls:
                a,b=infer(control,obs)
                control_results[control][w['name']]=evaluate_candidate(a,b,hosts,ref,c,args.output/'controls'/control/w['name'])
                print('benchmark control',control,w['name'],control_results[control][w['name']]['score'],flush=True)
            report={'selection':selected,'by_window':results,'mean_score':float(np.mean([r['score'] for r in results.values()])),
                    'controls':control_results,'control_mean_scores':{key:float(np.mean([r['score'] for r in values.values()])) for key,values in control_results.items()},'independent_holdout':False}
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')



if __name__=='__main__':main()
