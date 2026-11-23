"""Evaluate a frozen Identity-only residual on previously studied days14–16."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from evaluation.identity import identity_retrieval
from evaluation import scoreboard
from .model import ModelConfig,TemporalDonorIdentity


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path);a=ap.parse_args()
    frozen=json.loads((a.run/'frozen_selection.json').read_text())
    if not frozen['training_completed'] or sha(a.run/'model.pt')!=frozen['checkpoint_sha256']:
        raise ValueError('checkpoint is not the frozen selection')
    output=a.run/'benchmark_evaluation.json'
    if output.exists():raise FileExistsError('evaluation already exists; do not repeat the benchmark')
    base=Path('PT_PROJECT_ML/late_test');vec=base/'vectors/multiday_codebook'
    source=pd.read_csv(vec/'embeddings.csv',dtype={'host_id':str}).set_index('host_id')
    checkpoint=torch.load(a.run/'model.pt',weights_only=True)
    if checkpoint.get('experiment',{}).get('feature_family', 'baseline_vector_only') != 'baseline_vector_only':
        raise ValueError('this checkpoint requires contact features; evaluate_fixed cannot substitute zero context')
    model=TemporalDonorIdentity(ModelConfig(**checkpoint['model_config']));model.load_state_dict(checkpoint['state_dict']);model.eval();torch.set_num_threads(4)
    results={};scores_old=[];scores_new=[]
    for w in ['test','test_day15','test_day16']:
        h=pd.read_csv(base/'reference'/w/'eval_hosts.csv',dtype=str).host_id.tolist()
        late=pd.read_csv(vec/(w+'.csv'),dtype={'host_id':str}).set_index('host_id').loc[h]
        src=source.loc[h]
        x=torch.from_numpy(late.to_numpy(dtype=np.float32))
        with torch.no_grad():result=model.late_vectors(x,torch.zeros(len(x),model.config.context_dim)).numpy()
        candidate=pd.DataFrame(result,index=late.index,columns=late.columns)
        old=identity_retrieval(src,late,alignment='procrustes',seed=0,n_boot=0)
        new=identity_retrieval(src,candidate,alignment='procrustes',seed=0,n_boot=0)
        cached=json.loads((Path('runs/original_codebook_recovery')/(w+'.json')).read_text())
        if abs(cached['metrics']['identity_top1']-old['top1'])>1e-12:raise ValueError('cached source baseline identity mismatch')
        cols=[s['metric'] for s in cached['summary']]
        corrected=json.loads(json.dumps(cached));corrected['metrics']['identity_top1']=new['top1']
        score0=scoreboard._one_score(cached,cols,window=w);score1=scoreboard._one_score(corrected,cols,window=w)
        scores_old.append(score0);scores_new.append(score1)
        candidate.astype('float64').rename_axis('host_id').to_csv(a.run/(w+'.csv'))
        results[w]={'baseline_identity':old['top1'],'candidate_identity':new['top1'],
                    'baseline_score':score0,'candidate_score':score1,'n_eval':new['n_eval_hosts']}
        print(w,results[w],flush=True)
    report={'scope':'previously studied benchmark windows, not independent holdout','fixed_checkpoint_sha256':frozen['checkpoint_sha256'],
            'checkpoint_step':frozen['selected_step'],'no_test_tuning':True,'windows':results,
            'baseline_score_mean':float(np.mean(scores_old)),'candidate_score_mean':float(np.mean(scores_new)),
            'score_gain':float(np.mean(scores_new)-np.mean(scores_old)),
            'source_metrics':'reused unchanged from reproduced original codebook, same source and dimension',
            'limitations':'identity rerun; source metrics cached from streaming logistic fit, numerics differ slightly from published original'}
    output.write_text(json.dumps(report,indent=2));print('result',report,flush=True)

if __name__=='__main__':main()
