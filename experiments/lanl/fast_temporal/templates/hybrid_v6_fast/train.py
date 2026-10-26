from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ["PT_REPO_ROOT"])
WORK = Path(os.environ["PT_RUN_ROOT"])
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from evaluation.io import load_embeddings
from evaluation.scoreboard import overall_score, headline_table, check_compatible
from model import Gate, SIZES, apply_gate, reliability

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('hybrid_core',WORK / 'hybrid_v4/train.py')
core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
core.OUT=OUT
core.STATUS={'state':'running','started_at':datetime.now().astimezone().isoformat(),'pid':os.getpid()}
log,dump=core.log,core.dump
SOURCE=WORK / 'hybrid_v5/candidates/v5_candidate_07'
torch.set_num_threads(2)
torch.manual_seed(29)
np.random.seed(29)


def frame(x, hosts):
    if not np.isfinite(x).all() or (np.linalg.norm(x,axis=1)==0).any():
        raise ValueError('invalid embedding')
    return pd.DataFrame(x,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')


def main():
    # Шесть вариантов объявлены до чтения результатов нового holdout.
    recipes=[{'alpha':0.0,'dim':256},{'alpha':0.0,'dim':384},
             {'alpha':0.5,'dim':0},{'alpha':1.0,'dim':0},
             {'alpha':1.0,'dim':256},{'alpha':1.0,'dim':384}]
    dump(OUT/'plan.json',{'recipes':recipes,'epochs':20,'training_budget_seconds':60,
                          'selection_budget_seconds':300,'seed':29,'baseline':'v5 checkpoint20',
                          'fit_windows':['train'],'selection_window':'validation',
                          'fresh_holdout':[140400,142200],
                          'test_policy':'select before fresh holdout, no retuning afterwards'})
    log('load_cached_embeddings')
    train=load_embeddings(SOURCE/'embeddings.csv'); val=load_embeddings(SOURCE/'validation.csv')
    x=train.to_numpy(); xv=val.to_numpy()
    if x.shape[1] != sum(SIZES):
        raise ValueError('unexpected v5 block widths')
    baseline=json.loads((SOURCE/'validation_result.json').read_text())
    base_score=float(overall_score([baseline],{s['metric']:s['title'] for s in baseline['summary']})[baseline['method']])
    records=[{'name':'v5_baseline','recipe':{'alpha':0.0,'dim':0},'validation_score':base_score}]
    scaler=StandardScaler().fit(reliability(x))
    feature=torch.from_numpy(np.clip(scaler.transform(reliability(x)),-8,8).astype(np.float32))
    xt=torch.from_numpy(x.astype(np.float32))
    teacher=F.normalize(xt[:,:57],dim=1)
    gate=Gate(); optimizer=torch.optim.AdamW(gate.parameters(),lr=0.002,weight_decay=0.01)
    rng=np.random.default_rng(29)
    start=time.monotonic()
    for epoch in range(1,21):
        gate.train(); losses=[]
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
            index=torch.from_numpy(ix)
            weight=gate(feature[index])
            expanded=torch.repeat_interleave(weight,torch.tensor(SIZES),dim=1)
            a=F.normalize(F.dropout(xt[index],p=0.1,training=True)*expanded,dim=1)
            b=F.normalize(F.dropout(xt[index],p=0.1,training=True)*expanded,dim=1)
            logits=a@b.T/0.12
            labels=torch.arange(len(ix))
            contrastive=0.5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
            role=F.mse_loss(a@a.T,teacher[index]@teacher[index].T)
            loss=contrastive+0.5*role+0.2*((weight-1)**2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        log('gate_training',epoch=epoch,epochs=20,loss=float(np.mean(losses)),seconds=round(time.monotonic()-start,1))
        if time.monotonic()-start>=60:
            break
    torch.save(gate.state_dict(),OUT/'gate.pt'); joblib.dump(scaler,OUT/'gate_scaler.joblib')
    train_seconds=time.monotonic()-start
    selection_start=time.monotonic()
    projections={}
    for number,recipe in enumerate(recipes,1):
        if time.monotonic()-selection_start>300:
            log('selection_budget_reached',completed=number-1)
            break
        log('validation',candidate=number,total_candidates=6,recipe=recipe)
        a=apply_gate(x,gate,scaler,recipe['alpha']); b=apply_gate(xv,gate,scaler,recipe['alpha'])
        if recipe['dim']:
            if recipe['alpha'] not in projections:
                projection=PCA(n_components=384,svd_solver='randomized',random_state=29).fit(a)
                projections[recipe['alpha']]=projection
                joblib.dump(projection,OUT/f"projection_alpha{recipe['alpha']:g}.joblib")
            projection=projections[recipe['alpha']]
            a=projection.transform(a)[:,:recipe['dim']]; b=projection.transform(b)[:,:recipe['dim']]
        name=f'fast_candidate_{number:02d}'
        score,_=core.evaluate(name,frame(a,train.index),frame(b,val.index),'validation')
        records.append({'name':name,'recipe':recipe,'validation_score':score})
        dump(OUT/'search_results.json',records)
        log('validation_done',candidate=number,score=score,best=max(r['validation_score'] for r in records))
    selected=max(records,key=lambda r:r['validation_score'])
    dump(OUT/'selected.json',selected)
    log('selection_frozen',selected=selected,training_seconds=round(train_seconds,2))
    from model import transform
    chosen_train=frame(transform(x,OUT,selected),train.index)
    chosen_train.to_csv(OUT/'embeddings.csv',float_format='%.8g')
    if selected['name']=='v5_baseline':
        result=json.loads((WORK / 'hybrid_v5/test_result.json').read_text())
        log('baseline_retained')
    else:
        oldtest=load_embeddings(WORK / 'hybrid_v5/candidates/hybrid_temporal_graph_v5/test.csv')
        test=frame(transform(oldtest.to_numpy(),OUT,selected),oldtest.index)
        log('historical_test_evaluation')
        _,result=core.evaluate('hybrid_adaptive_fast_v6',chosen_train,test,'test',200)
    dump(OUT/'test_result.json',result)
    previous=json.loads((WORK / 'hybrid_v5/test_result.json').read_text())
    leader=json.loads((ROOT/'scoreboard/results/hybrid_tfidf.json').read_text())
    comparisons=[leader,previous]+([result] if selected['name']!='v5_baseline' else [])
    check_compatible(comparisons)
    (OUT/'comparison.md').write_text('# Fast v6: historical test (exploratory)\n\n'+headline_table(comparisons))
    score=float(overall_score([result],{s['metric']:s['title'] for s in result['summary']})[result['method']])
    log('completed',state='completed',test_score=score,selected=selected,
        training_seconds=round(train_seconds,2),finished_at=datetime.now().astimezone().isoformat())


if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        log('failed',state='failed',error=repr(e))
        raise
