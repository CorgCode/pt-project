"""Оценка замороженных моделей на заранее выбранном позднем окне day02."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(os.environ["PT_REPO_ROOT"])
WORK = Path(os.environ["PT_RUN_ROOT"])
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(WORK / 'hybrid_v6_fast'))

def save(status):
    status['updated_at']=datetime.now().astimezone().isoformat()
    p=OUT/'evaluation_status.json'
    temp=p.with_suffix('.tmp'); temp.write_text(json.dumps(status,indent=2)+'\n'); os.replace(temp,p)
    print(status['updated_at'],status.get('phase'),flush=True)

status={'state':'running','phase':'waiting_for_data_and_selection','pid':os.getpid()}
save(status)
try:
    deadline=time.monotonic()+3600
    while True:
        prep=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
        search_path=WORK / 'hybrid_v6_fast/status.json'
        search=json.loads(search_path.read_text()) if search_path.exists() else {}
        if prep.get('state')=='failed' or search.get('state')=='failed':
            raise RuntimeError(f"preparation/search failed: {prep.get('error') or search.get('error')}")
        if prep.get('state')=='completed' and (WORK / 'hybrid_v6_fast/selected.json').exists():
            break
        if time.monotonic()>deadline:
            raise TimeoutError('preparation or model selection exceeded one hour')
        time.sleep(15)
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from evaluation.io import load_embeddings
    from evaluation.scoreboard import overall_score,headline_table,check_compatible
    from pipeline.data import ProcessedData
    from methods.baselines.temporal import BehavioralBaseline
    from model import transform
    spec=importlib.util.spec_from_file_location('v5_frozen',WORK / 'hybrid_v5/train.py')
    v5=importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
    v5.DATA=OUT; v5.core.DATA=OUT
    v5.core.OUT=OUT/'evaluation'
    v5.core.OUT.mkdir(exist_ok=True)
    v5.core.STATUS={'state':'running','pid':os.getpid()}
    data=ProcessedData(OUT/'processed')
    cache=joblib.load(WORK / 'hybrid_v5/preprocessing.joblib')
    if data.cohort!=cache['hosts']:
        raise ValueError('Fresh holdout must preserve original training cohort')
    status['phase']='fresh_feature_extraction'; save(status)
    raw,_=v5.collect('test',data.cohort,cache['vocabs'])
    values=v5.features(raw['test'],data.cohort,cache['models'],cache['vocabs'])
    del raw
    hs=data.evaluable('test'); rows=pd.Index(data.cohort).get_indexer(hs)
    behavior=BehavioralBaseline().fit(data).embed(data,'test',hs).to_numpy()
    checkpoint=torch.load(WORK / 'hybrid_v5/encoder_epoch20.pt',map_location='cpu',weights_only=True)
    net=v5.Encoder(checkpoint['inputs']); net.load_state_dict(checkpoint['state_dict'])
    inputs=np.clip(cache['scaler'].transform(values['input']),-8,8).astype(np.float32)
    blocks={kind:v5.core.block(values[kind][rows]) for kind in ('peers','services','diffusion')}
    blocks['behavioral']=v5.core.block(behavior)
    blocks['neural']=v5.core.block(v5.infer(net,inputs)[rows])
    def frame(parts):
        return pd.DataFrame(np.hstack(parts),index=pd.Index(hs,name='host_id')).add_prefix('emb_')
    original=frame([blocks['behavioral']*2,blocks['peers'],blocks['services']])
    neural=frame([blocks['behavioral']*2,blocks['peers'],blocks['services'],blocks['diffusion']*0.5,blocks['neural']])
    selected=json.loads((WORK / 'hybrid_v6_fast/selected.json').read_text())
    candidates=[('hybrid_tfidf_reproduced',WORK / 'hybrid_v5/candidates/v5_candidate_00/embeddings.csv',original),
                ('hybrid_temporal_graph_v5',WORK / 'hybrid_v5/candidates/v5_candidate_07/embeddings.csv',neural)]
    if selected['name']!='v5_baseline':
        newtest=pd.DataFrame(transform(neural.to_numpy(),WORK / 'hybrid_v6_fast',selected),index=neural.index).add_prefix('emb_')
        candidates.append(('hybrid_adaptive_fast_v6',WORK / 'hybrid_v6_fast/embeddings.csv',newtest))
    results=[]
    for name,path,test in candidates:
        status.update(phase='evaluate/'+name);save(status)
        score,result=v5.core.evaluate(name,load_embeddings(path),test,'test',200)
        results.append(result)
        v5.core.dump(OUT/f'{name}_result.json',result)
        print(name,'fresh holdout score',score,flush=True)
    check_compatible(results)
    (OUT/'comparison.md').write_text('# Fresh temporal holdout: day02 15:00–15:30\n\n'
        'Training: day02 11:00–12:00. Selection: 12:00–12:30. '
        'This late window was fixed before v6 selection. No model is fitted or retuned on it. '
        'It is a later window of the same day, not an independent day.\n\n'+headline_table(results))
    scores={r['method']:float(overall_score([r],{s['metric']:s['title'] for s in r['summary']})[r['method']]) for r in results}
    status.update(state='completed',phase='completed',scores=scores,finished_at=datetime.now().astimezone().isoformat());save(status)
except BaseException as e:
    status.update(state='failed',phase='failed',error=repr(e));save(status)
    raise
