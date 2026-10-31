"""Полное обучение bipartite encoder; контроль10/12, закрытый13, порог80."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
from sklearn.preprocessing import normalize

from experiments.fingerprint_bigraph import (ALIGNED,ORIGINAL,BiGraphModel,tensor_sparse,source_signature,validate_revision)
from experiments.fingerprint_aligned import passes_gate
from experiments.fingerprint_donor_search import INCUMBENT,PREVIOUS,SOURCE,run,download
from experiments.fingerprint_expanded_search import pool,slim
from experiments.fingerprint_memory import read_fingerprints
from experiments.fingerprint_search import ArrayEmbeddings,BASE,dump,load,state,write_json
from experiments.multiday_finish import freeze_selection
from experiments.multiday_gain import evaluate
from pipeline.data import ProcessedData


def prepare_evaluation(work,bank,pca):
    cells={
        'day10':(ORIGINAL/'paired_validation',ORIGINAL/'reference_validation/validation','validation',
                 Path(json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']['directory'])/'later.csv'),
        'day12':(PREVIOUS/'paired_test',PREVIOUS/'reference_test/test','test',PREVIOUS/'final/winner/later.csv')}
    result={}
    for day,(data_path,ref,window,aux_file) in cells.items():
        data=ProcessedData(data_path);hosts=data.evaluable(window)
        raw_path=work/f'eval_{day}.npz'
        if raw_path.exists():raw=sp.load_npz(raw_path)
        else:
            counts=read_fingerprints(data,window,bank.hosts,ORIGINAL/'count_cache')
            raw=bank.transform(counts)[pd.Index(bank.hosts).get_indexer(hosts)]
            sp.save_npz(raw_path,raw)
            del counts;gc.collect()
        aux=pd.read_csv(aux_file,index_col=0);aux.index=aux.index.astype(str)
        if set(aux.index)!=set(hosts):raise ValueError('Teacher кэш другой когорты')
        aux=normalize(pca.transform(normalize(aux.loc[hosts].to_numpy(dtype=np.float32)))).astype(np.float32)
        graph=normalize(raw[:,:bank.graph_width],norm='l1')
        result[day]=(data,ref,window,hosts,tensor_sparse(raw),tensor_sparse(graph),torch.from_numpy(aux))
    return result


def selection(work):
    validate_revision(work)
    if (work/'selection_summary.json').exists():return
    torch.set_num_threads(4)
    old=json.loads((PREVIOUS/'final/frozen_selection.json').read_text())['winner']
    baseline={'day10':old['score'],'day12':json.loads((PREVIOUS/'final/summary.json').read_text())['scores']['winner']}
    bank=load(work/'bank.pkl');pca=load(work/'aux_pca.pkl')
    source_aux=torch.from_numpy(np.load(work/'source_aux.npy'))
    graph=normalize(bank.prototype[:,:bank.graph_width],norm='l1')
    p=tensor_sparse(bank.prototype);g=tensor_sparse(graph);reverse=tensor_sparse(normalize(graph.T.tocsr(),norm='l1'))
    cells=prepare_evaluation(work,bank,pca)
    initial=work/'candidates/incumbent';initial.mkdir(parents=True,exist_ok=True)
    initialrow={'name':'incumbent','score':min(baseline.values()),'scores_by_day':baseline,
                'selection_metric':'minimum across development10/12'}
    write_json(initial/'best.json',initialrow)
    if not (initial/'best.pkl').exists():(initial/'best.pkl').symlink_to(INCUMBENT)
    eligible=[initial];records=[initialrow]
    for label in ['graph_rank05','graph_rank20']:
        for epoch in [600,1800]:
            name=f'{label}_e{epoch}';out=work/'candidates'/name
            if (out/'assessment.json').exists():
                row=json.loads((out/'assessment.json').read_text());records.append(row)
                if (out/'best.json').exists():eligible.append(out)
                continue
            net=load(work/label/f'epoch{epoch}.pkl');net.eval()
            with torch.no_grad():z,context=net.prototype(p,g,reverse,source_aux)
            train=pd.DataFrame(z.numpy(),index=bank.hosts).add_prefix('emb_')
            scores,identities,partners={},{},{}
            for day,(data,ref,window,hosts,x,xg,aux) in cells.items():
                if psutil.virtual_memory().available<2*1024**3:raise MemoryError('Меньше2 ГБ RAM перед метриками')
                saved=out/day/'scored.json'
                if saved.exists():
                    row=json.loads(saved.read_text());scores[day]=row['score'];identities[day]=row['identity'];partners[day]=row['partner_new_mrr'];continue
                with torch.no_grad():values=net.observed(x,xg,aux,context).numpy()
                later=pd.DataFrame(values,index=hosts).add_prefix('emb_')
                state(work,'validation',candidate=f'{name}/{day}',completed=len(records),total=5)
                score,result=evaluate(ArrayEmbeddings(train,later),data,window,ref,out/day,f'{name}_{day}')
                scores[day]=score;identities[day]=result['metrics']['identity_top1'];partners[day]=result['metrics']['partner_new_mrr']
                write_json(saved,{'score':score,'identity':identities[day],'partner_new_mrr':partners[day]});gc.collect()
            row={'name':name,'score':min(scores.values()),'scores_by_day':scores,'identity':min(identities.values()),
                 'identity_by_day':identities,'partner_mrr_by_day':partners,'head':label,'epoch':epoch,
                 'gains_by_day':{d:scores[d]-baseline[d] for d in baseline},'native_output_dim':128}
            write_json(out/'assessment.json',row);records.append(row)
            if all(scores[d]>=baseline[d] for d in baseline):write_json(out/'best.json',row);eligible.append(out)
            write_json(work/'validation_results.json',records)
            print(json.dumps(row),flush=True)
            del net;gc.collect()
    winner=min([json.loads((path/'best.json').read_text()) for path in eligible],key=lambda r:(-r['score'],r['name']))
    if winner['name']!='incumbent':
        state(work,'export_model',winner=winner)
        net=load(work/winner['head']/f"epoch{winner['epoch']}.pkl")
        model=BiGraphModel(slim(load(INCUMBENT)),bank,net,pca,source_aux.numpy(),ORIGINAL/'count_cache')
        dump(model,work/'candidates'/winner['name']/'best.pkl')
    freeze_selection(eligible,work/'final')
    write_json(work/'validation_results.json',records)
    write_json(work/'selection_summary.json',{'winner':winner,'baseline':baseline,
        'validation_gain':min(winner['scores_by_day'][d]-baseline[d] for d in baseline),
        'test_will_run':passes_gate(winner['scores_by_day']),'test_gate':'both development scores >=80',
        'day12_is_development':True})


def final_test(work):
    result=json.loads((work/'selection_summary.json').read_text())
    if not result['test_will_run']:
        state(work,'complete',result='below_80_on_development',**result);return
    raw=SOURCE/'multiday/raw/netflow_day-13.bz2'
    if not raw.exists() or raw.stat().st_size!=1499152444:download(work)
    if not (work/'day13/manifest.json').exists():
        run(work,'prepare_day13','experiments.multiday_prepare','--raw',raw,'--base',BASE,'--out',work/'day13','--day',13)
    run(work,'reference_test','experiments.multiday_reference','--base',BASE,'--target',work/'day13',
        '--out',work/'paired_test','--reference',work/'reference_test','--window','test')
    from experiments.fingerprint_quick import candidate_directories
    eligible=[p for p in candidate_directories(work) if (p/'best.json').exists()]
    run(work,'final_evaluation','experiments.multiday_finish','--candidates',*eligible,'--data',work/'paired_test',
        '--reference',work/'reference_test/test','--out',work/'final','--incumbent',INCUMBENT,
        '--test-label','day13,12:30–13:30','--protocol-text',
        'Полностью обучаемый host↔token энкодер,128 координат; source-контакты02, исторические цели03–09. Development10/12; test13 после фиксации.')
    state(work,'complete',report=str(work/'final/REPORT.md'),**json.loads((work/'final/summary.json').read_text()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--selection-only',action='store_true')
    args=ap.parse_args();work=args.work;work.mkdir(parents=True,exist_ok=True)
    if args.selection_only:selection(work);return
    if (work/'final/REPORT.md').exists():
        state(work,'complete',**json.loads((work/'final/summary.json').read_text()));return
    protocol={'train_days':list(range(2,10)),'dimensions':[],'head_labels':['graph_rank05','graph_rank20'],
        'checkpoints':[600,1800],'candidates':5,'validation_day':'10 и12 (development)','development_days':[10,12],
        'test_day':13,'target_score':80,'model':'trainable token embeddings + host↔token message passing + joint classifier/InfoNCE/new-partner ranking',
        'output_dim':128,'raw_training_windows':16,'source_contacts_day':2,'target_hours':'12:30–13:30',
        'auxiliary_teacher':'train-only128 PCA of previous frozen encoder; raw token path is trainable',
        'rank_weights':[0.5,2.0],'epochs':1800,'warmup_steps':200,'test_gate':'both development scores >=80',
        'neighbor_training':'soft to90% positive-cosine top10, with historical target population mask',
        'auxiliary_dropout':0.25,'training_revision':2,
        'original_scoreboard_comparable':False}
    if not (work/'protocol.json').exists():write_json(work/'protocol.json',protocol)
    if not (work/'inputs_ready.json').exists():
        run(work,'donor_inputs','experiments.fingerprint_bigraph','--work',work,'--stage','inputs')
    if not (work/'training_revision.json').exists():
        write_json(work/'training_revision.json',{'sources':source_signature(),'torch_version':str(torch.__version__),
            'training_revision':2,'neighbor_training':'soft to90% positive-cosine top10','auxiliary_dropout':0.25})
    validate_revision(work)
    jobs=[(label,'experiments.fingerprint_bigraph',['--work',work,'--stage','train','--rank-weight',weight,'--threads',4])
          for label,weight in [('graph_rank05',0.5),('graph_rank20',2.0)] if not (work/label/'complete.json').exists()]
    slots=2 if psutil.virtual_memory().available>=5.5*1024**3 else 1
    pool(work,'graph_training',jobs,slots=slots)
    run(work,'search','experiments.fingerprint_bigraph_search','--work',work,'--selection-only')
    final_test(work)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]),'failed',error=f'{type(error).__name__}: {error}')
        raise
