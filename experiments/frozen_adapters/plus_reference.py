"""Matching local references from prepared flows, with an explicit persistence ceiling."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.frozen_neighbors import save_frozen_neighbors, evaluate_frozen_neighbors
from evaluation.future_links import build_edges
from evaluation.structural_pairs import build_judgments
from .codebook import behavior
from .data import contacts, signature
from .run_neighbors import partner_frame


def build(source_obs, target_obs, hosts, output, cache, source_behavior=None):
    output=Path(output)
    inputs={'source':asdict(source_obs.window),'target':asdict(target_obs.window),
            'cohort_sha256':hashlib.sha256(json.dumps(hosts).encode()).hexdigest(),
            'recipe':'binary graph edges/pairs + exact count partners + project behavior, seed0, pairs10'}
    inputs['flow_files']={}
    for label,observation in [('source',source_obs),('target',target_obs)]:
        stat=Path(observation.window.flows).stat()
        inputs['flow_files'][label]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    manifest_file=output/'manifest.json'
    if manifest_file.exists():
        old=json.loads(manifest_file.read_text())
        if old.get('inputs')!=inputs: raise ValueError('reference inputs changed')
        if any(signature(output/name)!=sha for name,sha in old['files_sha256'].items()):
            raise ValueError('reference CSV changed')
        return old
    output.mkdir(parents=True,exist_ok=True)
    population=np.flatnonzero(target_obs.events>=300)
    active=[hosts[i] for i in population]
    pd.DataFrame({'host_id':active}).to_csv(output/'eval_hosts.csv',index=False)
    source_rows,target_rows=contacts(source_obs),contacts(target_obs)
    partner_frame(source_rows,target_rows,hosts,population).to_csv(output/'partners.csv',index=False)

    def graphrows(rows, window):
        # Both graph judgments depend on contact SETS, not multiplicity. partners.csv above keeps counts.
        f=pd.DataFrame([(hosts[i],peer) for i in population for peer in rows[i]],columns=['src','dst'])
        f['ts']=pd.Timestamp(window.start,unit='s')
        # Declared half-open prepared window, encoded like the canonical edge files.
        f.loc[f.index[-1],'ts']=pd.Timestamp(window.end-1,unit='s')
        return f
    first=graphrows(source_rows,source_obs.window)
    later=graphrows(target_rows,target_obs.window)
    build_edges(first,later,active,seed=0,split='test').to_csv(output/'edges.csv',index=False)
    pairs,triplets,judge=build_judgments(later,active,seed=0,pairs_per_anchor=10)
    pairs.to_csv(output/'pairs.csv',index=False)
    triplets.to_csv(output/'triplets.csv',index=False)
    features=behavior(target_obs.window,active,cache)[0]
    ref=pd.DataFrame(features,index=pd.Index(active,name='host_id')).add_prefix('ref_')
    ref.to_csv(output/'features.csv')
    save_frozen_neighbors(ref,output/'neighbors.csv',reference_window=target_obs.window.name,k_max=20,split='test')
    if source_behavior is None: source_behavior=behavior(source_obs.window,hosts,cache)[0]
    persist=pd.DataFrame(source_behavior[population],index=active).add_prefix('emb_')
    ceiling=float(evaluate_frozen_neighbors(persist,output/'neighbors.csv',ks=(10,),split='test').loc[(10,'consistency'),'mean'])
    manifest={'inputs':inputs,'split':'test','target_window':target_obs.window.name,
              'kind':'heldout_temporal_behavior_proxy','ground_truth':False,'data_kind':'lanl_declared',
              'n_cohort':len(hosts),'n_evaluable':len(active),'n_hosts':len(active),
              'chance_consistency_at_k':{str(k):k/(len(active)-1) for k in (5,10,20)},
              'judgments':judge,'neighbor_ceiling':ceiling,
              'windows':{'train':asdict(source_obs.window),'test':asdict(target_obs.window)},
              'score_scope':'local diagnostic; not canonical days14-16',
              'files_sha256':{p.name:signature(p) for p in sorted(output.glob('*.csv'))}}
    manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
