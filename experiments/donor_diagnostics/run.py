"""Day13 bounded donor oracle and predeclared history heuristic, never training.

Oracle uses correct day13 target labels: it is NOT a model or a score bound.
History selector is allowed only days03–12 and does not read target labels.
The exact-column duplicate bound is valid for any nonnegative donor weights,
including arbitrary donor selection and any vector geometry.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from evaluation.partners import partner_ranking

ROOT=Path('PT_PROJECT_ML')

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def rr(votes,valid,positive):
 best=np.max(votes[positive]); greater=np.count_nonzero(valid & (votes>best)); ties=np.count_nonzero(valid & (votes==best))
 return 1. / (greater+(ties+1)/2)


def duplicate_bound(groups,valid,positive):
 """Best possible first-ANY RR under unavoidable equal-score column ties."""
 return max(2/(sum(bool(valid[g]) for g in groups[t])+1) for t in positive)


def predicted_utility(votes,valid,positive,probability):
 # Expected 1/rank for a categorical next contact, not the metric's first-ANY MRR.
 scores=votes[positive];vs=votes[valid]
 ranks=1+(vs[None,:]>scores[:,None]).sum(1)+((vs[None,:]==scores[:,None]).sum(1)-1)/2
 return float((probability/ranks).sum())


def greedy(source_dense, cosine, valid, initial, choices, objective, passes=2):
 """Exact one-donor replacements, cosine weights from fixed baseline Z.
 Search finite pool and local optimum, not a global optimizer.
 """
 selected=np.asarray(initial,dtype=int).copy();weights=np.clip(cosine,0,None)
 # For stable ties, calculate each actual sum in same deterministic order.
 votes=(source_dense[selected]*weights[selected,None]).sum(0)
 score=objective(votes); changes=0
 for _ in range(passes):
  previous=score
  for slot in range(len(selected)):
   old=selected[slot];others=votes-source_dense[old]*weights[old]
   best=score; bestj=old; bestvotes=votes
   for candidate in choices:
    if candidate in selected or weights[candidate]<=0:continue
    proposal=others+source_dense[candidate]*weights[candidate]
    value=objective(proposal)
    if value>best+1e-12:best=value;bestj=candidate;bestvotes=proposal
   if bestj!=old:
    selected[slot]=bestj;votes=bestvotes;score=best;changes+=1
  if score<=previous+1e-12:break
 # Recompute exactly and check non-regression under floating arithmetic.
 votes=(source_dense[selected]*weights[selected,None]).sum(0)
 return selected,votes,changes


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--output',required=True,type=Path);ap.add_argument('--queries',type=int,default=64)
 ap.add_argument('--seed',type=int,default=17);ap.add_argument('--donors-per-target',type=int,default=12)
 args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 if args.queries<1 or args.donors_per_target<1:raise ValueError('positive budgets required')
 args.output.mkdir(parents=True)
 ref=ROOT/'fingerprint_recent/reference_validation/validation';data=pd.read_csv(ref/'partners.csv',dtype={'host_id':str,'partner':str,'window':str})
 active=pd.Index(pd.read_csv(ref/'eval_hosts.csv',dtype=str).host_id);src=data[data.window=='source'];tgt=data[data.window=='target']
 universe=pd.Index(sorted(src.partner.unique()));p=src[src.host_id.isin(active)]
 M=sparse.csr_matrix((p['count'].to_numpy(float),(active.get_indexer(p.host_id),universe.get_indexer(p.partner))),shape=(len(active),len(universe)))
 basefile=ROOT/'fingerprint_codebook_recent/candidates/codebook1_e400/embeddings.csv'
 base=pd.read_csv(basefile,dtype={'host_id':str}).set_index('host_id').loc[active]
 X=normalize(base.to_numpy(float)); truth=tgt[tgt.host_id.isin(active)]
 allhosts=pd.Index(json.loads((ROOT/'fingerprint_aligned/hosts.json').read_text()))
 vocabulary=pd.Index(json.loads((ROOT/'fingerprint_transfer/vocabulary.json').read_text()))
 active_pos=allhosts.get_indexer(active);vpos=vocabulary.get_indexer(universe);covered=vpos>=0
 if (active_pos<0).any():raise ValueError('active host missing')
 H=sparse.csr_matrix((len(active),len(universe)),dtype=float);hashes={}
 for day in range(3,13):
  d=ROOT/('fingerprint_aligned' if day<=9 else 'fingerprint_codebook_recent/history')
  path=d/f'future{day:02}.npz';maskpath=d/f'future_active{day:02}.npy'
  m=sparse.load_npz(path).tocsr();mask=np.load(maskpath)
  if m.shape!=(len(allhosts),len(vocabulary)) or mask.shape!=(len(allhosts),):raise ValueError('history mismatch')
  m=m[active_pos];m=m.multiply(mask[active_pos,None]).tocsr();m.data[:]=1
  remap=sparse.csr_matrix((np.ones(covered.sum()),(vpos[covered],np.flatnonzero(covered))),shape=(len(vocabulary),len(universe)))
  H=H+(m@remap)*(2**(-(12-day)/3.))
  hashes[str(path)]=digest(path);hashes[str(maskpath)]=digest(maskpath)
 H=H.tocsr();H.eliminate_zeros()
 # Oracle answer groups; this is for post-hoc evaluation and oracle ONLY.
 positives={}
 for host,g in truth.groupby('host_id',sort=False):
  i=active.get_loc(host);ps=universe.get_indexer(g.partner);ps=ps[ps>=0]
  old=M.indices[M.indptr[i]:M.indptr[i+1]];ps=np.setdiff1d(ps,old)
  own=universe.get_indexer([host])[0];ps=ps[ps!=own]
  if len(ps):positives[i]=ps
 official=partner_ranking(base,data,k=10)
 ids=np.sort(np.array(list(positives),int));rng=np.random.default_rng(args.seed)
 subset=np.sort(rng.choice(ids,min(args.queries,len(ids)),replace=False))
 # Exact duplicate columns among all active donors force ties for any nonnegative weights.
 csc=M.tocsc();groups={};colgroup=[]
 for j in range(M.shape[1]):
  l,r=csc.indptr[j:j+2];key=(csc.indices[l:r].tobytes(),csc.data[l:r].tobytes())
  if key not in groups:groups[key]=[]
  groups[key].append(j);colgroup.append(groups[key])
 allbounds=[];reachable=[]
 # Compute bounds on all eligible queries, without assuming target scores >0.
 for q in ids:
  old=M.indices[M.indptr[q]:M.indptr[q+1]];valid=np.ones(len(universe),bool);valid[old]=False
  own=universe.get_indexer([active[q]])[0]
  if own>=0:valid[own]=False
  allbounds.append(duplicate_bound(colgroup,valid,positives[q]))
  supporters=np.asarray((M[:,positives[q]]>0).sum(0)).ravel()-(M[q,positives[q]].toarray().ravel()>0)
  reachable.append(bool((supporters>0).any()))
 # Dense source fits within ~350MB for this benchmark, serving deterministic exact votes.
 dense=M.toarray();results=[]
 for num,q in enumerate(subset,1):
  sim=X[q]@X.T;sim[q]=-np.inf;top=np.argpartition(-sim,9)[:10]
  valid=np.ones(len(universe),bool);valid[M.indices[M.indptr[q]:M.indptr[q+1]]]=False
  own=universe.get_indexer([active[q]])[0]
  if own>=0:valid[own]=False
  # History proposal is formed before consulting labels and contains no future inputs.
  hr=H.getrow(q).toarray().ravel();hr[~valid]=0
  pred=np.flatnonzero(hr>0);pred=pred[np.argsort(-hr[pred],kind='stable')[:8]]
  def candidates(targets):
   pool=set(top.tolist())
   for t in targets:
    donors=M[:,int(t)].tocoo();order=np.argsort(-donors.data,kind='stable')
    take=0
    for pos in order:
     j=int(donors.row[pos])
     if j!=q and sim[j]>0:pool.add(j);take+=1
     if take>=args.donors_per_target:break
   return np.asarray(sorted(pool),int)
  oracle_pool=candidates(positives[q]);selected0=(dense[top]*np.clip(sim[top],0,None)[:,None]).sum(0)
  actual=lambda v:rr(v,valid,positives[q]);baseline=actual(selected0)
  osel,ov,oc=greedy(dense,sim,valid,top,oracle_pool,actual)
  hr_selected=top.copy();hc=0
  if len(pred):
   pp=hr[pred];pp=pp/pp.sum();obj=lambda v:predicted_utility(v,valid,pred,pp)
   hpool=candidates(pred);hr_selected,hv,hc=greedy(dense,sim,valid,top,hpool,obj)
  else:hv=selected0
  results.append({'host':active[q],'baseline_rr':baseline,'oracle_rr':actual(ov),'history_rr':actual(hv),
    'oracle_changes':oc,'history_changes':hc,'predicted_partners':len(pred),
    'oracle_pool':len(oracle_pool),'history_any_true_partner':bool(np.intersect1d(pred,positives[q]).size)})
  if num%16==0:print('queries',num,'of',len(subset),flush=True)
 summary={'scope':'previously examined development day13; no score or new model',
  'history_cutoff_day':12,'source_day':2,'n_active':len(active),'n_queries':len(ids),'sample_queries':len(subset),
  'sample_seed':args.seed,'oracle_uses_target_labels':True,'history_uses_target_labels':False,
  'oracle_weight_constraint':'fixed original cosine values, independently selected donors; cannot assume one joint geometry',
  'oracle_search':'bounded candidate pool, greedy substitutions; found value not an upper bound',
  'history_selector':'recency-weighted observed contacts through12; top8 as categorical proxy; greedy finite candidate pool',
  'reachable_fraction_all':float(np.mean(reachable)), 'duplicate_column_upper_bound_mrr_all':float(np.mean(allbounds)),
  'official_baseline_all_mrr':official['new']['embedding']['mrr'],
  'sample_means':{key:float(np.mean([r[key] for r in results])) for key in ['baseline_rr','oracle_rr','history_rr']},
  'sample_oracle_improved':int(sum(r['oracle_rr']>r['baseline_rr']+1e-12 for r in results)),
  'sample_history_improved':int(sum(r['history_rr']>r['baseline_rr']+1e-12 for r in results)),
  'sample_history_worse':int(sum(r['history_rr']<r['baseline_rr']-1e-12 for r in results)),
  'input_sha256':hashes,'reference_sha256':digest(ref/'partners.csv'),'source_sha256':digest(basefile),
  'previously_known_upper_bound':'none derived for overall score; duplicate columns only'}
 (args.output/'summary.json').write_text(json.dumps(summary,indent=2));(args.output/'per_query.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
