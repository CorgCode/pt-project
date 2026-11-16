"""Descriptive EDA on fixed source/development/benchmark windows; no model tuning."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans

ROOT=Path('PT_PROJECT_ML'); DATA=ROOT/'late_test';OUTPUT_COLUMNS=['src','dst','bytes','packets','ts','duration','proto','dst_port']


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()


def stats(x):
 x=np.asarray(x,dtype=float)
 return {'n':len(x),'zero_fraction':float(np.mean(x==0)),'quantiles':{str(q):float(np.quantile(x,q)) for q in [.1,.25,.5,.75,.9,.99]},'max':float(x.max())}


def summarize(path,hosts,start,end):
 n=len(hosts);out=np.zeros(n,dtype=np.int64);inn=out.copy();byt=out.copy();pkt=out.copy()
 total=missing=selfn=boundary=0;minute=np.zeros(60,dtype=np.int64)
 for f in pd.read_csv(path,usecols=OUTPUT_COLUMNS,dtype={'src':str,'dst':str,'proto':str,'dst_port':str},chunksize=250000):
  total+=len(f);missing+=int(f[OUTPUT_COLUMNS].isna().any(axis=1).sum());selfn+=int((f.src==f.dst).sum())
  boundary+=int(((f.ts<start)|(f.ts>=end)|((f.ts+f.duration)>=end)).sum())
  valid=(f.src!=f.dst)&f.src.notna()&f.dst.notna();f=f.loc[valid]
  pos=np.floor((f.ts.to_numpy()-start)*60/(end-start)).astype(int);pos=pos[(pos>=0)&(pos<60)];minute+=np.bincount(pos,minlength=60)
  src=hosts.get_indexer(f.src);dst=hosts.get_indexer(f.dst);s=src>=0;d=dst>=0
  out+=np.bincount(src[s],minlength=n);inn+=np.bincount(dst[d],minlength=n)
  vals=pd.to_numeric(f['bytes'],errors='coerce').fillna(0).to_numpy();v=pd.to_numeric(f['packets'],errors='coerce').fillna(0).to_numpy()
  byt+=np.bincount(src[s],weights=vals[s],minlength=n).astype(np.int64)
  pkt+=np.bincount(src[s],weights=v[s],minlength=n).astype(np.int64)
 return {'out':out,'in':inn,'bytes_out':byt,'packets_out':pkt,'rows':total,'missing_rows_any_selected_field':missing,'self_flows':selfn,'boundary_mismatches':boundary,'minute_flow_counts':minute.tolist()}


def neighbor_analysis(z,events,counts,seed=17):
 n=len(z);x=normalize(z.astype(np.float32));rng=np.random.default_rng(seed);near=[];random_indices=[];same=[];rand=[]
 count=np.log1p(events)
 for start in range(0,n,256):
  end=min(start+256,n);sim=x[start:end]@x.T;sim[np.arange(end-start),np.arange(start,end)]=-np.inf
  nn=np.argmax(sim,axis=1);near.extend(nn);same.extend(np.abs(count[start:end]-count[nn]))
  neg=rng.integers(0,n-1,size=end-start);ix=np.arange(start,end);neg+=neg>=ix;random_indices.extend(neg);rand.extend(np.abs(count[start:end]-count[neg]))
 nearest=np.array(near);norm=np.linalg.norm(z,axis=1);rho=spearmanr(norm,np.log1p(events)).statistic
 common=[];common_random=[]
 for i,j in enumerate(nearest):
  a=counts[i];b=counts[j];common.append(len(a&b)/len(a|b) if a|b else 0)
  b=counts[random_indices[i]];common_random.append(len(a&b)/len(a|b) if a|b else 0)
 return {'n':n,'metric':'source cosine top1 within active cohort','mean_abs_log1p_event_difference_nearest':float(np.mean(same)),
  'mean_abs_log1p_event_difference_random_one_draw':float(np.mean(rand)),'source_partner_jaccard_nearest_mean':float(np.mean(common)),'source_partner_jaccard_random_mean':float(np.mean(common_random)),
  'spearman_source_norm_vs_target_activity':float(rho) if np.isfinite(rho) else None}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 hosts=pd.Index(pd.read_csv(DATA/'processed/cohort.csv',dtype=str).host_id)
 if hosts.has_duplicates:raise ValueError('duplicate source hosts')
 manifest=json.loads((DATA/'processed/manifest.json').read_text());devman=json.loads((ROOT/'fingerprint_recent/day13/manifest.json').read_text())
 source=pd.read_csv(DATA/'vectors/multiday_codebook/embeddings.csv',dtype={'host_id':str}).set_index('host_id').loc[hosts]
 if source.isna().any().any():raise ValueError('missing source embeddings')
 names=['train','day13','test','test_day15','test_day16'];result={'scope':'observational EDA; day13 and test14–16 previously studied; not validation or blind test','cohort_size':len(hosts),'windows':{},'hashes':{},'limitations':['processed hour files, not full-day statistics','missing fields count restricted to eight selected fields','exact duplicate flow detection not performed','relationships are descriptive, not role labels']}
 runs={}
 for name in names:
  if name=='day13':path=ROOT/'fingerprint_recent/day13/flows/test.csv';start,end=devman['config']['windows']['test'];reference=ROOT/'fingerprint_recent/reference_validation/validation';late=None
  else:
   path=DATA/'processed/flows'/f'{name}.csv';window=manifest['windows'][name];start,end=window['start'],window['end'];reference=DATA/'reference'/('test' if name=='train' else name);late=DATA/'vectors/multiday_codebook'/f'{name}.csv' if name!='train' else None
  print('processing',name,flush=True);v=summarize(path,hosts,start,end);result['hashes'][str(path)]=sha(path)
  events=v['out']+v['in'];active=(events>=300);reference_hosts=pd.read_csv(reference/'eval_hosts.csv',dtype=str).host_id if name!='train' else None
  if reference_hosts is not None:
   mask=hosts.isin(reference_hosts);discrepancy=int(np.count_nonzero(active!=mask))
  else:mask=active;discrepancy=None
  result['windows'][name]={'start':start,'end':end,'rows':v['rows'],'selected_field_missing_rows':v['missing_rows_any_selected_field'],'self_flows':v['self_flows'],'boundary_mismatches':v['boundary_mismatches'],
   'active_300_among_source_cohort':int(active.sum()),'active_reference_mask_disagreements':discrepancy,
   'events_all_cohort':stats(events),'events_active':stats(events[mask]),
   'top1pct_event_share_all_cohort':float(np.sort(events)[-max(1,int(np.ceil(len(hosts)*.01))):].sum()/max(events.sum(),1)),
   'outgoing_share_events':float(v['out'].sum()/max(events.sum(),1)),
   'minute_flow_counts':v['minute_flow_counts']}
  runs[name]={'events':events,'mask':mask,'out':v['out'],'in':v['in']}
  if name=='train':continue
  partners=pd.read_csv(reference/'partners.csv',dtype={'host_id':str,'partner':str,'window':str})
  partners=partners[partners.host_id!=partners.partner]
  target=partners[partners.window=='target'];sourcep=partners[partners.window=='source']
  ti=target.groupby('host_id').partner.nunique().reindex(hosts[mask],fill_value=0);result['windows'][name]['unique_partners_active']=stats(ti)
  old=set(zip(sourcep.host_id,sourcep.partner));targetkeys=list(zip(target.host_id,target.partner));fresh=np.fromiter((key not in old for key in targetkeys),bool,len(targetkeys))
  result['windows'][name]['new_partner_pairs_fraction']=float(fresh.mean());result['windows'][name]['new_partner_flow_count_fraction']=float(target['count'].to_numpy()[fresh].sum()/max(target['count'].sum(),1))
  active_ids=hosts[mask];sets={h:set(g.partner) for h,g in sourcep.groupby('host_id',sort=False)}
  result['windows'][name]['neighbors']=neighbor_analysis(source.loc[active_ids].to_numpy(float),events[mask],[sets.get(h,set()) for h in active_ids])
  if late:
   y=pd.read_csv(late,dtype={'host_id':str}).set_index('host_id').loc[active_ids].to_numpy(float);z=source.loc[active_ids].to_numpy(float)
   zn=normalize(z);yn=normalize(y);matched=(zn*yn).sum(1);rng=np.random.default_rng(17);ix=np.arange(len(z));neg=rng.integers(0,len(z)-1,len(z));neg+=neg>=ix
   result['windows'][name]['direct_cosine_unaligned']={'same_host':stats(matched),'random_different_host':stats((zn*yn[neg]).sum(1)),'note':'direct shared-coordinate cosine; not official Procrustes Identity'}
  print('done',name,'active',int(mask.sum()),flush=True)
 # Stability of source cohort membership over the processed windows, not entire days.
 result['observed_window_coverage']=stats(np.stack([runs[x]['events']>0 for x in names]).sum(0))
 result['active_300_window_coverage']=stats(np.stack([runs[x]['events']>=300 for x in names]).sum(0))
 pd.DataFrame({'host_id':hosts,**{name+'_events':runs[name]['events'] for name in names},**{name+'_out':runs[name]['out'] for name in names},**{name+'_in':runs[name]['in'] for name in names}}).to_csv(a.output/'host_activity.csv',index=False)
 x=normalize(source.to_numpy(dtype=np.float32));km=MiniBatchKMeans(n_clusters=8,random_state=17,n_init=3,batch_size=512).fit(x)
 result['source_clustering_k8']={'method':'exploratory MiniBatchKMeans on normalized source vectors; k fixed at 8, seed17','cluster_sizes':np.bincount(km.labels_,minlength=8).tolist(),'not_true_role_labels':True}
 (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('completed',a.output,flush=True)
if __name__=='__main__':main()
