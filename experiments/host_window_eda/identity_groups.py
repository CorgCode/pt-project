"""Post-hoc baseline identity by activity; same seed and Procrustes as evaluator."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.linalg import orthogonal_procrustes
from evaluation.identity import identity_retrieval


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--eda',type=Path,required=True);a=ap.parse_args()
 root=Path('PT_PROJECT_ML/late_test');data=pd.read_csv(a.eda/'host_activity.csv',dtype={'host_id':str}).set_index('host_id')
 source=pd.read_csv(root/'vectors/multiday_codebook/embeddings.csv',dtype={'host_id':str}).set_index('host_id');summary={}
 for w in ['test','test_day15','test_day16']:
  hosts=pd.read_csv(root/'reference'/w/'eval_hosts.csv',dtype=str).host_id.sort_values().tolist()
  a0=source.loc[hosts];b0=pd.read_csv(root/'vectors/multiday_codebook'/f'{w}.csv',dtype={'host_id':str}).set_index('host_id').loc[hosts]
  x=a0.to_numpy(float);y=b0.to_numpy(float);n=len(hosts);perm=np.random.default_rng(0).permutation(n);ev=np.sort(perm[:int(round(n*.5))]);cal=np.sort(perm[int(round(n*.5)):])
  ma=x[cal].mean(0);mb=y[cal].mean(0);R,_=orthogonal_procrustes(y[cal]-mb,x[cal]-ma);u=x[ev]-ma;v=(y[ev]-mb)@R
  u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
  sim=u@v.T;true=np.diag(sim)[:,None];greater=(sim>true).sum(1);ties=np.isclose(sim,true,rtol=0,atol=1e-12).sum(1)-1;credit=np.clip(1-greater,0,ties+1)/(ties+1)
  official=identity_retrieval(a0,b0,alignment='procrustes',seed=0,n_boot=0)['top1']
  if abs(credit.mean()-official)>1e-12:raise ValueError('official identity mismatch')
  evhosts=np.array(hosts)[ev];events=data.loc[evhosts,w+'_events'].to_numpy();bins=[]
  for label,mask in [('300–499',(events>=300)&(events<500)),('500–999',(events>=500)&(events<1000)),('1000–4999',(events>=1000)&(events<5000)),('5000+',(events>=5000))]:
   bins.append({'events':label,'n':int(mask.sum()),'identity_top1':float(credit[mask].mean()) if mask.any() else None})
  summary[w]={'official_top1':official,'groups':bins,'note':'descriptive groups after shared Procrustes, not independent within group'}
  print(w,summary[w],flush=True)
 (a.eda/'identity_by_activity.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
