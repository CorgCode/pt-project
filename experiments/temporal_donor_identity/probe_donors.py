"""Development-only hard-voting screening of historical contact blocks.

This diagnostic does not train a late adapter or calculate full score. History
02–12 and the source codebook are fixed before the day13 development window.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .donor_grid import evaluate_grid

ROOT=Path('PT_PROJECT_ML')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dim',type=int,default=32)
 a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 hosts=json.loads((ROOT/'fingerprint_aligned/hosts.json').read_text())
 base=pd.read_csv(ROOT/'fingerprint_codebook_recent/candidates/codebook1_e400/embeddings.csv',dtype={'host_id':str}).set_index('host_id').loc[hosts].to_numpy(float)
 history=ROOT/'fingerprint_codebook_recent/history';old=ROOT/'fingerprint_aligned'
 days=[];hashes={}
 for day in range(3,13):
  path=(old if day<=9 else history)/f'future{day:02}.npz';maskpath=(old if day<=9 else history)/f'future_active{day:02}.npy'
  m=sparse.load_npz(path).tocsr();mask=np.load(maskpath)
  if m.shape[0]!=len(hosts) or mask.shape!=(len(hosts),):raise ValueError('cohort alignment failure')
  m=m.multiply(mask[:,None]).tocsr();m.eliminate_zeros();m.data[:]=1
  days.append(m);hashes[str(path)]=sha(path);hashes[str(maskpath)]=sha(maskpath)
  print('history',day,'nonzeros',m.nnz,flush=True)
 # All input observations share 12:30 start: periodic is mathematically
 # identical to unweighted day-presence; avoid claiming a separate signal.
 presence=sum(days[1:],days[0].copy()).tocsr()
 recent=sum((m*(2**(-(12-day)/3)) for m,day in zip(days,range(3,13))),sparse.csr_matrix(presence.shape))
 # Reduce using training-prefix hosts only. Matrices are normalized per host.
 profiles=[]
 for label,x in [('days',presence),('recent',recent)]:
  x=normalize(x,norm='l2');svd=TruncatedSVD(n_components=a.dim,random_state=17,n_iter=5)
  z=svd.fit_transform(x).astype(np.float32);np.save(a.output/f'{label}.npy',z)
  profiles.append(z);print(label,'variance',svd.explained_variance_ratio_.sum(),flush=True)
 profiles.append(profiles[0].copy())
 ref=ROOT/'fingerprint_recent/reference_validation/validation'
 partners=pd.read_csv(ref/'partners.csv',dtype={'host_id':str,'partner':str,'window':str})
 active=pd.read_csv(ref/'eval_hosts.csv',dtype=str).host_id.tolist()
 grid=[[1,0,0,0],[.95,.05,0,0],[.90,.10,0,0],[.80,.20,0,0],[.95,0,.05,0],[.90,0,.10,0],[.80,0,.20,0],[.90,.05,.05,0]]
 (a.output/'protocol.json').write_text(json.dumps({'train_days':list(range(3,13)),'development_day':13,'source_day':2,
  'projection':'independent TruncatedSVD on normalized training-only profile matrices','dim_per_block':a.dim,'seed':17,
  'periodic_not_separate':'all historical target windows have same hour', 'weights':grid,
  'identity_and_total_score':'not measured; new late adapter not trained', 'test14_16_read':False,
  'input_sha256':hashes,'reference_partners_sha256':sha(ref/'partners.csv'),'source_sha256':sha(ROOT/'fingerprint_codebook_recent/candidates/codebook1_e400/embeddings.csv')},indent=2))
 rows=evaluate_grid(base,profiles,hosts,partners,active,grid)
 (a.output/'development.json').write_text(json.dumps(rows,indent=2))
 for row in rows:print(row,flush=True)
 print('diagnostic only; no score or model selection',flush=True)
if __name__=='__main__':main()
