"""Fit training-prefix peer TF-IDF/SVD, transform historical and day13 windows."""
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from experiments.temporal_donor_identity.common.data import Window,read_window,contacts
from .train_identity import sha

ROOT=Path('PT_PROJECT_ML')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dim',type=int,default=64)
 a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 hosts=json.loads((ROOT/'fingerprint_aligned/hosts.json').read_text())
 vocabulary=json.loads((ROOT/'fingerprint_transfer/vocabulary.json').read_text());vindex=pd.Index(vocabulary)
 matrices=[];paths=[]
 for day in range(3,13):
  folder=ROOT/('fingerprint_aligned' if day<=9 else 'fingerprint_codebook_recent/history')
  path=folder/f'future{day:02}.npz';maskpath=folder/f'future_active{day:02}.npy'
  matrix=sparse.load_npz(path).tocsr().astype(np.float32);mask=np.load(maskpath)
  if matrix.shape!=(len(hosts),len(vocabulary)) or mask.shape!=(len(hosts),):raise ValueError('shape mismatch')
  matrix=matrix.multiply(mask[:,None]).tocsr();matrix.eliminate_zeros();matrices.append(matrix);paths.extend([path,maskpath])
 # Every observation is one host-window; fit only on active train observations.
 active=[np.flatnonzero(np.asarray(m.getnnz(axis=1))>0) for m in matrices]
 train=sparse.vstack([m[ix] for m,ix in zip(matrices,active)],format='csr')
 train.data=np.log1p(train.data)
 tfidf=TfidfTransformer().fit(train);weighted=normalize(tfidf.transform(train))
 svd=TruncatedSVD(n_components=a.dim,random_state=17,n_iter=5).fit(weighted)
 print('fit complete; explained variance',svd.explained_variance_ratio_.sum(),flush=True)
 for day,matrix in zip(range(3,13),matrices):
  matrix=matrix.copy();matrix.data=np.log1p(matrix.data)
  x=svd.transform(normalize(tfidf.transform(matrix))).astype(np.float32)
  np.save(a.output/f'day{day:02}.npy',x)
 # Development input, not used for fitting TF-IDF or SVD.
 dev=ROOT/'fingerprint_recent/day13';manifest=json.loads((dev/'manifest.json').read_text());start,end=manifest['config']['windows']['test']
 obs=read_window(Window('day13',13,start,end,str(dev/'flows/test.csv')),hosts,'runs/masked_multiday_recent_seed17/cache')
 rr=[];cc=[];vv=[]
 for i,row in enumerate(contacts(obs)):
  for peer,count in row.items():
   j=vindex.get_indexer([peer])[0]
   if j>=0:rr.append(i);cc.append(j);vv.append(count)
 raw=sparse.csr_matrix((np.asarray(vv,dtype=np.float32),(rr,cc)),shape=(len(hosts),len(vocabulary)))
 raw.data=np.log1p(raw.data)
 np.save(a.output/'day13.npy',svd.transform(normalize(tfidf.transform(raw))).astype(np.float32))
 joblib.dump({'tfidf':tfidf,'svd':svd,'vocabulary':vocabulary},a.output/'transform.joblib',compress=3)
 info={'train_days':list(range(3,13)),'development_day':13,'target_family':'undirected aggregate partner counts, no own ID; source vocabulary as previously frozen',
 'train_only_fit':True,'dim':a.dim,'source_window_start':start,'source_window_end':end,'training_hashes':{str(p):sha(p) for p in paths},
 'cohort_sha256':sha(ROOT/'fingerprint_aligned/hosts.json'),'vocabulary_sha256':sha(ROOT/'fingerprint_transfer/vocabulary.json'),
 'dev_input_sha256':sha(dev/'flows/test.csv'),'test14_16_read':False,'variance':float(svd.explained_variance_ratio_.sum())}
 (a.output/'manifest.json').write_text(json.dumps(info,indent=2));print('prepared',flush=True)
if __name__=='__main__':main()
