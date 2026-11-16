"""Read-only, post-hoc paired Identity diagnostics for an existing result."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from .model import TemporalDonorIdentity,ModelConfig
from scipy.linalg import orthogonal_procrustes


def evaluate(a,b):
    n=len(a);order=np.random.default_rng(0).permutation(n);count=int(round(n*.5))
    ev,cal=np.sort(order[:count]),np.sort(order[count:])
    ma,mb=a[cal].mean(0),b[cal].mean(0)
    R,_=orthogonal_procrustes(b[cal]-mb,a[cal]-ma)
    aa=a[ev]-ma;bb=(b[ev]-mb)@R
    aa/=np.maximum(np.linalg.norm(aa,axis=1,keepdims=True),1e-12)
    bb/=np.maximum(np.linalg.norm(bb,axis=1,keepdims=True),1e-12)
    sim=aa@bb.T;true=np.diag(sim).copy();sim[np.diag_indices(len(sim))]=-np.inf
    # Match the official absolute tolerance and partial credit on ties.
    greater=(sim>true[:,None]).sum(1)
    ties=np.isclose(sim,true[:,None],rtol=0,atol=1e-12).sum(1)
    credit=np.clip(1-greater,0,ties+1)/(ties+1)
    margin=true-np.max(sim,axis=1)
    return credit,margin


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    out=args.run/'paired_identity_exact_checkpoint.json'
    if out.exists():raise FileExistsError(out)
    ck=torch.load(args.run/'model.pt',map_location='cpu',weights_only=True)
    model=TemporalDonorIdentity(ModelConfig(**ck['model_config']));model.load_state_dict(ck['state_dict']);model.eval()
    root=Path('PT_PROJECT_ML');source=pd.read_csv(root/'late_test/vectors/multiday_codebook/embeddings.csv').set_index('host_id')
    refs=root/'late_test/reference'; results={}
    for w in ['validation','test','test_day15','test_day16']:
        ref=(root/'fingerprint_recent/reference_validation/validation' if w=='validation' else refs/w)
        old=(root/'fingerprint_codebook_recent/candidates/codebook1_e400/later.csv' if w=='validation' else root/f'late_test/vectors/multiday_codebook/{w}.csv')
        ids=pd.read_csv(ref/'eval_hosts.csv',dtype=str).host_id.sort_values().tolist()
        a=source.loc[ids].to_numpy(float)
        b=pd.read_csv(old,dtype={'host_id':str}).set_index('host_id').loc[ids].to_numpy(float)
        x=torch.from_numpy(b.astype(np.float32))
        with torch.no_grad():c=model.late_vectors(x,torch.zeros(len(x),model.config.context_dim)).numpy().astype(float)
        old_credit,old_margin=evaluate(a,b);new_credit,new_margin=evaluate(a,c)
        improved=(new_credit>old_credit);worsened=(new_credit<old_credit)
        full_fixed=(old_credit==0)&(new_credit==1);full_broken=(old_credit==1)&(new_credit==0)
        delta=new_margin-old_margin
        def mean(mask):return float(delta[mask].mean()) if mask.any() else None
        results[w]={'n':len(old_credit),'baseline':float(old_credit.mean()),'candidate':float(new_credit.mean()),
                    'credit_increased':int(improved.sum()),'credit_decreased':int(worsened.sum()),
                    'zero_to_one':int(full_fixed.sum()),'one_to_zero':int(full_broken.sum()),
                    'other_credit_changes':int((improved|worsened).sum()-full_fixed.sum()-full_broken.sum()),
                    'margin_change_mean_all':mean(np.ones(len(delta),dtype=bool)),
                    'margin_change_mean_fixed':mean(full_fixed),'margin_change_mean_broken':mean(full_broken)}
        print(w,results[w],flush=True)
    out.write_text(json.dumps({'scope':'post-hoc, no selection; existing frozen model','windows':results},indent=2))
if __name__=='__main__':main()
