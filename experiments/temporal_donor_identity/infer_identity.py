"""Apply a trained residual to precomputed original-codebook late vectors."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from .model import ModelConfig,TemporalDonorIdentity


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--late',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck.get('experiment',{}).get('note') is None:
        raise ValueError('expected checkpoint from train_identity')
    if ck.get('experiment',{}).get('feature_family', 'baseline_vector_only') != 'baseline_vector_only':
        raise ValueError('this checkpoint requires contact features; this CLI supports only baseline-vector models')
    m=TemporalDonorIdentity(ModelConfig(**ck['model_config']));m.load_state_dict(ck['state_dict']);m.eval()
    data=pd.read_csv(a.late,dtype={'host_id':str}).set_index('host_id')
    expected=[f'emb_{i}' for i in range(m.config.base_dim)]
    if data.index.has_duplicates or data.index.isna().any() or list(data.columns)!=expected:
        raise ValueError('bad host ids or columns')
    x=torch.from_numpy(data.to_numpy(dtype=np.float32))
    with torch.no_grad():y=m.late_vectors(x,torch.zeros(len(x),m.config.context_dim)).numpy()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(y.astype('float64'),index=data.index,columns=data.columns).to_csv(a.output)

if __name__=='__main__':main()
