"""Recent-history pilot: final vectors, raw-count votes and episodic Procrustes.
Outputs are local diagnostics, never an official scoreboard submission.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from .common.data import Window, read_window, contacts, signature
from .common.evaluate import geometry, with_identity


def soft_topk(sim, k=10, temperature=.03):
    # Implicit derivative of the threshold enforces sum(weights)=k.
    with torch.no_grad():
        lo=sim.min(1,keepdim=True).values-2
        hi=sim.max(1,keepdim=True).values+2
        for _ in range(40):
            mid=(lo+hi)/2
            mass=torch.sigmoid((sim-mid)/temperature).sum(1,keepdim=True)
            lo=torch.where(mass>k,mid,lo);hi=torch.where(mass>k,hi,mid)
        threshold=(lo+hi)/2
        p=torch.sigmoid((sim-threshold)/temperature)
        derivative=p*(1-p)
    implicit=(sim*derivative).sum(1,keepdim=True)/derivative.sum(1,keepdim=True).clamp_min(1e-12)
    threshold=threshold+implicit-implicit.detach()
    return torch.sigmoid((sim-threshold)/temperature)


def aligned_logits(a,b):
    n=len(a);half=n//2
    aa=a[:half];bb=b[:half];ma=aa.mean(0);mb=bb.mean(0)
    # Fixed distinct diagonal jitter regularizes repeated singular values.
    cross=(bb-mb).T@(aa-ma)
    cross=cross+torch.diag(torch.linspace(1e-5,2e-5,cross.shape[0]))
    u,_,vh=torch.linalg.svd(cross,full_matrices=False)
    late=F.normalize((b[half:]-mb)@(u@vh),dim=1)
    src=F.normalize(a[half:]-ma,dim=1)
    return src@late.T/.07


class Model(nn.Module):
    def __init__(self,initial,fit_input):
        super().__init__();self.source=nn.Parameter(initial.clone())
        self.adapter=nn.Linear(fit_input.shape[1],initial.shape[1])
        self.link_head=nn.Linear(2*initial.shape[1],1)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.linalg.lstsq(initial.new_tensor(fit_input), initial).solution.T)
            self.adapter.bias.zero_()
    def vectors(self):return F.normalize(self.source,dim=1)
    def late(self,x):return F.normalize(self.adapter(x),dim=1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=160)
    ap.add_argument('--cache',type=Path);ap.add_argument('--dim',type=int,default=64);ap.add_argument('--seed',type=int,default=17)
    a=ap.parse_args()
    if a.steps<1 or not 2<=a.dim<=128:raise ValueError('steps>=1 and 2<=dim<=128 required for this pilot')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed);root=Path('PT_PROJECT_ML')
    hosts=json.loads((root/'fingerprint_aligned/hosts.json').read_text());idx=pd.Index(hosts)
    anchor=None
    # Stateless raw feature encoder; newest train windows11/12, development13.
    train_paths=[root/'fingerprint_memory/reference_test/test',root/'fingerprint_expanded/reference_test/test']
    dev=root/'fingerprint_recent/reference_validation/validation'
    probe=root/'late_test/reference/validation'
    cache=a.cache or a.output/'cache';cache.mkdir(exist_ok=True)
    source_path=Path('runs/temporal_stable_hybrid_v2_exact/processed/flows/train.csv')
    obs=read_window(Window('source02',2,126000,129600,str(source_path)),hosts,'runs/frozen_identity_local/cache')
    rows=contacts(obs);vocab=pd.Index(sorted({p for row in rows for p in row}));rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for p,count in row.items():rr.append(i);cc.append(vocab.get_loc(p));vv.append(count)
    matrix=sp.csr_matrix((vv,(rr,cc)),shape=(len(hosts),len(vocab)),dtype='float32')
    tensor=torch.sparse_coo_tensor(np.vstack(matrix.tocoo().nonzero()),matrix.tocoo().data,matrix.shape).coalesce()
    # Source-only universe, exact old-contact and self exclusions.
    valid=np.ones(matrix.shape,bool);r,c=matrix.nonzero();valid[r,c]=False
    own=vocab.get_indexer(idx);valid[np.flatnonzero(own>=0),own[own>=0]]=False
    valid=torch.from_numpy(valid)
    hasher=FeatureHasher(n_features=1024,input_type='dict',alternate_sign=False)
    def features(o):
        raw=hasher.transform([{k:np.log1p(v) for k,v in row.items()} for row in o.counts]).toarray().astype('float32')
        return raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
    source_features=features(obs)
    late_observations=[]
    for day,path in [(11,root/'fingerprint_memory/day11'),(12,root/'fingerprint_expanded/day12'),(13,root/'fingerprint_recent/day13')]:
        manifest=json.loads((path/'manifest.json').read_text())
        start,end=manifest['config']['windows']['test']
        print('extracting',day,start,end,flush=True)
        late_observations.append(read_window(Window(f'day{day}',day,start,end,str(path/'flows/test.csv')),hosts,cache))
    # Train only on11/12, newest allowed days. Source/history features never include13.
    anchor=(source_features+features(late_observations[0])+features(late_observations[1]))/3
    pca=PCA(n_components=a.dim,random_state=a.seed).fit(anchor)
    initial=torch.from_numpy(pca.transform(anchor).astype('float32'));initial=F.normalize(initial,dim=1)
    model=Model(initial,anchor);optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    episodes=[];gate=[]
    for day,ref in zip([11,12],train_paths):
        pp=pd.read_csv(ref/'partners.csv');pp=pp[pp.window=='target']
        ri=idx.get_indexer(pp.host_id);ci=vocab.get_indexer(pp.partner);ok=(ri>=0)&(ci>=0)
        target=sp.csr_matrix((np.ones(ok.sum()),(ri[ok],ci[ok])),shape=matrix.shape)>0
        target=target.multiply(valid.numpy()).tocsr()
        active=idx.get_indexer(pd.read_csv(ref/'eval_hosts.csv').host_id)
        eligible=active[np.asarray(target.sum(1)).ravel()[active]>0]
        coverage=np.array(matrix[active].sum(0)).ravel()>0
        # Loose rigorous upper bound: uncovered positives get at most their zero-vote midrank.
        bounds=[]
        for q in eligible:
            positive=target[q].indices
            if coverage[positive].any():bounds.append(1.)
            else:
                zeros=int((valid[q].numpy()&~coverage).sum())
                bounds.append(1/(1+(zeros-1)/2))
        gate.append({'day':day,'queries':len(eligible),'availability_upper_bound':float(np.mean(bounds))})
        neighbors=pd.read_csv(ref/'neighbors.csv');neighbors=neighbors[neighbors['rank']<=10]
        nr=idx.get_indexer(neighbors.host_id);nc=idx.get_indexer(neighbors.neighbor_id)
        good=(nr>=0)&(nc>=0);pairs=np.stack([nr[good],nc[good]],axis=1)
        pairframe=pd.read_csv(ref/'pairs.csv');pa=idx.get_indexer(pairframe.host_a);pb=idx.get_indexer(pairframe.host_b)
        good=(pa>=0)&(pb>=0);shared=(pa[good],pb[good],pairframe.label.to_numpy()[good])
        edges=pd.read_csv(ref/'edges.csv');ea=idx.get_indexer(edges.host_a);eb=idx.get_indexer(edges.host_b);ok=(ea>=0)&(eb>=0)
        links=(ea[ok],eb[ok],edges.label.to_numpy()[ok])
        episodes.append((torch.from_numpy(features(late_observations[len(episodes)])),active,eligible,target,pairs,shared,links))
    protocol={'train_days':[11,12],'input_encoder':'stateless hashed log traffic tokens, PCA trained on source and11/12 only','development_day':13,'test_days':[14,15,16],
              'seed':a.seed,'dimension':a.dim,'steps':a.steps,'source_contacts_day':2,
              'scope':'local pilot, different references; not official leaderboard','code_sha256':signature(Path(__file__)),'early_bounds':gate,
              'files_sha256':{str(p):signature(p) for p in [source_path]+[Path(o.window.flows) for o in late_observations[:2]]}}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if any(g['availability_upper_bound']<.54 for g in gate):
        (a.output/'STOP.json').write_text(json.dumps({'reason':'historical availability below .54 budget','not_impossibility_proof_for_test':True}));return
    def frame(x):return pd.DataFrame(x,index=pd.Index(hosts,name='host_id')).add_prefix('emb_')
    dev_features=torch.from_numpy(features(late_observations[2]))
    history=[]
    for step in range(a.steps):
        x,active,eligible,target,neighbors,shared,links=episodes[step%2]
        z=model.vectors();q=rng.choice(eligible,min(16,len(eligible)),replace=False)
        similarity=z[q]@z[active].T
        similarity=similarity.masked_fill(torch.from_numpy(q[:,None]==active[None,:]),-20)
        weights=soft_topk(similarity)
        full=torch.zeros((len(q),len(hosts))).scatter(1,torch.from_numpy(np.broadcast_to(active,(len(q),len(active))).copy()),weights)
        votes=torch.sparse.mm(tensor.T,full.T).T
        logs=torch.log1p(votes);positive=torch.from_numpy(target[q].toarray())
        best=logs.masked_fill(~positive,-1e9).max(1).values
        ranks=.5+(torch.sigmoid((logs-best[:,None])/.1)*valid[q]).sum(1)
        vote=ranks.log().mean()
        ids=rng.choice(active,min(512,len(active)),replace=False)
        logits=aligned_logits(z[ids],model.late(x[ids]));identity=F.cross_entropy(logits,torch.arange(logits.shape[0]))
        select=neighbors[rng.choice(len(neighbors),256)]
        neg=rng.choice(active,256)
        neighbor=F.softplus((z[select[:,0]]*z[neg]).sum(1)-(z[select[:,0]]*z[select[:,1]]).sum(1)+.05).mean()
        sa,sb,sy=shared;pick=rng.choice(len(sa),256)
        pair=F.binary_cross_entropy_with_logits(8*(z[sa[pick]]*z[sb[pick]]).sum(1),torch.tensor(sy[pick],dtype=torch.float32))
        anchor_loss=(z-initial).square().sum(1).mean()
        la,lb,ly=links;pick=rng.choice(len(la),256)
        left=z[la[pick]];right=z[lb[pick]]
        link=F.binary_cross_entropy_with_logits(model.link_head(torch.cat([(left-right).abs(),left*right],1)).flatten(),torch.tensor(ly[pick],dtype=torch.float32))
        loss=vote+.5*identity+neighbor+.2*pair+.2*link+2*anchor_loss
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training')
        if step%20==0 or step==a.steps-1:
            row={'step':step,'loss':float(loss.detach()),'vote':float(vote.detach()),'identity':float(identity.detach())};history.append(row);print(row,flush=True)
    torch.save({'state':model.state_dict(),'pca':pca,'hosts':hosts,'protocol':protocol},a.output/'model.pt')
    source=frame(model.vectors().detach().numpy());source.to_csv(a.output/'source.csv')
    (a.output/'training.json').write_text(json.dumps(history,indent=2))
    metadata=json.loads((dev/'manifest.json').read_text())
    metadata['neighbor_ceiling']=json.loads((root/'fingerprint_recent/candidates/ema3/result.json').read_text())['score_neighbors_ideal']
    local_ref=a.output/'reference_development';shutil.copytree(dev,local_ref)
    (local_ref/'manifest.json').write_text(json.dumps(metadata,indent=2))
    core=geometry(source,local_ref,probe,cache)
    late=frame(model.late(dev_features).detach().numpy());late.to_csv(a.output/'validation.csv')
    result=with_identity(source,late,local_ref,core)
    (a.output/'development.json').write_text(json.dumps(result,indent=2))
    print('development',result['score'],flush=True)

if __name__=='__main__':main()
