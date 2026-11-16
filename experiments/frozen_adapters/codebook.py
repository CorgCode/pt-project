"""Portable reconstruction of the published codebook/plus design, trained from flows.

The unpublished original base checkpoint and plus trainer are not reproduced bitwise.
Losses follow historical codebook.py/listwise.py at 86d709d; plus dimensions, weights
and graph/lookup recipe follow its public submission card at 70347a3.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize, StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from evaluation.knn_consistency import build_reference_features
from .data import contacts
from .neighbors import contact_matrix


def unit(values):
    return normalize(np.asarray(values, dtype=np.float32)).astype(np.float32)


def behavior(window, hosts, cache, fitted=None):
    """Project's actual behavioral features; columns/scalers fitted on source only."""
    path = Path(window.flows)
    stamp = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns,
             window.start, window.end, hosts, joblib.hash(fitted), 'behavior-v1')
    saved = Path(cache) / ('behavior-' + hashlib.sha256(repr(stamp).encode()).hexdigest() + '.joblib')
    if saved.exists():
        return joblib.load(saved)
    columns = ['src','dst','dst_port','proto','bytes','packets','ts','duration']
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000,
                             dtype={c:'str' for c in ('src','dst','dst_port','proto')}):
        part = chunk.loc[(chunk.ts >= window.start) & (chunk.ts < window.end) &
                         (chunk.ts + chunk.duration < window.end) & (chunk.src != chunk.dst)]
        parts.append(part)
    flows = pd.concat(parts, ignore_index=True)
    for column in ('src','dst','dst_port','proto'):
        flows[column] = flows[column].astype('category')
    features, state = build_reference_features(flows, hosts=hosts, fitted=fitted, return_fitted=True)
    result = (features.to_numpy(dtype=np.float32), state)
    saved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, saved, compress=3)
    return result


class TrafficFeatures:
    """Behavior + hashed peer/service TF-IDF + directed neighbor diffusion."""
    def __init__(self, hosts, cache, width=32768, token_dim=128, linear_dim=512):
        self.hosts, self.cache = list(hosts), str(cache)
        self.width, self.token_dim, self.linear_dim = width, token_dim, linear_dim
        self.hasher = FeatureHasher(n_features=width, alternate_sign=False)

    def hashed(self, obs, kind):
        prefix = 'p:' if kind == 'peers' else 's:'
        rate = 3600 / (obs.window.end - obs.window.start)
        return self.hasher.transform([{k:np.log1p(v*rate) for k,v in row.items()
                                      if k.startswith(prefix)} for row in obs.counts]).astype(np.float32)

    def fit(self, observations, source_observation=None):
        source_observation = source_observation or observations[0]
        self.behavior_state = behavior(source_observation.window, self.hosts, self.cache)[1]
        self.models = {}
        for kind in ('peers','services'):
            matrices = [self.hashed(o,kind) for o in observations]
            training = sp.vstack([x[o.events >= 300] for x,o in zip(matrices,observations)],format='csr')
            tfidf = TfidfTransformer().fit(training)
            rank = min(self.token_dim, min(training.shape)-1)
            svd = TruncatedSVD(rank, random_state=17).fit(tfidf.transform(training))
            self.models[kind] = (tfidf,svd)
        raw = [self.raw(o) for o in observations]
        active = np.vstack([x[o.events >= 300] for x,o in zip(raw,observations)])
        self.scaler = StandardScaler().fit(active)
        self.pca = PCA(n_components=min(self.linear_dim,*active.shape), random_state=17).fit(active)
        return self

    def raw(self, observation):
        blocks = {}
        for kind,(tfidf,svd) in self.models.items():
            blocks[kind] = unit(svd.transform(tfidf.transform(self.hashed(observation,kind))))
        index = pd.Index(self.hosts)
        rows,cols,weights = [],[],[]
        for row,tokens in enumerate(observation.counts):
            for token,count in tokens.items():
                if token.startswith('p:o:'):
                    col = index.get_indexer([token[4:]])[0]
                    if col >= 0 and col != row:
                        rows.append(row); cols.append(col); weights.append(np.log1p(count))
        graph = sp.csr_matrix((weights,(rows,cols)),shape=(len(index),len(index)))
        out,inn = normalize(graph,norm='l1'),normalize(graph.T,norm='l1')
        svc = blocks['services']
        diffusion = unit(np.hstack([out@svc,inn@svc,.5*(out@(inn@svc)+inn@(out@svc))]))
        beh = behavior(observation.window,self.hosts,self.cache,self.behavior_state)[0]
        return np.hstack([unit(beh)*4,blocks['peers'],blocks['services'],diffusion*.5]).astype(np.float32)

    def transform(self, observation):
        return self.pca.transform(self.raw(observation)).astype(np.float32)


class TrafficHead(nn.Module):
    def __init__(self, inputs, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs,512),nn.LayerNorm(512),nn.GELU(),
                                 nn.Dropout(.1),nn.Linear(512,dim))
        self.skip = nn.Linear(inputs,dim,bias=False)

    def forward(self, x):
        return F.normalize(self.net(x)+.2*self.skip(x),dim=1)


def train_traffic_head(history, active, teacher, dim, steps=300, seed=17):
    torch.manual_seed(seed)
    n = len(teacher)
    head = TrafficHead(history[0].shape[1],dim)
    classifier = nn.Parameter(torch.randn(n,dim)*.02)
    optimizer = torch.optim.AdamW([*head.parameters(),classifier],lr=1e-3,weight_decay=1e-4)
    tensors = [torch.from_numpy(x) for x in history]
    present = np.stack(active)
    eligible = np.flatnonzero(present.sum(0)>=2)
    if not len(eligible):
        raise ValueError('traffic heads need hosts observed in at least two training windows')
    rng = np.random.default_rng(seed)
    reference = torch.from_numpy(unit(teacher))
    log = []
    for step in range(steps):
        labels = rng.choice(eligible,min(128,len(eligible)),replace=False)
        pairs = np.array([rng.choice(np.flatnonzero(present[:,h]),2,replace=False) for h in labels])
        a = torch.stack([tensors[d][h] for d,h in zip(pairs[:,0],labels)])
        b = torch.stack([tensors[d][h] for d,h in zip(pairs[:,1],labels)])
        za,zb = head(a),head(b)
        q = torch.from_numpy(labels)
        target = torch.cat([q,q])
        logits = torch.cat([za,zb])@F.normalize(classifier,dim=1).T/.1
        logits[torch.arange(len(target)),target] -= 1.
        classification = F.cross_entropy(logits,target)
        pair = za@zb.T/.1
        truth = torch.arange(len(labels))
        contrast = .5*(F.cross_entropy(pair,truth)+F.cross_entropy(pair.T,truth))
        relation = F.mse_loss(za@za.T,reference[q]@reference[q].T)
        loss = classification+.25*contrast+.1*relation
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_([*head.parameters(),classifier],3)
        optimizer.step()
        if (step+1)%50==0:
            row={'stage':'traffic','dim':dim,'step':step+1,'loss':float(loss.detach())}
            log.append(row);print(row,flush=True)
    head.eval()
    return head,log


def encode_head(head, values):
    with torch.no_grad():
        return np.vstack([head(torch.from_numpy(values[i:i+256])).numpy()
                          for i in range(0,len(values),256)])


class PartnerTask:
    """Exact source population/universe and ANY valid new historical partner."""
    def __init__(self, source_rows, target, hosts):
        self.population = np.asarray(target.events >= 300)
        visible = [r if self.population[i] else {} for i,r in enumerate(source_rows)]
        self.vocabulary = sorted(set().union(*(set(r) for r in visible)))
        self.matrix = contact_matrix(visible,self.vocabulary)
        vocab = {v:i for i,v in enumerate(self.vocabulary)}
        rows,cols = [],[]
        self.valid = np.ones((len(hosts),len(vocab)),bool)
        self.valid[:,np.asarray(self.matrix.sum(0)).ravel()==0] = False
        for q,(old,new,host) in enumerate(zip(source_rows,contacts(target),hosts)):
            for peer in old:
                if peer in vocab: self.valid[q,vocab[peer]]=False
            if host in vocab: self.valid[q,vocab[host]]=False
            for peer in new:
                if peer in vocab and self.valid[q,vocab[peer]]:
                    rows.append(q);cols.append(vocab[peer])
        self.positive = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=self.matrix.shape)
        self.eligible = np.flatnonzero(self.population & (np.diff(self.positive.indptr)>0))
        coo = self.matrix.T.tocoo()
        self.tensor = torch.sparse_coo_tensor(np.vstack([coo.row,coo.col]),coo.data.astype(np.float32),coo.shape).coalesce()


def partner_loss(z, queries, task):
    q = torch.as_tensor(queries)
    sim = z[q]@z.T
    sim[:,~torch.from_numpy(task.population)] = -1e9
    sim[torch.arange(len(q)),q] = -1e9
    soft = torch.softmax(sim/.1,dim=1)
    k = min(10,int(task.population.sum())-1)
    values,indices = sim.topk(k,dim=1)
    hard = torch.zeros_like(sim).scatter(1,indices,values.clamp_min(0))
    hard = hard/hard.sum(1,keepdim=True).clamp_min(1e-12)
    votes = torch.sparse.mm(task.tensor,(.1*soft+.9*hard).T).T
    valid = torch.from_numpy(task.valid[queries])
    positive = torch.from_numpy(task.positive[queries].toarray()>0)
    logits = votes.clamp_min(1e-30).log()/.2
    loss = (torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-
            torch.logsumexp(logits.masked_fill(~positive,-torch.inf),1)).mean()
    return loss


def train_codes(anchor, tasks, steps, regularization=1., seed=17, dim=None):
    torch.manual_seed(seed)
    teacher = torch.from_numpy(unit(anchor))
    if dim is None:
        initial = teacher.clone()
    else:
        initial = torch.from_numpy(unit(PCA(n_components=min(dim,*anchor.shape),random_state=seed).fit_transform(anchor)))
    codes = nn.Parameter(initial)
    optimizer = torch.optim.AdamW([codes],lr=3e-4,weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(steps,1))
    rng = np.random.default_rng(seed)
    usable = [t for t in tasks if len(t.eligible)]
    if not usable: raise ValueError('no historical new-partner targets')
    log = []
    for step in range(steps):
        task = usable[int(rng.integers(len(usable)))]
        queries = rng.choice(task.eligible,min(64,len(task.eligible)),replace=False)
        z = F.normalize(codes,dim=1)
        partner = partner_loss(z,queries,task)
        q = torch.from_numpy(queries)
        logits = z[q]@z.T/.05
        target = teacher[q]@teacher.T/.05
        logits[torch.arange(len(q)),q] = -1e9
        target[torch.arange(len(q)),q] = -1e9
        preserve = F.kl_div(F.log_softmax(logits,1),F.softmax(target,1),reduction='batchmean')
        anchor_loss = (1-(z*teacher).sum(1)).mean() if dim is None else codes.square().mean()
        loss = partner+regularization*preserve+.1*anchor_loss
        if not torch.isfinite(loss): raise FloatingPointError('non-finite codebook loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_([codes],3);optimizer.step();schedule.step()
        if (step+1)%50==0 or step+1==steps:
            row={'stage':'source' if dim is None else 'partner_head','step':step+1,
                 'loss':float(loss.detach()),'partner_loss':float(partner.detach()),
                 'neighbor_kl':float(preserve.detach())}
            log.append(row);print(row,flush=True)
    return F.normalize(codes,dim=1).detach().numpy(),log


def graph_block(observations, hosts, dim=64, half_life=3.):
    index = pd.Index(hosts)
    graph = sp.csr_matrix((len(hosts),len(hosts)),dtype=np.float64)
    cutoff = max(o.window.day for o in observations)
    for obs in observations:
        rows,cols,weights = [],[],[]
        for row,peers in enumerate(contacts(obs)):
            for peer,count in peers.items():
                col = index.get_indexer([peer])[0]
                if col>=0 and col!=row:
                    rows.append(row);cols.append(col);weights.append(count*2**(-(cutoff-obs.window.day)/half_life))
        graph += sp.csr_matrix((weights,(rows,cols)),shape=graph.shape)
    graph = (graph+graph.T)*.5
    graph.data = np.log1p(graph.data)
    inv = 1/np.sqrt(np.maximum(np.asarray(graph.sum(1)).ravel(),1e-12))
    norm = sp.diags(inv)@graph@sp.diags(inv)
    rank = min(dim,len(hosts)-2)
    eigenvalues,vectors = eigsh(norm,k=rank,which='LM',v0=np.random.default_rng(17).standard_normal(len(hosts)))
    vectors = vectors[:,np.argsort(-np.abs(eigenvalues))]
    # Resolve each eigenvector's sign deterministically for checkpoint reproducibility.
    vectors *= np.sign(vectors[np.abs(vectors).argmax(0),np.arange(rank)])[None,:]
    return unit(vectors)


class RebuiltPlus:
    """Known-host source profiles; late inference sees traffic, never a host-ID lookup."""
    def traffic(self, observation):
        linear = self.features.transform(observation)
        standardized = np.clip(self.linear_scaler.transform(linear),-8,8).astype(np.float32)
        blocks = [linear,*[encode_head(h,standardized) for h in self.heads]]
        return unit(np.hstack([(x-mean)/scale for x,(mean,scale) in zip(blocks,self.block_states)]))

    def late(self, observation, blend=.5):
        traffic = self.traffic(observation)
        adapted = unit((1-blend)*traffic+blend*unit(self.ridge.predict(traffic)))
        return self.fuse(adapted,self.lookup(adapted,self.partner),self.lookup(adapted,self.graph))

    def lookup(self, queries, values):
        result=[]
        for start in range(0,len(queries),128):
            scores=unit(queries[start:start+128])@unit(self.codes).T
            top=np.argsort(-scores,axis=1,kind='stable')[:,:min(20,len(values))]
            weights=np.take_along_axis(scores,top,axis=1)
            weights=np.exp((weights-weights[:,:1])/.02);weights/=weights.sum(1,keepdims=True)
            result.append(unit(np.einsum('bk,bkd->bd',weights,values[top])))
        return np.vstack(result)

    @staticmethod
    def fuse(base, partner, graph):
        return unit(np.hstack([unit(base),unit(partner)*2.,unit(graph)*.1]))

    @property
    def source(self):
        return self.fuse(self.codes,self.partner,self.graph)

    def fit(self, observations, hosts, cache, traffic_steps=300, code_steps=400, head_steps=200,
            source_observation=None):
        source_observation = source_observation or observations[0]
        self.hosts=list(hosts)
        self.training_end=max(o.window.end for o in [*observations,source_observation])
        self.features=TrafficFeatures(hosts,cache).fit(observations,source_observation)
        linear=[self.features.transform(o) for o in observations]
        source_linear=self.features.transform(source_observation)
        active=[o.events>=300 for o in observations]
        self.linear_scaler=StandardScaler().fit(np.vstack([x[a] for x,a in zip(linear,active)]))
        standardized=[np.clip(self.linear_scaler.transform(x),-8,8).astype(np.float32) for x in linear]
        self.heads=[];self.training_log=[]
        for dim in (128,256):
            head,log=train_traffic_head(standardized,active,source_linear,dim,steps=traffic_steps,seed=17+dim)
            self.heads.append(head);self.training_log+=log
        source_standardized=np.clip(self.linear_scaler.transform(source_linear),-8,8).astype(np.float32)
        source_blocks=[source_linear,*[encode_head(h,source_standardized) for h in self.heads]]
        self.block_states=[(x.mean(0),max(np.sqrt(((x-x.mean(0))**2).sum(1).mean()),1e-12)) for x in source_blocks]
        history=[self.traffic(o) for o in observations]
        tasks=[PartnerTask(contacts(source_observation),o,hosts) for o in observations
               if o.window.day>source_observation.window.day and o.window.start>=source_observation.window.end]
        self.codes,log=train_codes(self.traffic(source_observation),tasks,code_steps)
        self.training_log+=log
        xs=np.vstack([x[a] for x,a in zip(history,active)])
        ys=np.vstack([self.codes[a] for a in active])
        days=np.concatenate([np.full(a.sum(),o.window.day) for a,o in zip(active,observations)])
        weights=2.**(-(days.max()-days)/3.);weights*=len(weights)/weights.sum()
        self.ridge=Ridge(alpha=1.).fit(xs,ys,sample_weight=weights)
        self.partner,log=train_codes(self.codes,tasks,head_steps,regularization=3.,dim=64)
        self.training_log+=log
        self.graph=graph_block(observations,hosts)
        self.history_traffic=history
        return self


def predict_identity(adapter, baseline, observation, variant, temperature):
    if variant=='baseline': return baseline
    if variant.startswith('baseline_mix:'):
        weight=float(variant.split(':')[1])
        rare=adapter.predict(observation,'rare',temperature)
        return normalize((1-weight)*normalize(baseline)+weight*rare)
    return adapter.predict(observation,variant,temperature)


class PlusDelivery:
    """Stable importable checkpoint type, including when the trainer runs with -m."""
    def __init__(self,base,adapter,selection,blend,source,residual=None,strength=0.):
        self.base,self.adapter,self.selection=base,adapter,selection
        self.blend,self.source,self.residual,self.strength=blend,source,residual,strength
        self.hosts=base.hosts

    def predict(self,observation):
        baseline=self.base.late(observation,self.blend)
        if self.strength:
            from .run_neighbors import transfer_late
            baseline=transfer_late(baseline,self.base.source,self.source)
        return predict_identity(self.adapter,baseline,observation,**self.selection)
