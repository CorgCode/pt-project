"""Обучаемый host↔token граф: raw peers/services/context, без frozen-output bottleneck."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F

from experiments.fingerprint_donors import peer_matrix
from experiments.fingerprint_memory import read_fingerprints
from experiments.fingerprint_search import BASE,SOURCE,dump,load,write_json
from experiments.fingerprint_transfer import PartnerPair
from experiments.multiday_gain import check_chronology
from experiments.overnight.model import T
from pipeline.data import ProcessedData


ALIGNED=SOURCE/'fingerprint_aligned'
TRANSFER=SOURCE/'fingerprint_transfer'
ORIGINAL=SOURCE/'fingerprint_memory'
SERVICE_BINS,CONTEXT_BINS=8192,65536
DIM=128


def source_signature():
    folder=Path(__file__).resolve().parent
    files=[folder/name for name in ['fingerprint_bigraph.py','fingerprint_bigraph_search.py',
        'fingerprint_transfer.py','fingerprint_memory.py']]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_revision(work):
    path=work/'training_revision.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('sources')!=source_signature() or saved.get('torch_version')!=str(torch.__version__):
            raise RuntimeError('Код/версия PyTorch изменились после фиксации обучения; запуск остановлен')


def top_rows(matrix,k):
    matrix=matrix.tocsr(copy=True)
    for row in range(matrix.shape[0]):
        start,end=matrix.indptr[row:row+2]
        if end-start>k:
            values=matrix.data[start:end]
            keep=np.argpartition(values,-k)[-k:]
            mask=np.ones(len(values),dtype=bool);mask[keep]=False;values[mask]=0
    matrix.eliminate_zeros()
    return matrix


def hashed(tokens,width):
    return (pd.util.hash_array(np.asarray(tokens,dtype=object),hash_key='0123456789123456')%width).astype(np.int32)


class TokenBank:
    def __init__(self,hosts,peers):
        self.hosts=list(hosts);self.peers=pd.Index(peers)
        self.graph_width=len(peers)+SERVICE_BINS
        self.width=self.graph_width+CONTEXT_BINS

    def fit(self,raw):
        self.seen=np.asarray(raw.getnnz(axis=0)).ravel()>0
        self.tfidf=TfidfTransformer(norm=None,sublinear_tf=True).fit(raw)
        return self

    def raw(self,counts):
        peer=T.matrix(counts['counts']['peers'],self.hosts,self.peers).astype(np.float32)
        context=counts['counts']['context']
        tokens=context.index.get_level_values('t').astype(str)
        rows=pd.Index(self.hosts).get_indexer(context.index.get_level_values('h'))
        if (rows<0).any():raise ValueError('Хост вне исходной когорты')
        services=np.array([v[:2]+v.rsplit('|',1)[-1] for v in tokens],dtype=object)
        values=context.to_numpy(dtype=np.float32)
        service=sp.csr_matrix((values,(rows,hashed(services,SERVICE_BINS))),shape=(len(self.hosts),SERVICE_BINS))
        ctx=sp.csr_matrix((values,(rows,hashed(tokens,CONTEXT_BINS))),shape=(len(self.hosts),CONTEXT_BINS))
        return sp.hstack([peer,service,ctx],format='csr')

    def transform_raw(self,raw):
        value=self.tfidf.transform(raw).multiply(self.seen[None,:]).tocsr().astype(np.float32)
        boundaries=[0,len(self.peers),self.graph_width,self.width]
        parts=[]
        for lo,hi,k,weight in zip(boundaries[:-1],boundaries[1:],[256,64,256],[1.,0.5,0.75]):
            parts.append(normalize(top_rows(value[:,lo:hi],k))*weight)
        return normalize(sp.hstack(parts,format='csr')).astype(np.float32)

    def transform(self,counts):
        return self.transform_raw(self.raw(counts))


def tensor_sparse(matrix):
    matrix=matrix.tocoo()
    return torch.sparse_coo_tensor(np.vstack([matrix.row,matrix.col]),matrix.data.astype(np.float32),
        matrix.shape,check_invariants=True).coalesce()


def contact_scores(z,queries,source,partners,hard_mix=0.0):
    """Плавный переход от soft-соседей к исходным top10 с положительным cosine."""
    q=torch.as_tensor(queries,dtype=torch.long)
    similarity=z[q]@z.T
    allowed=np.diff(source.indptr)>0
    similarity[:,torch.from_numpy(~allowed)]=-1e9
    similarity[torch.arange(len(q)),q]=-1e9
    soft=torch.softmax(similarity/0.1,dim=1)
    k=min(10,int(allowed.sum())-1)
    if k<1:raise ValueError('Недостаточно допустимых доноров')
    values,indices=torch.topk(similarity,k,dim=1)
    hard=torch.zeros_like(similarity).scatter(1,indices,values.clamp_min(0))
    hard=hard/hard.sum(1,keepdim=True).clamp_min(1e-12)
    weights=soft*(1-hard_mix)+hard*hard_mix
    matrix=source[:,partners].T.tocoo()
    sparse=tensor_sparse(matrix)
    return torch.sparse.mm(sparse,weights.T).T


class BipartiteEncoder(nn.Module):
    def __init__(self,width,dim=DIM):
        super().__init__()
        self.embedding=nn.Parameter(torch.randn(width,dim)*0.02)
        self.aux_map=nn.Linear(dim,dim,bias=False)
        nn.init.eye_(self.aux_map.weight)
        self.role=nn.Sequential(nn.Linear(dim*2,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(0.1),nn.Linear(256,dim))
        nn.init.zeros_(self.role[-1].weight);nn.init.zeros_(self.role[-1].bias)

    def own(self,features,aux):
        raw=torch.sparse.mm(features,self.embedding)
        if self.training:
            # Случайно убираем весь teacher-вход части хостов: raw-путь должен
            # учиться самостоятельно, а не стать ещё одной frozen-head моделью.
            aux=aux*(torch.rand((len(aux),1),device=aux.device)>=0.25).to(aux.dtype)/0.75
        return raw+self.aux_map(aux)

    def combine(self,own,role):
        return F.normalize(own+self.role(torch.cat([own,role],dim=1)),dim=1)

    def prototype(self,features,graph,reverse,aux):
        own=self.own(features,aux)
        token_context=torch.sparse.mm(reverse,own)
        role=torch.sparse.mm(graph,token_context)
        return self.combine(own,role),token_context

    def observed(self,features,graph,aux,token_context):
        own=self.own(features,aux)
        return self.combine(own,torch.sparse.mm(graph,token_context))


class BiGraphModel:
    def __init__(self,base,bank,net,pca,source_aux,cache):
        self.base,self.bank,self.net=base,bank,net
        self.pca,self.source_aux,self.cache=pca,source_aux,str(cache)
        self._context=None

    def _prepare(self):
        self.net.eval()
        if self._context is None:
            graph=normalize(self.bank.prototype[:,:self.bank.graph_width],norm='l1')
            reverse=normalize(graph.T.tocsr(),norm='l1')
            with torch.no_grad():
                source,context=self.net.prototype(tensor_sparse(self.bank.prototype),tensor_sparse(graph),
                    tensor_sparse(reverse),torch.from_numpy(self.source_aux))
            self._context=(source.numpy(),context)

    def embed(self,data,window,hosts):
        self._prepare()
        rows=pd.Index(self.bank.hosts).get_indexer(hosts)
        if (rows<0).any():raise ValueError('Хост вне исходной когорты')
        if window=='train':
            values=self._context[0][rows]
        else:
            counts=read_fingerprints(data,window,self.bank.hosts,self.cache)
            raw=self.bank.transform(counts)[rows]
            graph=normalize(raw[:,:self.bank.graph_width],norm='l1')
            aux=normalize(self.pca.transform(normalize(self.base.embed(data,window,hosts).to_numpy(dtype=np.float32)))).astype(np.float32)
            with torch.no_grad():
                values=self.net.observed(tensor_sparse(raw),tensor_sparse(graph),torch.from_numpy(aux),self._context[1]).numpy()
        return pd.DataFrame(values,index=hosts).add_prefix('emb_')


def inputs(work):
    if (work/'inputs_ready.json').exists():return
    metadata=json.loads((ORIGINAL/'bank_metadata.json').read_text())
    days=[ProcessedData(Path(p)) for p in metadata['training_sources']]
    check_chronology(days,9*86400+12.5*3600)
    if any(d.manifest['windows']['test']['end']>9*86400+12.5*3600 for d in days):
        raise ValueError('Историческое позднее окно пересекает development10')
    hosts=days[0].cohort
    sources=[(day,data,window) for day,data in zip(range(2,10),days) for window in ['train','test']]
    peers=set()
    for day,data,window in sources:
        counts=read_fingerprints(data,window,hosts,ORIGINAL/'count_cache')
        peers.update(counts['counts']['peers'].index.get_level_values('t'))
        del counts;gc.collect()
    bank=TokenBank(hosts,sorted(peers))
    paths,events=[],[]
    for day,data,window in sources:
        counts=read_fingerprints(data,window,hosts,ORIGINAL/'count_cache')
        raw=bank.raw(counts)
        target=work/f'raw{day:02d}_{window}.npz';sp.save_npz(target,raw)
        paths.append(target);events.append(counts['events'])
        del counts,raw;gc.collect()
        print(json.dumps({'prepared_peer_day':day,'window':window,'raw_tokens':bank.width}),flush=True)
    stacked=sp.vstack([sp.load_npz(path) for path in paths],format='csr')
    bank.fit(stacked)
    del stacked;gc.collect()
    prototype=sp.csr_matrix((len(hosts),bank.width),dtype=np.float32)
    for i,(path,(day,_,window),active) in enumerate(zip(paths,sources,events)):
        view=bank.transform_raw(sp.load_npz(path)).multiply((active>=20)[:,None]).tocsr()
        view.eliminate_zeros()
        sp.save_npz(work/f'view{day:02d}_{window}.npz',view)
        prototype+=view*(2.**(-(9-day)/3))
    # Капы сохраняют все три канала и ограничивают стоимость sparse message passing.
    cuts=[0,len(bank.peers),bank.graph_width,bank.width]
    bank.prototype=normalize(sp.hstack([normalize(top_rows(prototype[:,a:b],k))*w
        for a,b,k,w in zip(cuts[:-1],cuts[1:],[256,64,256],[1.,0.5,0.75])],format='csr')).astype(np.float32)
    anchor=np.load(ALIGNED/'anchor.npy')
    if len(anchor)!=len(hosts):raise ValueError('Teacher и raw cohort различаются')
    pca=PCA(n_components=DIM,svd_solver='randomized',random_state=73).fit(normalize(anchor))
    source_aux=normalize(pca.transform(normalize(anchor))).astype(np.float32)
    np.save(work/'source_aux.npy',source_aux)
    for day in range(2,10):
        full=np.load(ALIGNED/f'historical{day:02d}.npy')
        aux=normalize(pca.transform(normalize(full))).astype(np.float32)
        np.save(work/f'aux{day:02d}.npy',aux)
    dump(pca,work/'aux_pca.pkl');dump(bank,work/'bank.pkl')
    write_json(work/'hosts.json',hosts)
    write_json(work/'inputs_ready.json',{'n_hosts':len(hosts),'width':bank.width,'peer_tokens':len(bank.peers),
        'graph_width':bank.graph_width,'prototype_nnz':bank.prototype.nnz,'raw_windows':16,
        'pairs':json.loads((ALIGNED/'inputs_ready.json').read_text())['pairs'],
        'validation_used_for_training':False,'teacher_role':'auxiliary train-only projection; raw paths are trainable'})


def classifier_loss(z,classifier,targets):
    logits=z@F.normalize(classifier,dim=1).T/0.1
    logits[torch.arange(len(targets)),targets]-=1.0
    return F.cross_entropy(logits,targets)


def train(work,rank_weight,epochs=1800,checkpoints=(600,1800),warmup=200,threads=None):
    validate_revision(work)
    torch.set_num_threads(max(1,int(threads or os.environ.get('PT_TORCH_THREADS','4'))));torch.manual_seed(17)
    label='graph_rank05' if rank_weight==0.5 else 'graph_rank20'
    out=work/label;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    bank=load(work/'bank.pkl');net=BipartiteEncoder(bank.width)
    source_aux=torch.from_numpy(np.load(work/'source_aux.npy'))
    classifier=nn.Parameter(source_aux.clone())
    graph=normalize(bank.prototype[:,:bank.graph_width],norm='l1')
    p=tensor_sparse(bank.prototype);g=tensor_sparse(graph);reverse=tensor_sparse(normalize(graph.T.tocsr(),norm='l1'))
    xs=[];gs=[];aux=[];populations=[]
    for day in range(3,10):
        view=sp.load_npz(work/f'view{day:02d}_test.npz')
        xs.append(tensor_sparse(view));gs.append(tensor_sparse(normalize(view[:,:bank.graph_width],norm='l1')))
        aux.append(torch.from_numpy(np.load(work/f'aux{day:02d}.npy')))
        populations.append(np.load(ALIGNED/f'future_active{day:02d}.npy'))
    vocabulary=json.loads((TRANSFER/'vocabulary.json').read_text())
    source=sp.load_npz(TRANSFER/'source0.npz')
    pairs=[]
    for day,pop in zip(range(3,10),populations):
        matrix=source.multiply(pop[:,None]).tocsr();matrix.eliminate_zeros()
        pairs.append(PartnerPair(matrix,sp.load_npz(ALIGNED/f'future{day:02d}.npz'),bank.hosts,vocabulary,pop))
    parameters=[*net.parameters(),classifier]
    optimizer=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=np.random.default_rng(17);n=len(bank.hosts)
    for epoch in range(1,epochs+1):
        net.train();i=int(rng.integers(7));pop=np.flatnonzero(populations[i])
        z,context=net.prototype(p,g,reverse,source_aux)
        observed=net.observed(xs[i],gs[i],aux[i],context)
        selected=rng.choice(pop,min(512,len(pop)),replace=False);q=torch.from_numpy(selected)
        distill=(1-(z*source_aux).sum(1)).mean()+(1-(observed[q]*aux[i][q]).sum(1)).mean()
        identity=torch.zeros(());partner=torch.zeros(());classification=torch.zeros(())
        if epoch<=warmup:
            loss=distill
        else:
            src=torch.from_numpy(rng.choice(n,min(1024,n),replace=False))
            classification=0.5*(classifier_loss(z[src],classifier,src)+classifier_loss(observed[q],classifier,q))
            gallery=torch.from_numpy(pop);targets=torch.from_numpy(pd.Index(pop).get_indexer(selected))
            identity=0.5*(F.cross_entropy(z[q]@observed[gallery].T/0.1,targets)+
                          F.cross_entropy(observed[q]@z[gallery].T/0.1,targets))
            queries,candidates,unique,inverse=pairs[i].sample(rng)
            mix=0.9*min(1.,(epoch-warmup)/max(1,epochs-warmup))
            scores=contact_scores(z,queries,pairs[i].scaled,unique,mix)
            logits=scores.gather(1,torch.from_numpy(inverse)).clamp_min(1e-30).log()
            partner=F.cross_entropy(logits,torch.zeros(len(queries),dtype=torch.long))
            loss=classification+0.5*identity+rank_weight*partner+0.1*distill
        if not torch.isfinite(loss):raise FloatingPointError('Нечисловая потеря bipartite encoder')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(parameters,3)
        if not torch.isfinite(norm):raise FloatingPointError('Нечисловой градиент bipartite encoder')
        optimizer.step();schedule.step()
        if epoch%10==0:
            print(json.dumps({'epoch':epoch,'loss':float(loss.detach()),'phase':'warmup' if epoch<=warmup else 'joint',
                'classification_loss':float(classification.detach()),'identity_loss':float(identity.detach()),
                'partner_loss':float(partner.detach()),'distillation_loss':float(distill.detach()),'rank_weight':rank_weight,
                'hard_neighbor_mix':0.0 if epoch<=warmup else mix}),flush=True)
        if epoch in checkpoints:dump(net,out/f'epoch{epoch}.pkl')
    write_json(out/'complete.json',{'epochs':list(checkpoints),'rank_weight':rank_weight,'native_output_dim':DIM})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',required=True,type=Path)
    ap.add_argument('--stage',choices=['inputs','train'],required=True)
    ap.add_argument('--rank-weight',type=float,choices=[0.5,2.0],default=0.5)
    ap.add_argument('--threads',type=int)
    args=ap.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    if args.stage=='inputs':inputs(args.work)
    else:train(args.work,args.rank_weight,threads=args.threads)


if __name__=='__main__':main()
