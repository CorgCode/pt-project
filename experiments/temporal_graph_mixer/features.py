"""Четыре блока и направленная диффузия: fit только на train, без GNN."""
import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.preprocessing import StandardScaler,normalize


def unit(x):
    return normalize(np.asarray(x,np.float32)).astype(np.float32)


def rate_log(x,duration):
    x=x.copy().astype(np.float32);x.data=np.log1p(x.data*3600/duration);return x


def own_raw(obs,schema=1):
    duration=(obs['window']['end']-obs['window']['start'])/4
    numeric=np.log1p(obs['numeric']*3600/duration)
    # Распределение протоколов дополняет суммарные объёмы.
    proto=normalize(rate_log(obs['protocols'],duration)).toarray()
    if schema==1:return np.hstack([numeric,proto]).astype(np.float32)
    # Собственное распределение сервисов не требует таблицы идентификаторов.
    projection=sp.csr_matrix((np.ones(1024),(np.arange(1024),np.arange(1024)%128)),shape=(1024,128))
    services=obs['services'][:,:1024]@projection
    port_profile=normalize(rate_log(services,duration)).toarray()
    summaries=[]
    for matrix in [obs['peers'],obs['services'][:,:1024]]:
        probs=normalize(matrix,norm='l1').tocsr();entropy=probs.copy()
        entropy.data=-entropy.data*np.log(np.maximum(entropy.data,1e-30))
        count=np.diff(matrix.indptr).astype(float)
        summaries.extend([np.log1p(count),np.asarray(entropy.sum(1)).ravel(),
                          np.asarray(probs.max(axis=1).toarray()).ravel(),
                          np.asarray(matrix.sum(1)).ravel()/np.maximum(count,1)])
    summary=np.stack(summaries,axis=1);summary[:,[3,7]]=np.log1p(summary[:,[3,7]])
    return np.hstack([numeric,proto,port_profile,summary]).astype(np.float32)


def row_graph(matrix):
    a=matrix.copy().astype(np.float32);a.data=np.log1p(a.data)
    # Ослабление популярных узлов назначения до нормировки строк.
    degree=np.asarray((a>0).sum(0)).ravel();a=a@sp.diags(1/np.sqrt(np.maximum(degree,1)))
    return normalize(a,norm='l1',axis=1).tocsr()


def diffusion(obs,quarter_self):
    n=len(obs['hosts']);out=[]
    # Граф четверти часа использует только её собственные рёбра.
    # Извлечение сохраняет отдельный граф каждого quarter.
    for t in range(4):
        x=quarter_self[t*n:(t+1)*n];parts=[]
        for key in ['quarter_counts','quarter_bytes']:
            graph=obs[key][t]
            for a in [graph,graph.T]:
                p=row_graph(a);first=p@x;second=p@first
                parts.extend([first,second,np.abs(first-second)])
        out.append(np.hstack(parts))
    return np.vstack(out).astype(np.float32)


class BlockFeatures:
    def __init__(self,dim=128,own_schema=2):self.dim,self.own_schema=dim,own_schema

    def fit(self,observations):
        self.training_end=max(o['window']['end'] for o in observations)
        self.train_windows=[o['window'] for o in observations]
        self.own_scaler=StandardScaler().fit(np.vstack([own_raw(o,getattr(self,'own_schema',1)) for o in observations]))
        own=np.vstack([self.own_scaler.transform(own_raw(o,getattr(self,'own_schema',1))) for o in observations])
        self.own_pca=PCA(n_components=min(self.dim,*own.shape),svd_solver='randomized',random_state=17).fit(own)
        self.models={}
        for kind in ['peers','services']:
            matrices=[rate_log(o[kind],(o['window']['end']-o['window']['start'])/4) for o in observations]
            # DF считает уникальные хосты по всей разрешённой истории.
            n=len(observations[0]['hosts']);seen=sp.csr_matrix((n,matrices[0].shape[1]))
            for x in matrices:
                for t in range(4):seen+=x[t*n:(t+1)*n].astype(bool).astype(np.float32)
            df=np.asarray((seen>0).sum(0)).ravel();idf=(np.log((n+1)/(df+1))+1).astype(np.float32)
            fit=sp.vstack([normalize(x.multiply(idf)) for x in matrices],format='csr')
            svd=TruncatedSVD(n_components=min(self.dim,min(fit.shape)-1),n_iter=4,random_state=17).fit(fit)
            self.models[kind]=(idf,svd)
            print('fitted block',kind,fit.shape,flush=True)
        graph=[]
        for o in observations:
            # Начальная проекция поведения/сервисов ограничивает RAM;
            # окончательный графовый профиль сжимается после диффузии.
            s=self.pre_graph(o);graph.append(diffusion(o,s))
        values=np.vstack(graph);self.graph_scaler=StandardScaler().fit(values)
        self.graph_svd=PCA(n_components=min(self.dim,*values.shape),svd_solver='randomized',random_state=17).fit(self.graph_scaler.transform(values))
        print('graph block fitted',values.shape,flush=True)
        return self

    def project(self,obs,kind):
        idf,svd=self.models[kind];x=rate_log(obs[kind],(obs['window']['end']-obs['window']['start'])/4)
        return unit(svd.transform(normalize(x.multiply(idf))))

    def own(self,obs):
        return self.own_pca.transform(np.clip(self.own_scaler.transform(own_raw(obs,getattr(self,'own_schema',1))),-8,8)).astype(np.float32)

    def pre_graph(self,obs):
        # Первые 32 компоненты двух блоков; преобразования общие для всех окон.
        return np.hstack([unit(self.own(obs))[:,:32],self.project(obs,'services')[:,:32]]).astype(np.float32)

    def transform(self,obs):
        parts=[self.own(obs),self.project(obs,'peers'),self.project(obs,'services'),
               self.graph_svd.transform(np.clip(self.graph_scaler.transform(diffusion(obs,self.pre_graph(obs))),-8,8))]
        # Масштабы фиксированы по train; поздний граф ничего не fit.
        if hasattr(self,'block_scale'):parts=[(x-m)/s for x,(m,s) in zip(parts,self.block_scale)]
        return np.stack([x.reshape(4,len(obs['hosts']),-1).transpose(1,0,2) for x in parts],axis=2).astype(np.float32)

    def calibrate(self,observations):
        self.block_scale=[(0.,1.)]*4
        values=[self.transform(o) for o in observations]
        self.block_scale=[]
        for k in range(4):
            x=np.concatenate([v[:,:,k,:].reshape(-1,v.shape[-1]) for v in values])
            mean=x.mean(0);scale=max(np.sqrt(((x-mean)**2).sum(1).mean()),1e-8)
            self.block_scale.append((mean,float(scale)))
        return self
