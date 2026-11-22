"""Признаки четвертей часа, направленные объёмы и часовые частоты."""
from .config import load as load_config
from collections import Counter
from pathlib import Path
import argparse,json,hashlib
import joblib,numpy as np,pandas as pd
from scipy import sparse as sp
from sklearn.utils import murmurhash3_32

SCHEMA=2

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def bucket(strings,width):
    values,inverse=np.unique(np.asarray(strings,str),return_inverse=True)
    hashed=np.array([murmurhash3_32(v,seed=17,positive=True)%width for v in values],np.int32)
    return hashed[inverse]


def add_chunk(state,f,hosts,start,end):
    """Quarter features never see a flow ending after that quarter; hour counts do."""
    f=f.loc[(f.ts>=start)&(f.ts<end)&(f.ts+f.duration<end)&(f.src!=f.dst)].copy()
    if f.empty:return
    n=len(hosts);src=hosts.get_indexer(f.src);dst=hosts.get_indexer(f.dst)
    b=np.minimum(3,((f.ts.to_numpy()-start)*4/(end-start)).astype(int))
    quarter=(f.ts.to_numpy()+f.duration.to_numpy())<start+(b+1)*(end-start)/4
    tx=f['bytes'].to_numpy(float);rx=f.resp_bytes.to_numpy(float)
    pt=f.packets.to_numpy(float);pr=f.resp_packets.to_numpy(float)
    if not np.isfinite(np.c_[tx,rx,pt,pr,f.duration]).all() or np.any(np.c_[tx,rx,pt,pr,f.duration]<0):raise ValueError('invalid volumes/duration')
    both=(src>=0)&(dst>=0)
    state['counts']+=sp.csr_matrix((np.ones(both.sum()),(src[both],dst[both])),shape=(n,n))
    state['bytes']+=sp.csr_matrix((tx[both],(src[both],dst[both])),shape=(n,n))
    state['bytes']+=sp.csr_matrix((rx[both],(dst[both],src[both])),shape=(n,n))
    for t in range(4):
        selected=both&quarter&(b==t)
        a,d=src[selected],dst[selected]
        state['quarter_counts'][t]+=sp.csr_matrix((np.ones(len(a)),(a,d)),shape=(n,n))
        state['quarter_bytes'][t]+=sp.csr_matrix((tx[selected],(a,d)),shape=(n,n))
        state['quarter_bytes'][t]+=sp.csr_matrix((rx[selected],(d,a)),shape=(n,n))
    proto=f.proto.astype(str).to_numpy();service=f.proto.astype(str)+'/'+f.dst_port.astype(str)
    for side,(ids,peers,volumes) in enumerate([(src,f.dst.to_numpy(),(tx,rx,pt,pr)),(dst,f.src.to_numpy(),(rx,tx,pr,pt))]):
        active=ids>=0;state['events']+=np.bincount(ids[active],minlength=n)
        frame=pd.DataFrame({'id':ids[active],'peer':peers[active]})
        grouped=frame.groupby(['id','peer'],sort=False).size()
        for (i,p),v in grouped.items():state['contacts'][i][p]+=int(v)
        good=active&quarter
        rows=b[good]*n+ids[good]
        # TX/RX считаются отдельно для обоих концов потока.
        values=[*(v[good] for v in volumes),np.ones(good.sum()),f.duration.to_numpy()[good],
                tx[good]+rx[good],(proto[good]=='6').astype(float),(proto[good]=='17').astype(float)]
        for col,v in enumerate(values):state['numeric'][:,side*9+col]+=np.bincount(rows,weights=v,minlength=4*n)
        tokens=np.char.add('o:' if side==0 else 'i:',peers[good].astype(str))
        cc=bucket(tokens,8192)
        state['peers']+=sp.csr_matrix((np.ones(len(rows)),(rows,cc)),shape=(4*n,8192))
        token=np.char.add('o:' if side==0 else 'i:',service.to_numpy()[good].astype(str))
        # Совместные peer/service токены занимают отдельную область хеширования.
        cc=bucket(token,1024)
        joint=np.char.add(np.char.add(tokens,'|'),service.to_numpy()[good].astype(str))
        jj=bucket(joint,3072)+1024
        state['services']+=sp.csr_matrix((np.ones(2*len(rows)),(np.r_[rows,rows],np.r_[cc,jj])),shape=(4*n,4096))
        pp=bucket(np.char.add('o:' if side==0 else 'i:',proto[good]),128)
        state['protocols']+=sp.csr_matrix((np.ones(len(rows)),(rows,pp)),shape=(4*n,128))
    state['discarded_quarter_straddles']+=int((~quarter).sum())


def extract(window,hosts):
    n=len(hosts);state={k:sp.csr_matrix(shape,dtype=np.float64) for k,shape in
        [('counts',(n,n)),('bytes',(n,n)),('peers',(4*n,8192)),('services',(4*n,4096)),('protocols',(4*n,128))]}
    state.update(quarter_counts=[sp.csr_matrix((n,n),dtype=float) for _ in range(4)],quarter_bytes=[sp.csr_matrix((n,n),dtype=float) for _ in range(4)])
    state.update(numeric=np.zeros((4*n,18)),events=np.zeros(n),contacts=[Counter() for _ in hosts],discarded_quarter_straddles=0)
    columns=['src','dst','proto','dst_port','ts','duration','bytes','resp_bytes','packets','resp_packets']
    for i,f in enumerate(pd.read_csv(window['flows'],usecols=columns,dtype={c:str for c in ['src','dst','proto','dst_port']},chunksize=250000)):
        add_chunk(state,f,pd.Index(hosts),window['start'],window['end'])
        if (i+1)%20==0:print(window['name'],'chunks',i+1,flush=True)
    state.update(hosts=list(hosts),window=window,flow_sha256=sha(window['flows']),schema=SCHEMA)
    return state


def prepare(window,hosts,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/(window['name']+'.joblib')
    if path.exists():
        state=joblib.load(path)
        if state['window']!=window or state['hosts']!=list(hosts) or state['schema']!=SCHEMA or state['flow_sha256']!=sha(window['flows']):raise ValueError('cached features differ')
        return path
    state=extract(window,hosts);joblib.dump(state,path,compress=3)
    print('prepared',window['name'],'active',int((state['events']>=300).sum()),flush=True);return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--part',choices=['train','development','benchmark'],required=True)
    a=p.parse_args();c=load_config(a.config);hosts=pd.read_csv(c['cohort'],dtype=str).host_id.tolist()
    if len(hosts)!=len(set(hosts)):raise ValueError('duplicate cohort')
    windows=c['train'] if a.part=='train' else ([c['development']] if a.part=='development' else c['benchmark'])
    for w in windows:prepare(w,hosts,c['feature_dir'])


if __name__=='__main__':main()
