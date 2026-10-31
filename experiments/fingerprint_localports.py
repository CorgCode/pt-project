"""Стабильные локальные порты обоих концов, включая анонимные src_port."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection
from threadpoolctl import threadpool_limits

from experiments.fingerprint_search import BASE,SOURCE,write_json


WIDTH=131072


def local_counts(frame,hosts,width=WIDTH):
    frame=frame.loc[frame.src!=frame.dst]
    index=pd.Index(hosts);parts=[]
    for side,end,port in [('o','src','src_port'),('i','dst','dst_port')]:
        rows=index.get_indexer(frame[end]);keep=(rows>=0)&frame[port].notna().to_numpy()
        tokens=(side+':'+frame.loc[keep,'proto'].astype(str)+'/'+frame.loc[keep,port].astype(str)).to_numpy(dtype=object)
        cols=(pd.util.hash_array(tokens,hash_key='0123456789123456')%width).astype(np.int32)
        parts.append(sp.csr_matrix((np.ones(keep.sum(),dtype=np.float32),(rows[keep],cols)),shape=(len(hosts),width)))
    return (parts[0]+parts[1]).tocsr()


def retain_frequent(matrix,minimum=10,fraction=0.002,topk=64):
    value=matrix.tocsr(copy=True)
    mass=np.asarray(value.sum(1)).ravel()
    for row in range(value.shape[0]):
        start,end=value.indptr[row:row+2];data=value.data[start:end]
        data[data<max(minimum,mass[row]*fraction)]=0
        nonzero=np.flatnonzero(data)
        if len(nonzero)>topk:
            kept=nonzero[np.argpartition(data[nonzero],-topk)[-topk:]]
            rejected=np.ones(len(data),bool);rejected[kept]=False;data[rejected]=0
    value.eliminate_zeros()
    return value


def read_local(path,hosts,cache):
    path=Path(path);stat=path.stat();cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    signature=(str(path.resolve()),stat.st_size,stat.st_mtime_ns,tuple(hosts),
               hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key=hashlib.sha256(repr(signature).encode()).hexdigest();saved=cache/f'{key}.npz'
    if saved.exists():return sp.load_npz(saved)
    total=sp.csr_matrix((len(hosts),WIDTH),dtype=np.float32)
    for chunk in pd.read_csv(path,usecols=['src','dst','proto','src_port','dst_port'],dtype=str,chunksize=250000):
        total+=local_counts(chunk,hosts)
    result=retain_frequent(total)
    temporary=saved.with_name(saved.stem+'.tmp.npz');sp.save_npz(temporary,result);temporary.replace(saved)
    return result


def pilot(work):
    history=SOURCE/'fingerprint_aligned'
    hosts=json.loads((history/'hosts.json').read_text())
    a=read_local(BASE/'flows/train.csv',hosts,work/'cache')
    b=read_local(SOURCE/'multiday_gain/day09/flows/test.csv',hosts,work/'cache')
    tfidf=TfidfTransformer(sublinear_tf=True).fit(a)
    seen=np.asarray(a.getnnz(axis=0)).ravel()>0
    aa=tfidf.transform(a);bb=normalize(tfidf.transform(b).multiply(seen[None,:]))
    projection=SparseRandomProjection(n_components=128,dense_output=True,random_state=73).fit(aa)
    pa=normalize(projection.transform(aa)).astype(np.float32)
    pb=normalize(projection.transform(bb)).astype(np.float32)
    active=np.load(history/'future_active09.npy')
    anchor=normalize(np.load(history/'anchor.npy')[active])
    later=normalize(np.load(history/'historical09.npy')[active])
    coverage=float((np.diff(b.indptr)[active]>0).mean())
    records=[]
    for weight in [0.,0.25,0.5,1.]:
        src=normalize(np.hstack([anchor,pa[active]*weight]))
        dst=normalize(np.hstack([later,pb[active]*weight]))
        sim=src@dst.T
        records.append({'weight':weight,'historical09_raw_identity':float((sim.argmax(1)==np.arange(len(src))).mean())})
    write_json(work/'pilot.json',{'historical_only':True,'day09_query_coverage':coverage,
        'source_nnz':a.nnz,'later_nnz':b.nnz,'token_hash_collisions_possible':True,'results':records})
    print(json.dumps(json.loads((work/'pilot.json').read_text()),ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--work',required=True,type=Path)
    args=ap.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(2):pilot(args.work)


if __name__=='__main__':main()
