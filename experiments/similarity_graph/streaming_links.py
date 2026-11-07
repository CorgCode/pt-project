"""Memory-bounded evaluation of the SAME C=1 logistic objective as future_links.

All features, source StandardScaler, random baseline, windows and labels are unchanged.
Only materialization of [abs(a-b),a*b] is replaced by batched loss/gradient evaluation.
L-BFGS-B uses sklearn's tolerances; small numeric optimizer differences are measured in tests.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from evaluation.future_links import _window_bounds,_seen,split_by_history,popularity_score
from evaluation.uncertainty import auc_with_ci,host_index


def fit_logistic(z,a,b,y,max_iter=2000,batch=512,materialize_threshold=50_000_000):
    n,d=len(y),z.shape[1]
    y=np.asarray(y,float)
    if n*d*2>materialize_threshold:
        # One bounded construction, then the actual sklearn solver reads a contiguous
        # memory map. Avoid re-gathering all endpoint vectors on every L-BFGS step.
        from tempfile import TemporaryDirectory
        from sklearn.linear_model import LogisticRegression
        print('Links memory-mapped sklearn fit',n,d,flush=True)
        with TemporaryDirectory(prefix='pt-similarity-links-') as directory:
            x=np.lib.format.open_memmap(Path(directory)/'features.npy',mode='w+',dtype=np.float64,shape=(n,2*d))
            for start in range(0,n,batch):
                stop=min(start+batch,n);left,right=z[a[start:stop]],z[b[start:stop]]
                x[start:stop,:d]=np.abs(left-right);x[start:stop,d:]=left*right
            x.flush()
            clf=LogisticRegression(C=1.,max_iter=max_iter,random_state=0).fit(x,y.astype(int))
            if int(clf.n_iter_[0])>=max_iter:raise RuntimeError('Links sklearn did not converge')
            weights=np.r_[clf.coef_[0],clf.intercept_[0]]
            iterations=int(clf.n_iter_[0]);del clf
            x._mmap.close();del x
        return weights,{'n_iter':iterations,'converged':True,'backend':'stock sklearn on batched float64 memmap'}
    def objective(w):
        grad=np.zeros_like(w);loss=0.
        for start in range(0,n,batch):
            stop=min(start+batch,n);left,right=z[a[start:stop]],z[b[start:stop]]
            absolute=np.abs(left-right);product=left*right
            logits=absolute@w[:d]+product@w[d:2*d]+w[-1]
            err=(expit(logits)-y[start:stop])/n
            loss+=np.sum(np.logaddexp(0,logits)-y[start:stop]*logits)/n
            grad[:d]+=absolute.T@err;grad[d:2*d]+=product.T@err;grad[-1]+=err.sum()
        loss+=.5*(w[:-1]@w[:-1])/n;grad[:-1]+=w[:-1]/n
        return loss,grad
    result=minimize(objective,np.zeros(2*d+1),jac=True,method='L-BFGS-B',
                    options={'maxiter':max_iter,'gtol':1e-4,'ftol':64*np.finfo(float).eps,'maxls':50})
    if not result.success:raise RuntimeError(f'Links optimization did not converge: {result.message}')
    return result.x,{'n_iter':result.nit,'converged':True,'loss':float(result.fun),'backend':'batched exact L-BFGS objective'}


def predict(z,a,b,weights,batch=512):
    d=z.shape[1];out=[]
    for start in range(0,len(a),batch):
        left,right=z[a[start:start+batch]],z[b[start:start+batch]]
        logits=np.abs(left-right)@weights[:d]+(left*right)@weights[d:2*d]+weights[-1]
        out.append(expit(logits))
    return np.concatenate(out)


def score_learned(embedding,fit_edges,eval_edges,*,fit_split='validation',eval_split='test',seed=0,
                  n_boot=0,degree=None,cache=None):
    fit=fit_edges[fit_edges.split==fit_split];ev=eval_edges[eval_edges.split==eval_split]
    if fit.empty or ev.empty:raise ValueError('empty fit or evaluation candidates')
    fw,ew=fit.iloc[0],ev.iloc[0]
    if fw.source_window!=ew.source_window:raise ValueError('different source windows')
    if _window_bounds(fw.target_window)[1]>_window_bounds(ew.target_window)[0]:raise ValueError('fit overlaps evaluation')
    index=embedding.index
    a,b=index.get_indexer(fit.host_a),index.get_indexer(fit.host_b)
    c,d=index.get_indexer(ev.host_a),index.get_indexer(ev.host_b)
    if min(a.min(),b.min(),c.min(),d.min())<0:raise ValueError('missing candidate hosts')
    yf,ye=fit.label.to_numpy(int),ev.label.to_numpy(int)
    if len(np.unique(yf))<2 or len(np.unique(ye))<2:raise ValueError('two label classes required')
    z=StandardScaler().fit_transform(embedding.to_numpy(float))
    true_key=joblib.hash((index.tolist(),embedding.to_numpy(),fit,seed,'actual-C1-v1'))
    true_saved=Path(cache)/f'links-actual-{true_key}.joblib' if cache else None
    if true_saved and true_saved.exists():weights,info=joblib.load(true_saved)
    else:
        weights,info=fit_logistic(z,a,b,yf)
        if true_saved:
            true_saved.parent.mkdir(parents=True,exist_ok=True);joblib.dump((weights,info),true_saved)
    scores=predict(z,c,d,weights);del z
    # Exactly the standard unscaled seeded random vectors, shared across methods of this dimension.
    random_z=np.random.default_rng(seed).standard_normal(embedding.shape)
    key=joblib.hash((index.tolist(),embedding.shape,seed,fit))
    saved=Path(cache)/f'links-random-{key}.joblib' if cache else None
    if saved and saved.exists():random_weights,random_info=joblib.load(saved)
    else:
        random_weights,random_info=fit_logistic(random_z,a,b,yf)
        if saved:saved.parent.mkdir(parents=True,exist_ok=True);joblib.dump((random_weights,random_info),saved)
    random_score=predict(random_z,c,d,random_weights)
    out=auc_with_ci(ye,scores,n_boot=n_boot,seed=seed,hosts=host_index(ev.host_a,ev.host_b))
    out['random_roc_auc']=float(roc_auc_score(ye,random_score))
    seen=_seen(ev)
    if seen is not None:
        out['by_history']=split_by_history(ye,scores,seen)
        out['by_history']['random']=split_by_history(ye,random_score,seen)
        history=seen.astype(float)
        out['history_roc_auc']=float(roc_auc_score(ye,history))
    if degree is not None:
        pop=popularity_score(ev,degree)
        out['popularity_roc_auc']=float(roc_auc_score(ye,pop))
        if seen is not None:out['by_history']['popularity']=split_by_history(ye,pop,seen)
    out.update(n_fit=len(fit),n_eval=len(ev),fit_split=fit_split,eval_split=eval_split,
               fit_target_window=fw.target_window,eval_target_window=ew.target_window,
               pr_auc_lift=out['pr_auc']/float(ye.mean()),
               optimizer=info,random_optimizer=random_info,
               feature_materialization='batched exact float64 loss/gradient, C=1')
    return out
