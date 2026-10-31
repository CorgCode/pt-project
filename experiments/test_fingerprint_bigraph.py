import cloudpickle
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pytest
from sklearn.preprocessing import normalize

from experiments.fingerprint_bigraph import (BipartiteEncoder,TokenBank,tensor_sparse,train,contact_scores,validate_revision)
from experiments.fingerprint_memory import fingerprint_counts


def traffic():
    rows=[]
    for h,port in [('a','443'),('b','80'),('c','443'),('d','80')]:
        rows += [{'src':h,'dst':'peer_'+h,'proto':'6','dst_port':port}]*30
    return pd.DataFrame(rows)


def test_token_bank_freezes_dictionary_and_ignores_untrained_buckets():
    hosts=['a','b','c','d']
    counts=fingerprint_counts(traffic(),hosts)
    peers=sorted(counts['counts']['peers'].index.get_level_values('t').unique())
    bank=TokenBank(hosts,peers)
    bank.fit(bank.raw(counts))
    expected=bank.transform(counts)
    np.testing.assert_allclose(np.asarray(expected.multiply(expected).sum(1)).ravel(),1,atol=1e-6)
    before=bank.tfidf.idf_.copy()
    future=traffic().replace({'peer_a':'brand_new_peer'})
    result=bank.transform(fingerprint_counts(future,hosts))
    assert result.shape==expected.shape
    assert list(bank.peers)==peers
    np.testing.assert_array_equal(before,bank.tfidf.idf_)
    assert result[:,~bank.seen].nnz==0


def small_graph():
    raw=sp.csr_matrix([[1.,0,1,0,0,0,0,0],[0,1,0,1,0,0,0,0],
                       [1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0]],dtype=np.float32)
    raw=normalize(raw)
    graph=normalize(raw[:,:6],norm='l1')
    reverse=normalize(graph.T.tocsr(),norm='l1')
    return raw,graph,reverse


def test_graph_query_follows_features_not_host_position():
    torch.set_num_threads(1);torch.manual_seed(3)
    net=BipartiteEncoder(8).eval()
    raw,graph,reverse=small_graph()
    aux=torch.nn.functional.normalize(torch.randn(4,128),dim=1)
    with torch.no_grad():
        source,context=net.prototype(tensor_sparse(raw),tensor_sparse(graph),tensor_sparse(reverse),aux)
        expected=net.observed(tensor_sparse(raw),tensor_sparse(graph),aux,context)
        order=[3,0,2,1]
        shuffled=net.observed(tensor_sparse(raw[order]),tensor_sparse(graph[order]),aux[order],context)
        clones=net.observed(tensor_sparse(raw[[0,0]]),tensor_sparse(graph[[0,0]]),aux[[0,0]],context)
    torch.testing.assert_close(shuffled,expected[order])
    torch.testing.assert_close(clones[0],clones[1])


def test_core_token_table_receives_gradient_and_changes():
    torch.set_num_threads(1);torch.manual_seed(4)
    net=BipartiteEncoder(8)
    raw,graph,reverse=small_graph()
    aux=torch.nn.functional.normalize(torch.randn(4,128),dim=1)
    before=net.embedding.detach().clone()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
    source,context=net.prototype(tensor_sparse(raw),tensor_sparse(graph),tensor_sparse(reverse),aux)
    query=net.observed(tensor_sparse(raw),tensor_sparse(graph),aux,context)
    target=torch.tensor([1,0,3,2])
    loss=torch.nn.functional.cross_entropy(source@query.T/0.1,target)
    loss.backward()
    assert net.embedding.grad is not None and net.embedding.grad.abs().sum()>0
    assert torch.isfinite(net.embedding.grad).all()
    optimizer.step()
    assert not torch.allclose(before,net.embedding.detach())
    assert net.role[-1].weight.detach().abs().sum()>0


def test_hard_transfer_uses_positive_cosine_and_excludes_inactive_donors():
    z=torch.nn.functional.normalize(torch.tensor([[1.,0],[0.9,0.1],[0.8,0.2],[0.7,0.3],[1.,0]]),dim=1)
    source=sp.csr_matrix(np.vstack([np.eye(4),np.zeros((1,4))]).astype(np.float32))
    actual=contact_scores(z,np.array([0]),source,np.arange(4),hard_mix=1).numpy()
    weights=(z[0]@z[1:4].T).numpy();weights/=weights.sum()
    np.testing.assert_allclose(actual,[[0,*weights]],atol=1e-6)


def test_training_rejects_changed_code_revision(tmp_path):
    (tmp_path/'training_revision.json').write_text(json.dumps({'sources':{},'torch_version':str(torch.__version__)}))
    with pytest.raises(RuntimeError,match='Код/версия'):
        validate_revision(tmp_path)


def test_full_graph_training_writes_native_encoder_checkpoint(tmp_path,monkeypatch):
    import experiments.fingerprint_bigraph as module
    monkeypatch.setenv('PT_TORCH_THREADS','1')
    aligned=tmp_path/'aligned';aligned.mkdir()
    transfer=tmp_path/'transfer';transfer.mkdir()
    monkeypatch.setattr(module,'ALIGNED',aligned);monkeypatch.setattr(module,'TRANSFER',transfer)
    raw,graph,reverse=small_graph()
    bank=SimpleNamespace(hosts=['a','b','c','d'],prototype=raw,width=8,graph_width=6)
    with (tmp_path/'bank.pkl').open('wb') as stream:cloudpickle.dump(bank,stream)
    rng=np.random.default_rng(4)
    aux=normalize(rng.normal(size=(4,128))).astype(np.float32)
    np.save(tmp_path/'source_aux.npy',aux)
    source=sp.eye(4,format='csr',dtype=np.float32)
    sp.save_npz(transfer/'source0.npz',source)
    (transfer/'vocabulary.json').write_text(json.dumps(['p','q','r','s']))
    for day in range(3,10):
        sp.save_npz(tmp_path/f'view{day:02d}_test.npz',raw)
        np.save(tmp_path/f'aux{day:02d}.npy',aux)
        np.save(aligned/f'future_active{day:02d}.npy',np.ones(4,dtype=bool))
        sp.save_npz(aligned/f'future{day:02d}.npz',source[[1,0,3,2]])
    train(tmp_path,0.5,epochs=2,checkpoints=(2,),warmup=0)
    with (tmp_path/'graph_rank05/epoch2.pkl').open('rb') as stream:net=cloudpickle.load(stream)
    assert isinstance(net,BipartiteEncoder)
    assert net.embedding.shape==(8,128)
    assert torch.isfinite(net.embedding).all()
    assert (tmp_path/'graph_rank05/complete.json').exists()
