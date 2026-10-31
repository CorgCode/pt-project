import cloudpickle
import json
import numpy as np
import scipy.sparse as sp
import torch

from experiments.multiday.transfer import PartnerPair, sparse_partner_scores, train


def pair():
    source=sp.eye(4,format='csr',dtype=np.float32)
    future=source[[1,0,3,2]]
    return PartnerPair(source,future,['a','b','c','d'],['p','q','r','s'])


def test_partner_sampling_excludes_old_true_and_self_contacts():
    p=pair(); rng=np.random.default_rng(7)
    for _ in range(5):
        queries,candidates,unique,inverse=p.sample(rng,batch=4,negatives=8)
        np.testing.assert_array_equal(unique[inverse],candidates)
        for q,row in zip(queries,candidates):
            assert p.new[q,row[0]]>0
            assert p.source[q,row[0]]==0
            assert all(p.source[q,n]==0 and p.following[q,n]==0 for n in row[1:])


def test_probability_normalization_handles_realistic_count_dynamic_range():
    rng=np.random.default_rng(18)
    source=sp.csr_matrix(rng.lognormal(7,4,size=(6,200)).astype(np.float32))
    source.data[source.data<3000]=0; source.eliminate_zeros()
    future=sp.csr_matrix(np.ones((6,200),dtype=np.float32))
    p=PartnerPair(source,future,[f'h{i}' for i in range(6)],[f'p{i}' for i in range(200)])
    assert p.distribution.dtype==np.float64
    assert abs(p.distribution.sum()-1)<1e-14
    # Проверяется строгая проверка суммы самим NumPy Generator.
    rng.choice(200,1000,p=p.distribution)


def test_sparse_scores_equal_dense_neighbor_transfer_and_backpropagate():
    torch.set_num_threads(1); torch.manual_seed(2)
    z=torch.nn.functional.normalize(torch.randn(4,5),dim=1).requires_grad_()
    queries=np.array([0,2]); partners=np.array([0,1,3])
    source=pair().source
    result=sparse_partner_scores(z,queries,source,partners)
    logits=z[queries]@z.T/0.1
    logits[torch.arange(2),torch.tensor(queries)]=-1e9
    expected=torch.softmax(logits,dim=1)@torch.tensor(source[:,partners].toarray())
    torch.testing.assert_close(result,expected)
    result[:,1].sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0


def test_training_can_improve_actual_new_partner_scores():
    torch.set_num_threads(1); torch.manual_seed(3)
    net=torch.nn.Linear(4,4)
    optimizer=torch.optim.Adam(net.parameters(),lr=0.05)
    source=sp.eye(4,format='csr',dtype=np.float32)
    queries=np.arange(4)
    candidates=torch.tensor([[1,2,3],[0,2,3],[3,0,1],[2,0,1]])
    def loss():
        z=torch.nn.functional.normalize(net(torch.eye(4)),dim=1)
        values=sparse_partner_scores(z,queries,source,np.arange(4))
        return torch.nn.functional.cross_entropy(values.gather(1,candidates).clamp_min(1e-30).log(),torch.zeros(4,dtype=torch.long))
    before=float(loss().detach())
    for _ in range(20):
        value=loss(); optimizer.zero_grad(); value.backward(); optimizer.step()
    assert float(loss().detach())<before*0.25


def test_full_small_training_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setenv('PT_TORCH_THREADS','1')
    rng=np.random.default_rng(7)
    np.save(tmp_path/'features.npy',rng.normal(size=(3,4,6)).astype(np.float32))
    (tmp_path/'hosts.json').write_text(json.dumps(['a','b','c','d']))
    (tmp_path/'vocabulary.json').write_text(json.dumps(['p','q','r','s']))
    p=pair()
    for i in range(2):
        sp.save_npz(tmp_path/f'source{i}.npz',p.source)
        sp.save_npz(tmp_path/f'future{i}.npz',p.following)
        np.save(tmp_path/f'active{i}.npy',np.ones(4,dtype=bool))
    train(tmp_path,0.25,epochs=2,checkpoints=(2,))
    with (tmp_path/'donor025/epoch2.pkl').open('rb') as stream:
        head=cloudpickle.load(stream)
    assert np.isfinite(head.encode(rng.normal(size=(4,6)))).all()
